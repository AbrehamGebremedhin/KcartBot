"""LangChain-based asynchronous conversational agent for KCartBot."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
	AIMessage,
	BaseMessage,
	HumanMessage,
	SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

from app.services.lllm_service import LLMService
from app.tools import (
	AnalyticsDataTool,
	DataAccessTool,
	FlashSaleTool,
	ImageGeneratorTool,
	IntentClassifierTool,
	VectorSearchTool,
)
from app.tools.intent_classifier import CLASSIFIER_SYSTEM_PROMPT
from app.tools.base import ToolBase


AGENT_SYSTEM_PROMPT = """
You are the orchestration layer for KCartBot. Coordinate between customers and suppliers by:
1. Calling the `intent_classifier` tool FIRST for every new user input.
2. Choosing downstream tools (`vector_search`, `data_access`, `analytics_data`, `flash_sale_manager`, `image_generator`) based on the intent outcome.
3. Synthesising concise, friendly replies (1-3 sentences) that confirm critical information.

Guidelines:
- Customer flow: support onboarding, advisory (using retrieved context), ordering, logistics, and payment confirmation.
- Supplier flow: support onboarding, inventory management, scheduling, pricing insights, flash sale decisions, and imagery.
- For advisory questions, fetch context via `vector_search` before answering.
- Always reflect missing slots back to the user to gather required details.
- Confirm actions (orders, price changes, flash sales) before finalising.
- Use `flash_sale_manager` to surface proposals, accept, or decline flash sale offers.
- Mention next logical step when helpful, but avoid overwhelming lists.
- If the classifier returns `intent.unknown`, ask the user for clarification.

Session context (if any) is provided as JSON to keep track of active orders, user type, or preferences. Incorporate it when deciding what to do next.
"""


def _ensure_event_loop() -> asyncio.AbstractEventLoop:
	"""Return running event loop or create a new one for sync fallbacks."""
	try:
		return asyncio.get_running_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		return loop


class LLMServiceChatModel(BaseChatModel):
	"""Adapter that exposes :class:`LLMService` as a LangChain chat model."""

	def __init__(self, llm_service: LLMService) -> None:
		super().__init__()
		self._service = llm_service
		self._base_system_prompt = llm_service.system_prompt

	@property
	def _llm_type(self) -> str:  # noqa: D401 - LangChain interface
		return "llm_service_chat_model"

	def _convert_messages(
		self, messages: Sequence[BaseMessage]
	) -> tuple[List[Dict[str, str]], str, str]:
		system_parts: List[str] = []
		history: List[Dict[str, str]] = []
		prompt_content = ""

		for message in messages:
			if isinstance(message, SystemMessage):
				system_parts.append(message.content)
			elif isinstance(message, HumanMessage):
				history.append({"role": "user", "content": message.content})
			elif isinstance(message, AIMessage):
				history.append({"role": "assistant", "content": message.content})

		# Extract latest user input as prompt
		for idx in range(len(history) - 1, -1, -1):
			if history[idx]["role"] == "user":
				prompt_content = history[idx]["content"]
				# Remove the last user-entry from history so it's treated as prompt
				del history[idx]
				break

		combined_system = "\n".join(part.strip() for part in system_parts if part.strip())
		return history, prompt_content, combined_system

	async def _agenerate(
		self,
		messages: Sequence[BaseMessage],
		stop: Optional[List[str]] = None,
		**kwargs: Any,
	) -> ChatResult:
		history, prompt, dynamic_system = self._convert_messages(messages)
		original_system = self._service.system_prompt
		if dynamic_system:
			updated_prompt = f"{self._base_system_prompt}\n\n{dynamic_system}"
			self._service.update_system_prompt(updated_prompt)
		try:
			response_text = await self._service.acomplete(prompt, history=history, **kwargs)
		finally:
			self._service.update_system_prompt(original_system)

		if response_text and stop:
			for token in stop:
				if token in response_text:
					response_text = response_text.split(token)[0]
					break
		elif not response_text:
			response_text = ""

		message = AIMessage(content=response_text)
		generation = ChatGeneration(message=message, text=response_text)
		return ChatResult(generations=[generation])

	def _generate(
		self,
		messages: Sequence[BaseMessage],
		stop: Optional[List[str]] = None,
		**kwargs: Any,
	) -> ChatResult:
		loop = _ensure_event_loop()
		if loop.is_running():
			raise RuntimeError("Synchronous generation is not supported while an event loop is running. Use 'ainvoke'.")
		return loop.run_until_complete(self._agenerate(messages, stop=stop, **kwargs))


class LangChainToolAdapter(BaseTool):
	"""Wrap :class:`ToolBase` instances for LangChain agents."""

	def __init__(
		self,
		tool: ToolBase,
		context_getter: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
	) -> None:
		super().__init__(name=tool.name, description=tool.description)
		self._tool = tool
		self._get_context = context_getter

	def _run(self, tool_input: Any, **kwargs: Any) -> Any:
		loop = _ensure_event_loop()
		if loop.is_running():
			raise RuntimeError("Synchronous tool execution not supported inside a running event loop.")
		context = self._get_context() if self._get_context else None
		return loop.run_until_complete(self._tool.run(tool_input, context=context))

	async def _arun(self, tool_input: Any, **kwargs: Any) -> Any:
		context = self._get_context() if self._get_context else None
		return await self._tool.run(tool_input, context=context)

	async def aclose(self) -> None:
		closer = getattr(self._tool, "aclose", None)
		if closer:
			await closer()
			return
		sync_closer = getattr(self._tool, "close", None)
		if sync_closer:
			loop = asyncio.get_running_loop()
			await loop.run_in_executor(None, sync_closer)


def _serialise_history(history: Sequence[BaseMessage]) -> List[Dict[str, str]]:
	serialised: List[Dict[str, str]] = []
	for message in history:
		if isinstance(message, HumanMessage):
			serialised.append({"role": "user", "content": message.content})
		elif isinstance(message, AIMessage):
			serialised.append({"role": "assistant", "content": message.content})
		elif isinstance(message, SystemMessage):
			serialised.append({"role": "system", "content": message.content})
	return serialised


def _normalise_history(
	entries: Optional[Sequence[Any]],
) -> List[BaseMessage]:
	if not entries:
		return []
	normalised: List[BaseMessage] = []
	for item in entries:
		if isinstance(item, BaseMessage):
			normalised.append(item)
		elif isinstance(item, dict):
			role = (item.get("role") or "").lower()
			content = item.get("content") or ""
			if role == "assistant":
				normalised.append(AIMessage(content=content))
			elif role == "system":
				normalised.append(SystemMessage(content=content))
			else:
				normalised.append(HumanMessage(content=content))
		else:
			normalised.append(HumanMessage(content=str(item)))
	return normalised


def _extract_classifier_output(intermediate_steps: List[Any]) -> Optional[Dict[str, Any]]:
	for action, observation in intermediate_steps:
		if getattr(action, "tool", "") == "intent_classifier":
			if isinstance(observation, dict):
				return observation
			if isinstance(observation, str):
				try:
					return json.loads(observation)
				except json.JSONDecodeError:
					return {"raw": observation}
	return None


def _summarise_tool_calls(intermediate_steps: List[Any]) -> List[Dict[str, Any]]:
	summary: List[Dict[str, Any]] = []
	for action, observation in intermediate_steps:
		summary.append(
			{
				"tool": getattr(action, "tool", "unknown"),
				"input": getattr(action, "tool_input", None),
				"observation": observation,
			}
		)
	return summary


@dataclass
class AgentTurn:
	response: str
	intent: Optional[str]
	flow: Optional[str]
	classifier_output: Optional[Dict[str, Any]]
	tool_calls: List[Dict[str, Any]]
	history: List[BaseMessage]
	trace: Dict[str, Any]


class KcartAgent:
	"""High-level orchestrator that wires tools, LLM, and memory together."""

	def __init__(
		self,
		*,
		llm_service: Optional[LLMService] = None,
		extra_tools: Optional[Iterable[ToolBase]] = None,
	) -> None:
		self._service = llm_service or LLMService(system_prompt=AGENT_SYSTEM_PROMPT)
		self._llm = LLMServiceChatModel(self._service)
		self._context: Dict[str, Any] = {}
		# ReAct format guard message keeps retries user-friendly when parsing fails.
		self._parsing_error_message = (
			"I stumbled while planning that step. I'll restate my reasoning using the Thought/Action/Action Input format."
		)

		# Build tool suite
		classifier_service = self._service.clone(system_prompt=CLASSIFIER_SYSTEM_PROMPT)
		intent_tool = IntentClassifierTool(
			llm_service=classifier_service,
			system_prompt=CLASSIFIER_SYSTEM_PROMPT,
		)
		default_tools: List[ToolBase] = [
			intent_tool,
			VectorSearchTool(),
			DataAccessTool(),
			AnalyticsDataTool(),
			FlashSaleTool(),
			ImageGeneratorTool(),
		]
		if extra_tools:
			default_tools.extend(extra_tools)

		self._tool_adapters: List[LangChainToolAdapter] = [
			LangChainToolAdapter(tool, context_getter=self._get_context)
			for tool in default_tools
		]

		self._react_instructions = (
			"Follow the ReAct pattern exactly each turn:\n"
			"Thought: describe your reasoning\n"
			"Action: name of the next tool to call, or 'Final Answer' when you are ready to respond to the user\n"
			"Action Input: plain text describing the argument for the tool (or your final reply when Action is 'Final Answer')\n"
			"Observation: paste the tool result you just received. Skip Observation only when Action is 'Final Answer'."
		)

		self._prompt = ChatPromptTemplate.from_messages(
			[
				("system", AGENT_SYSTEM_PROMPT + "\n\n" + self._react_instructions),
				("system", "Session context (JSON): {session_context}"),
				(
					"system",
					"Available tools:\n{tools}\nRefer to them exactly by name when planning: {tool_names}",
				),
				MessagesPlaceholder(variable_name="chat_history"),
				("assistant", "{agent_scratchpad}"),
				("human", "{input}"),
			]
		)

		tool_descriptions = "\n".join(
			f"- {adapter.name}: {adapter.description}" for adapter in self._tool_adapters
		)
		tool_names = ", ".join(adapter.name for adapter in self._tool_adapters)
		self._prompt = self._prompt.partial(
			tools=tool_descriptions or "- No tools configured",
			tool_names=tool_names or "none",
		)

		self._executor = self._build_executor()

	def _build_executor(self) -> AgentExecutor:
		agent = create_react_agent(self._llm, self._tool_adapters, self._prompt)
		return AgentExecutor(
			agent=agent,
			tools=self._tool_adapters,
			return_intermediate_steps=True,
			handle_parsing_errors=self._parsing_error_message,
			max_iterations=30,
			early_stopping_method="force",
		)

	def _get_context(self) -> Dict[str, Any]:
		return self._context

	async def aclose(self) -> None:
		for adapter in self._tool_adapters:
			await adapter.aclose()

	def close(self) -> None:
		loop = _ensure_event_loop()
		if loop.is_running():
			raise RuntimeError("Synchronous close not supported inside a running event loop. Use 'await aclose()'.")
		loop.run_until_complete(self.aclose())

	async def ainvoke(
		self,
		user_input: str,
		*,
		chat_history: Optional[Sequence[Any]] = None,
		context: Optional[Dict[str, Any]] = None,
	) -> AgentTurn:
		normalised_history = _normalise_history(chat_history)
		self._context = context or {}
		session_context = json.dumps(self._context, ensure_ascii=False) if self._context else "{}"

		result = await self._executor.ainvoke(
			{
				"input": user_input,
				"chat_history": normalised_history,
				"session_context": session_context,
			}
		)

		tool_calls = _summarise_tool_calls(result.get("intermediate_steps", []))
		classifier_output = _extract_classifier_output(result.get("intermediate_steps", []))
		intent = classifier_output.get("intent") if classifier_output else None
		flow = classifier_output.get("flow") if classifier_output else None

		response_text = result.get("output") or ""
		if not response_text:
			response_text = (
				"I hit my planning limit on that step. Could you rephrase or give me a little more detail?"
			)

		augmented_history = list(normalised_history)
		augmented_history.append(HumanMessage(content=user_input))
		augmented_history.append(AIMessage(content=response_text))

		trace = {
			"raw": result,
			"history": _serialise_history(augmented_history),
		}

		return AgentTurn(
			response=response_text,
			intent=intent,
			flow=flow,
			classifier_output=classifier_output,
			tool_calls=tool_calls,
			history=augmented_history,
			trace=trace,
		)

	def invoke(
		self,
		user_input: str,
		*,
		chat_history: Optional[Sequence[Any]] = None,
		context: Optional[Dict[str, Any]] = None,
	) -> AgentTurn:
		loop = _ensure_event_loop()
		return loop.run_until_complete(
			self.ainvoke(user_input, chat_history=chat_history, context=context)
		)
