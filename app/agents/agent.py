"""LangChain-based asynchronous conversational agent for KCartBot."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from datetime import date
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
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
	ScheduleTool,
)
from app.tools.intent_classifier import CLASSIFIER_SYSTEM_PROMPT
from app.tools.base import ToolBase


AGENT_SYSTEM_PROMPT = """
You are the orchestration layer for KCartBot. Coordinate between customers and suppliers by:
1. Calling the `intent_classifier` tool FIRST for every new user input.
2. Using conversation history to understand context - if user says "okay", "yes", "sure" etc., treat it as confirmation of the previous assistant message.
3. Choosing downstream tools (`vector_search`, `data_access`, `analytics_data`, `flash_sale_manager`, `image_generator`) based on the intent outcome.
4. Synthesising concise, friendly replies (1-3 sentences) that confirm critical information.

Guidelines:
- Customer flow: support onboarding, advisory (using retrieved context), ordering, logistics, and payment confirmation.
- Supplier flow: support onboarding, inventory management, scheduling, pricing insights, flash sale decisions, and imagery.
- During supplier inventory intake, gather product name, quantity, price per unit, expiry date, and available delivery days. Infer category when obvious (e.g., tomatoes → vegetable) instead of asking the user to supply it explicitly. If expiry date or delivery schedule is missing, ask for it explicitly, accepting natural phrases like "all week", "weekends", or "next week" and convert them to specific availability before calling tools.
 - During supplier inventory intake, gather product name, quantity, price per unit, expiry date, and available delivery days. Ask for one missing detail at a time (e.g., request the expiry date, wait for the reply, then ask about delivery days). Infer category when obvious (e.g., tomatoes → vegetable) instead of asking the user to supply it explicitly. If expiry date or delivery schedule is missing, ask for it explicitly, accepting natural phrases like "all week", "weekends", or "next week" and convert them to specific availability before calling tools. When a schedule or expiry phrase is resolved, reuse that observation instead of re-calling `schedule_helper`.
- Currency handling: the marketplace operates in Ethiopian Birr (ETB). Present all monetary amounts in ETB, treat user mentions of other currencies as informational only, and ask for or perform ETB conversion rather than switching currencies.
- Units: Solid goods are tracked per kilogram (kg) and liquids per liter (liter). Keep conversations and inventory updates in those units unless converting between them for clarity.
- Treat structured data as primary: for pricing, inventory, sales history, or comparisons, call `data_access` or `analytics_data` first (e.g. fetch competitor prices for "best price" style questions). Only fall back to `vector_search` if the answer clearly depends on documentation or knowledge base snippets.
- Use `vector_search` for documentation or policy lookups when structured data is insufficient.
- When a tool returns a valid result, incorporate it into the reply rather than calling the same tool repeatedly unless new parameters are supplied.
- Call `intent_classifier` exactly once per user turn; reuse its output for the remainder of the turn. If the classifier reports missing slots (e.g. `product_name`), ask the user to provide them instead of re-calling the classifier.
- When planning pricing insights, always include a concrete `product_name` (or `product_id`) for analytics calls. Reuse the exact spelling from inventory records when known, and normalise simple plural/singular variants (e.g. "tomatoes" → "Tomato"). If no product is specified, consult the supplier's inventory or ask the user to choose one before calling pricing tools.
- Use `schedule_helper` to convert natural-language delivery schedules or relative expiry phrases ("next week", "this weekend") into normalized schedules or concrete dates before updating inventory or flash sales.
- Before calling `data_access` to create or update supplier inventory pricing, query `analytics_data` with `operation: "pricing_guidance"` for that product and share the competitor/historical recommendation so the supplier can confirm or adjust their price.
- When user confirms with "okay", "yes", "sure" etc., look at conversation history to understand what they're confirming and proceed accordingly.
- Always reflect missing slots back to the user to gather required details.
- Confirm actions (orders, price changes, flash sales) before finalising.
- Use `flash_sale_manager` to surface proposals, accept, or decline flash sale offers.
- Mention next logical step when helpful, but avoid overwhelming lists.
- If the classifier returns `intent.unknown` AND there's no clear context from history, ask the user for clarification.

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


class ForgivingReActParser(ReActSingleInputOutputParser):
	"""Custom ReAct parser that handles common LLM formatting mistakes."""

	def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
		"""Parse LLM output, fixing common formatting issues before standard parsing."""
		# Fix standalone "Final Answer" without "Action:" prefix
		text = re.sub(
			r'\n(Final Answer):\s*',
			r'\nAction: \1\nAction Input: ',
			text,
			flags=re.IGNORECASE
		)
		
		# If model outputs just "Final Answer" with input on same/next line, fix it
		text = re.sub(
			r'\n(Final Answer)\s+([^\n]+)',
			r'\nAction: \1\nAction Input: \2',
			text,
			flags=re.IGNORECASE
		)
		
		try:
			parsed = super().parse(text)
			if isinstance(parsed, AgentAction) and parsed.tool.strip().lower() == "final answer":
				output_text = parsed.tool_input if isinstance(parsed.tool_input, str) else str(parsed.tool_input)
				return AgentFinish(
					return_values={"output": output_text.strip()},
					log=text,
				)
			return parsed
		except Exception:
			# If parsing still fails, check if there's any final answer content
			final_answer_match = re.search(
				r'(?:Action:\s*)?Final Answer[:\s]+(.+)',
				text,
				re.IGNORECASE | re.DOTALL
			)
			if final_answer_match:
				return AgentFinish(
					return_values={"output": final_answer_match.group(1).strip()},
					log=text,
				)
			raise


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
		self._last_call_key: Optional[str] = None
		self._last_call_result: Any = None

	def _make_cache_key(self, tool_input: Any, context: Optional[Dict[str, Any]]) -> Optional[str]:
		try:
			if isinstance(tool_input, str):
				key = tool_input.strip()
			elif isinstance(tool_input, (dict, list, tuple)):
				key = json.dumps(tool_input, sort_keys=True)
			else:
				key = str(tool_input)
		except Exception:  # pragma: no cover - defensive
			return None
		return key or None

	def _clone_result(self, result: Any) -> Any:
		try:
			return copy.deepcopy(result)
		except Exception:  # pragma: no cover - defensive
			self._last_call_key = None
			self._last_call_result = None
			return result

	def _run(self, tool_input: Any, **kwargs: Any) -> Any:
		loop = _ensure_event_loop()
		if loop.is_running():
			raise RuntimeError("Synchronous tool execution not supported inside a running event loop.")
		context = self._get_context() if self._get_context else None
		cache_key = self._make_cache_key(tool_input, context)
		if cache_key and cache_key == self._last_call_key:
			cached = self._clone_result(self._last_call_result)
			if isinstance(cached, dict):
				cached.setdefault("_cached_result", True)
			return cached
		try:
			result = loop.run_until_complete(self._tool.run(tool_input, context=context))
		except Exception:
			self._last_call_key = None
			self._last_call_result = None
			raise
		if cache_key:
			self._last_call_key = cache_key
			self._last_call_result = self._clone_result(result)
		else:
			self._last_call_key = None
			self._last_call_result = None
		return result

	async def _arun(self, tool_input: Any, **kwargs: Any) -> Any:
		context = self._get_context() if self._get_context else None
		cache_key = self._make_cache_key(tool_input, context)
		if cache_key and cache_key == self._last_call_key:
			cached = self._clone_result(self._last_call_result)
			if isinstance(cached, dict):
				cached.setdefault("_cached_result", True)
			return cached
		try:
			result = await self._tool.run(tool_input, context=context)
		except Exception:
			self._last_call_key = None
			self._last_call_result = None
			raise
		if cache_key:
			self._last_call_key = cache_key
			self._last_call_result = self._clone_result(result)
		else:
			self._last_call_key = None
			self._last_call_result = None
		return result

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
			"I stumbled while planning that step. Please restate your reasoning using the Thought/Action/Action Input/Observation format, "
			"reuse the latest tool observations instead of calling the same tool again, and then provide the final answer."
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
			ScheduleTool(),
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
			"CRITICAL: To respond to the user, you MUST return output in this EXACT format:\n\n"
			"Thought: [your reasoning about what to do next]\n"
			"Action: [tool name from the available tools list]\n"
			"Action Input: [the input to pass to that tool]\n"
			"Observation: [the tool's response will appear here]\n\n"
			"Repeat Thought/Action/Action Input/Observation as many times as needed.\n\n"
			"When you have enough information to answer the user, use this EXACT format:\n"
			"Thought: I now have enough information to respond to the user.\n"
			"Action: Final Answer\n"
			"Action Input: [your complete response to the user]\n\n"
			"NEVER write just 'Final Answer' alone - it must be preceded by 'Action: ' on the same line.\n"
			"NEVER skip the 'Thought:' line before 'Action: Final Answer'."
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
		agent = create_react_agent(
			self._llm,
			self._tool_adapters,
			self._prompt,
			output_parser=ForgivingReActParser(),
		)
		return AgentExecutor(
			agent=agent,
			tools=self._tool_adapters,
			return_intermediate_steps=True,
			handle_parsing_errors=self._parsing_error_message,
			max_iterations=8,
			early_stopping_method="force",
		)

	def _get_context(self) -> Dict[str, Any]:
		return self._context

	@staticmethod
	def _format_decimal(value: Any) -> Optional[str]:
		if value is None:
			return None
		try:
			number = Decimal(str(value))
		except (InvalidOperation, ValueError, TypeError):
			return None
		return f"{number.quantize(Decimal('0.01')):.2f}"

	@staticmethod
	def _find_supplier_listing(
		tool_calls: Sequence[Dict[str, Any]],
		product_id: Optional[str],
	) -> Optional[Dict[str, Any]]:
		if not product_id:
			return None
		for call in reversed(tool_calls):
			if call.get("tool") != "data_access":
				continue
			observation = call.get("observation")
			if not isinstance(observation, dict):
				continue
			entries = observation.get("results")
			if not isinstance(entries, list):
				continue
			for entry in entries:
				if isinstance(entry, dict) and str(entry.get("product_id")) == str(product_id):
					return entry
		return None

	@staticmethod
	def _extract_supplier_price(listing: Optional[Dict[str, Any]]) -> Optional[Any]:
		if not listing:
			return None
		return listing.get("unit_price_etb") or listing.get("unit_price")

	@staticmethod
	def _parse_tool_input(raw_input: Any) -> Optional[Dict[str, Any]]:
		if raw_input is None:
			return None
		if isinstance(raw_input, dict):
			return raw_input
		if isinstance(raw_input, str):
			try:
				return json.loads(raw_input)
			except json.JSONDecodeError:
				return None
		return None

	@staticmethod
	def _extract_filter_value(
		tool_calls: Sequence[Dict[str, Any]],
		filter_key: str,
	) -> Optional[Any]:
		for call in tool_calls:
			if call.get("tool") != "data_access":
				continue
			parsed = KcartAgent._parse_tool_input(call.get("input"))
			if not isinstance(parsed, dict):
				continue
			filters = parsed.get("filters")
			if isinstance(filters, dict) and filter_key in filters:
				return filters.get(filter_key)
		return None

	def _build_pricing_guidance_fallback(
		self,
		tool_calls: List[Dict[str, Any]],
	) -> Optional[str]:
		for call in reversed(tool_calls):
			if call.get("tool") != "analytics_data":
				continue
			observation = call.get("observation")
			if not isinstance(observation, dict):
				continue
			if observation.get("recommended_price") is None:
				continue
			return self._format_pricing_message(tool_calls, observation)
		return None

	def _format_pricing_message(
		self,
		tool_calls: List[Dict[str, Any]],
		observation: Dict[str, Any],
	) -> Optional[str]:
		product_name = observation.get("product_name") or "this product"
		currency = observation.get("currency") or "ETB"
		unit = observation.get("unit") or ""
		unit_suffix = f" per {unit}" if unit else ""
		recommended = self._format_decimal(observation.get("recommended_price"))
		price_band = observation.get("price_band") or {}
		floor_price = self._format_decimal(price_band.get("floor"))
		ceiling_price = self._format_decimal(price_band.get("ceiling"))
		listing = self._find_supplier_listing(tool_calls, observation.get("product_id"))
		current_price_value = self._extract_supplier_price(listing)
		current_price = self._format_decimal(current_price_value)
		confidence = observation.get("confidence")

		if not recommended:
			return None

		sentences: List[str] = []
		if current_price:
			sentences.append(
				f"You're currently listing {product_name} at {current_price} {currency}{unit_suffix}."
			)
		else:
			sentences.append(f"Here's what I found for {product_name}.")

		band_clause = ""
		if floor_price and ceiling_price:
			band_clause = f" (range {floor_price}-{ceiling_price} {currency}{unit_suffix})"
		elif floor_price and not ceiling_price:
			band_clause = f" (floor {floor_price} {currency}{unit_suffix})"
		elif ceiling_price and not floor_price:
			band_clause = f" (ceiling {ceiling_price} {currency}{unit_suffix})"

		sentences.append(
			f"My pricing guidance recommends {recommended} {currency}{unit_suffix}{band_clause} based on current market data."
		)

		if confidence:
			sentences.append(f"Confidence: {confidence}.")

		sentences.append("Let me know if you'd like me to adjust the price or review another item.")
		return " ".join(sentences)

	@staticmethod
	def _extract_product_name(
		tool_calls: Sequence[Dict[str, Any]],
		product_id: Optional[str],
	) -> Optional[str]:
		if not product_id:
			return None
		for call in reversed(tool_calls):
			if call.get("tool") != "data_access":
				continue
			observation = call.get("observation")
			if isinstance(observation, dict):
				if str(observation.get("product_id")) == str(product_id):
					for key in ("product_name_en", "product_name", "name"):
						if observation.get(key):
							return str(observation[key])
					return None
				results = observation.get("results")
				if isinstance(results, list):
					for entry in results:
						if not isinstance(entry, dict):
							continue
						if str(entry.get("product_id")) == str(product_id):
							for key in ("product_name_en", "product_name", "name"):
								if entry.get(key):
									return str(entry[key])
		return None

	def _build_inventory_fallback(
		self,
		tool_calls: List[Dict[str, Any]],
	) -> Optional[str]:
		listing: Optional[Dict[str, Any]] = None
		for call in reversed(tool_calls):
			if call.get("tool") != "data_access":
				continue
			observation = call.get("observation")
			if not isinstance(observation, dict):
				continue
			results = observation.get("results")
			if isinstance(results, list) and results:
				first_entry = results[0]
				if isinstance(first_entry, dict):
					listing = first_entry
					break
		if not isinstance(listing, dict):
			return None

		quantity = listing.get("quantity_available")
		if quantity is None:
			return None
		unit_raw = str(listing.get("unit") or "").strip()
		unit_label = unit_raw
		if unit_label.lower().startswith("unittype."):
			unit_label = unit_label.split(".", 1)[-1].lower()
		if unit_label and unit_label.isupper():
			unit_label = unit_label.lower()
		if unit_label:
			unit_label = unit_label.lower()
		unit_phrase = f" {unit_label}" if unit_label else ""
		quantity_display = self._format_decimal(quantity) or str(quantity)

		product_label = listing.get("product_name")
		if not product_label:
			product_id = listing.get("product_id")
			product_label = self._extract_product_name(tool_calls, product_id)
		if not product_label:
			product_label = self._extract_filter_value(tool_calls, "product_name")
		label = str(product_label) if product_label else "that product"

		unit_price = self._extract_supplier_price(listing)
		unit_price_display = self._format_decimal(unit_price)
		currency = "ETB"
		price_clause = ""
		if unit_price_display:
			if unit_label:
				price_clause = f" at {unit_price_display} {currency} per {unit_label}"
			else:
				price_clause = f" at {unit_price_display} {currency}"

		status_note = ""
		if listing.get("is_expired"):
			status_note = " This lot is expired."
		else:
			effective_status = str(listing.get("effective_status") or listing.get("status") or "").strip()
			if effective_status and effective_status.lower() not in {"", "active"}:
				status_note = f" This lot is currently marked {effective_status.replace('_', ' ')}."

		message = (
			f"You currently have {quantity_display}{unit_phrase} of {label} in inventory{price_clause}."
		)
		message += status_note or ""
		message += " Need any help adjusting the stock or price?"
		return message

	def _build_schedule_followup(
		self,
		tool_calls: List[Dict[str, Any]],
	) -> Optional[str]:
		for call in reversed(tool_calls):
			if call.get("tool") != "schedule_helper":
				continue
			observation = call.get("observation")
			if not isinstance(observation, dict):
				continue
			resolved = observation.get("resolved_date")
			if not resolved:
				continue
			try:
				resolved_date = date.fromisoformat(str(resolved))
			except (ValueError, TypeError):
				resolved_date = None
			if resolved_date:
				pretty = resolved_date.strftime("%d %b %Y")
			else:
				pretty = str(resolved)
			notes = observation.get("notes") or []
			if "Interpreted" in " ".join(notes):
				return (
					f"I'll record the expiry date as {pretty}. Which delivery days should I note for this stock?"
				)
			return f"I'll record the expiry date as {pretty}. Could you share your delivery schedule?"
		return None

	def _build_fallback_reply(self, tool_calls: List[Dict[str, Any]]) -> Optional[str]:
		builders = (
			self._build_schedule_followup,
			self._build_pricing_guidance_fallback,
			self._build_inventory_fallback,
		)
		for builder in builders:
			text = builder(tool_calls)
			if text:
				return text
		return None

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
		
		# Build context with chat history for tools to access
		self._context = context or {}
		self._context["chat_history"] = _serialise_history(normalised_history)
		
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
		fallback_reply = None
		if response_text.strip() in {"", "Agent stopped due to iteration limit or time limit."}:
			fallback_reply = self._build_fallback_reply(tool_calls)
		if fallback_reply:
			response_text = fallback_reply
		elif not response_text:
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
