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

from app.services.llm_service import LLMService
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
5. Format every reply as a single line with no newline characters; join sentences into one paragraph.

Guidelines:
- Customer flow: support onboarding, advisory (using retrieved context), ordering, logistics, and payment confirmation.
- Supplier flow: support onboarding, inventory management, scheduling, pricing insights, flash sale decisions, handling/storage guidance, and imagery.
- During supplier inventory intake, gather product name, quantity, price per unit, expiry date, and available delivery days. Infer category when obvious (e.g., tomatoes → vegetable) instead of asking the user to supply it explicitly. If expiry date or delivery schedule is missing, ask for it explicitly, accepting natural phrases like "all week", "weekends", or "next week" and convert them to specific availability before calling tools.
 - During supplier inventory intake, gather product name, quantity, price per unit, expiry date, and available delivery days. Ask for one missing detail at a time (e.g., request the expiry date, wait for the reply, then ask about delivery days). Infer category when obvious (e.g., tomatoes → vegetable) instead of asking the user to supply it explicitly. If expiry date or delivery schedule is missing, ask for it explicitly, accepting natural phrases like "all week", "weekends", or "next week" and convert them to specific availability before calling tools. When a schedule or expiry phrase is resolved, reuse that observation instead of re-calling `schedule_helper`.
- Currency handling: the marketplace operates in Ethiopian Birr (ETB). Present all monetary amounts in ETB, treat user mentions of other currencies as informational only, and ask for or perform ETB conversion rather than switching currencies.
- Units: Solid goods are tracked per kilogram (kg) and liquids per liter (liter). Keep conversations and inventory updates in those units unless converting between them for clarity.
- Language support: Detect and comfortably engage in English, Amharic (Ge'ez script), and phonetic Amharic (Latin script). Default to English responses unless the user's latest message is entirely Amharic or phonetic Amharic, in which case reply in Amharic (optionally adding a concise English gloss if it aids clarity). Confirm before switching languages mid-conversation.
- Product name handling: Catalog entries carry English, Amharic, and phonetic Amharic variants. When matching user inputs (including minor spelling mistakes) to products or quoting names back, search across all variants, normalise simple misspellings, and respond using the language variant that matches the conversation.
- Treat structured data as primary: for pricing, inventory, sales history, or comparisons, call `data_access` or `analytics_data` first (e.g. fetch competitor prices for "best price" style questions). Only fall back to `vector_search` if the answer clearly depends on documentation or knowledge base snippets.
- Use `vector_search` for documentation or policy lookups when structured data is insufficient.
- When a supplier asks how to store, preserve, or handle a product, respond with storage guidance drawn from retrieved context or well-known best practices unless they explicitly request stock levels or status updates.
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

	@staticmethod
	def _normalise_unit_label(unit: Any) -> str:
		if not unit:
			return ""
		label = str(unit).strip()
		if not label:
			return ""
		if label.lower().startswith("unittype."):
			label = label.split(".", 1)[-1]
		label = label.replace("_", " ").strip()
		return label.lower()

	@staticmethod
	def _coerce_decimal(value: Any) -> Optional[Decimal]:
		if value is None:
			return None
		try:
			return Decimal(str(value))
		except (InvalidOperation, ValueError, TypeError):
			return None

	def _format_quantity_phrase(self, quantity: Optional[Any], unit: Optional[Any]) -> Optional[str]:
		amount = self._coerce_decimal(quantity)
		if amount is None:
			return None
		quantity_display = self._format_decimal(amount)
		if quantity_display and quantity_display.endswith(".00"):
			quantity_display = quantity_display[:-3]
		if quantity_display and "." in quantity_display:
			quantity_display = quantity_display.rstrip("0").rstrip(".") or quantity_display
		unit_label = self._normalise_unit_label(unit)
		if unit_label == "liter" and amount != Decimal("1"):
			unit_label = "liters"
		elif unit_label in {"kilogram", "kilograms"}:
			unit_label = "kg"
		elif not unit_label:
			unit_label = "units"
		return f"{quantity_display} {unit_label}"

	@staticmethod
	def _find_listing_by_inventory_id(
		tool_calls: Sequence[Dict[str, Any]],
		inventory_id: Optional[str],
	) -> Optional[Dict[str, Any]]:
		if not inventory_id:
			return None
		for call in reversed(tool_calls):
			if call.get("tool") != "data_access":
				continue
			observation = call.get("observation")
			if not isinstance(observation, dict):
				continue
			results = observation.get("results")
			if not isinstance(results, list):
				continue
			for entry in results:
				if isinstance(entry, dict) and str(entry.get("inventory_id")) == str(inventory_id):
					return entry
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
		listings: List[Dict[str, Any]] = []
		for call in reversed(tool_calls):
			if call.get("tool") != "data_access":
				continue
			observation = call.get("observation")
			if not isinstance(observation, dict):
				continue
			results = observation.get("results")
			if not isinstance(results, list):
				continue
			for entry in results:
				if isinstance(entry, dict):
					listings.append(entry)
			if listings:
				break
		if not listings:
			return None

		formatted_entries: List[str] = []
		for entry in listings:
			quantity = entry.get("quantity_available")
			if quantity is None:
				continue
			quantity_display = self._format_decimal(quantity) or str(quantity)
			unit_label = self._normalise_unit_label(entry.get("unit"))
			unit_phrase = f" {unit_label}" if unit_label else ""
			product_label = entry.get("product_name")
			if not product_label:
				product_label = self._extract_product_name(tool_calls, entry.get("product_id"))
			if not product_label:
				product_label = self._extract_filter_value(tool_calls, "product_name")
			label = str(product_label) if product_label else "Unnamed product"

			unit_price = self._extract_supplier_price(entry)
			unit_price_display = self._format_decimal(unit_price)
			per_clause = f" per {unit_label}" if unit_label else ""
			price_clause = ""
			if unit_price_display:
				price_clause = f" at {unit_price_display} ETB{per_clause}"

			status_bits: List[str] = []
			if entry.get("is_expired"):
				status_bits.append("expired")
			else:
				effective_status = str(entry.get("effective_status") or entry.get("status") or "").strip()
				if effective_status and effective_status.lower() not in {"", "active"}:
					status_bits.append(effective_status.replace("_", " ").lower())
			expiry_date = entry.get("expiry_date")
			if expiry_date:
				status_bits.append(f"expiry {expiry_date}")
			status_suffix = f" ({', '.join(status_bits)})" if status_bits else ""

			formatted_entries.append(
				f"{label}: {quantity_display}{unit_phrase}{price_clause}{status_suffix}"
			)

		if not formatted_entries:
			return None

		visible_entries = formatted_entries[:5]
		remaining = len(formatted_entries) - len(visible_entries)
		summary = "; ".join(visible_entries)
		if remaining > 0:
			remainder_phrase = f"; and {remaining} more item{'s' if remaining != 1 else ''}"
			summary += remainder_phrase

		return f"Inventory snapshot: {summary}. Need any help adjusting stock or prices?"

	def _build_inventory_deletion_confirmation(
		self,
		tool_calls: List[Dict[str, Any]],
	) -> Optional[str]:
		for call in reversed(tool_calls):
			if call.get("tool") != "data_access":
				continue
			observation = call.get("observation")
			message: Optional[str] = None
			if isinstance(observation, dict):
				message = observation.get("message")
			elif isinstance(observation, str):
				message = observation
			if not message or "deleted" not in message.lower():
				continue
			parsed_input = self._parse_tool_input(call.get("input"))
			inventory_id: Optional[str] = None
			product_hint: Optional[str] = None
			if isinstance(parsed_input, dict):
				inventory_id = parsed_input.get("id") or parsed_input.get("inventory_id")
				filters = parsed_input.get("filters")
				if isinstance(filters, dict):
					inventory_id = inventory_id or filters.get("inventory_id")
					product_hint = filters.get("product_name") or filters.get("name")
			listing = self._find_listing_by_inventory_id(tool_calls, inventory_id)
			label = None
			if isinstance(listing, dict):
				label = listing.get("product_name") or listing.get("name")
			if not label:
				label = product_hint or "that item"
			quantity_display = None
			unit_phrase = ""
			price_clause = ""
			status_bits: List[str] = []
			expiry_note = None
			if isinstance(listing, dict):
				quantity_display = self._format_decimal(listing.get("quantity_available"))
				if not quantity_display and listing.get("quantity_available") is not None:
					quantity_display = str(listing.get("quantity_available"))
				unit_label = self._normalise_unit_label(listing.get("unit"))
				if unit_label:
					unit_phrase = f" {unit_label}"
				unit_price = self._extract_supplier_price(listing)
				unit_price_display = self._format_decimal(unit_price)
				if unit_price_display:
					per_clause = f" per {unit_label}" if unit_label else ""
					price_clause = f" at {unit_price_display} ETB{per_clause}"
				if listing.get("is_expired"):
					status_bits.append("expired")
				else:
					effective_status = str(listing.get("effective_status") or listing.get("status") or "").strip()
					if effective_status and effective_status.lower() not in {"", "active"}:
						status_bits.append(effective_status.replace("_", " ").lower())
				expiry_note = listing.get("expiry_date")
			if expiry_note:
				status_bits.append(f"expiry {expiry_note}")
			status_suffix = f" ({', '.join(status_bits)})" if status_bits else ""
			quantity_clause = f" {quantity_display}{unit_phrase}" if quantity_display else ""
			return (
				f"Removed {label}{quantity_clause}{price_clause}{status_suffix}. Anything else to update?"
			)
		return None

	def _build_customer_order_prompt(
		self,
		tool_calls: List[Dict[str, Any]],
		classifier_output: Optional[Dict[str, Any]],
		user_input: Optional[str],
	) -> Optional[str]:
		user_role = None
		if isinstance(self._context, dict):
			user_info = self._context.get("user")
			if isinstance(user_info, dict):
				user_role = str(user_info.get("role") or "").lower()
		if user_role and user_role != "customer":
			return None

		structured_items: List[Dict[str, Any]] = []
		relevant_intent = False
		if classifier_output:
			flow_value = (classifier_output.get("flow") or "").lower()
			if flow_value and flow_value != "customer":
				return None
			intent_label = classifier_output.get("intent") or ""
			if intent_label.startswith("intent.supplier"):
				return None
			relevant_intent = intent_label in {
				"intent.customer.place_order",
				"intent.customer.check_availability",
			}
			filled_slots = classifier_output.get("filled_slots") or {}
			order_items_slot = filled_slots.get("order_items")
			if isinstance(order_items_slot, list):
				structured_items = [item for item in order_items_slot if isinstance(item, dict)]
			elif isinstance(order_items_slot, dict):
				structured_items = [order_items_slot]
			else:
				maybe_item = {
					"product_name": filled_slots.get("product_name"),
					"quantity": filled_slots.get("quantity"),
					"unit": filled_slots.get("unit"),
					"supplier_name": filled_slots.get("supplier_name"),
				}
				if any(value for value in maybe_item.values()):
					structured_items = [maybe_item]

		selected_item: Dict[str, Any] = structured_items[0] if structured_items else {}
		quantity_value = selected_item.get("quantity")
		unit_value = selected_item.get("unit") or selected_item.get("unit_name")
		product_hint = selected_item.get("product_name") or selected_item.get("name")
		supplier_hint = selected_item.get("supplier_name") or selected_item.get("supplier")

		parsed_quantity = self._coerce_decimal(quantity_value)
		if parsed_quantity is None and user_input:
			match = re.search(r"(\d+(?:\.\d+)?)\s*(kg|kgs?|kilograms?|liter|liters?|litres?|ltr|ltrs?)?", user_input, re.IGNORECASE)
			if match:
				parsed_quantity = self._coerce_decimal(match.group(1))
				if not unit_value and match.group(2):
					unit_value = match.group(2)

		listing: Optional[Dict[str, Any]] = None
		filter_product: Optional[str] = None
		filter_supplier: Optional[str] = None
		for call in reversed(tool_calls):
			if call.get("tool") != "data_access":
				continue
			parsed_input = self._parse_tool_input(call.get("input"))
			if not isinstance(parsed_input, dict):
				continue
			if parsed_input.get("entity") != "supplier_products" or parsed_input.get("operation") != "search":
				continue
			filters = parsed_input.get("filters") if isinstance(parsed_input.get("filters"), dict) else {}
			if filter_product is None:
				filter_product = filters.get("product_name") or filters.get("name")
			if filter_supplier is None:
				filter_supplier = filters.get("supplier_name") or filters.get("supplier")
			observation = call.get("observation")
			if isinstance(observation, dict):
				results = observation.get("results")
				if isinstance(results, list) and results:
					listing = results[0]
					break

		if not listing:
			return None

		if not relevant_intent and not parsed_quantity and not structured_items:
			if not user_input or not re.search(r"\b(order|buy|get|need)\b", user_input, re.IGNORECASE):
				return None

		product_label = (
			product_hint
			or (listing.get("product_name") if isinstance(listing, dict) else None)
			or filter_product
		)
		supplier_label = supplier_hint or filter_supplier
		if not product_label:
			return None

		listing_unit = listing.get("unit") if isinstance(listing, dict) else None
		if unit_value is None:
			unit_value = listing_unit
		unit_price = None
		if isinstance(listing, dict):
			unit_price = (
				listing.get("unit_price_etb")
				or listing.get("unit_price")
				or listing.get("price_per_unit")
			)

		available_qty = listing.get("quantity_available") if isinstance(listing, dict) else None
		quantity_phrase = self._format_quantity_phrase(parsed_quantity, unit_value)
		available_phrase = None
		if available_qty is not None:
			available_phrase = self._format_quantity_phrase(available_qty, listing_unit or unit_value)

		price_display = self._format_decimal(unit_price)
		if price_display and price_display.endswith(".00"):
			price_display = price_display[:-3]
		per_unit_label = self._normalise_unit_label(unit_value)
		per_unit_suffix = ""
		if per_unit_label == "kg":
			per_unit_suffix = " per kg"
		elif per_unit_label == "liter":
			per_unit_suffix = " per liter"
		elif per_unit_label:
			per_unit_suffix = f" per {per_unit_label}"

		total_display = None
		if parsed_quantity is not None and unit_price is not None:
			unit_price_decimal = self._coerce_decimal(unit_price)
			if unit_price_decimal is not None:
				total_display = self._format_decimal(parsed_quantity * unit_price_decimal)
				if total_display and total_display.endswith(".00"):
					total_display = total_display[:-3]

		sentences: List[str] = []
		if supplier_label:
			lead_sentence = f"{supplier_label} has {product_label}"
		else:
			lead_sentence = f"I found {product_label}"
		if price_display:
			lead_sentence += f" at {price_display} ETB{per_unit_suffix}"
		elif per_unit_label:
			lead_sentence += f" sold per {per_unit_label}"
		sentences.append(lead_sentence.strip() + ".")

		if available_phrase:
			sentences.append(f"Current stock: {available_phrase}.")
		elif isinstance(listing, dict) and listing.get("is_expired"):
			sentences.append("This batch is marked expired.")

		if quantity_phrase:
			if total_display:
				sentences.append(f"I can line up {quantity_phrase} for roughly {total_display} ETB.")
			else:
				sentences.append(f"I can line up {quantity_phrase} for you.")
		else:
			sentences.append("How much should I add to your basket?")

		if isinstance(listing, dict) and listing.get("expiry_date"):
			sentences.append(f"Expiry: {listing['expiry_date']}.")

		sentences.append("Should I go ahead and place the order and confirm delivery details?")
		return " ".join(sentences)

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

	@staticmethod
	def _is_storage_question(user_input: Optional[str]) -> bool:
		if not user_input:
			return False
		lowered = user_input.lower()
		storage_markers = (
			" store",
			" storage",
			" keep",
			" keeping",
			" preserve",
			"ታከማ",
			"ማከማ",
		)
		return any(marker in lowered for marker in storage_markers)

	def _build_fallback_reply(
		self,
		tool_calls: List[Dict[str, Any]],
		user_input: Optional[str],
		classifier_output: Optional[Dict[str, Any]],
	) -> Optional[str]:
		text = self._build_schedule_followup(tool_calls)
		if text:
			return text

		text = self._build_pricing_guidance_fallback(tool_calls)
		if text:
			return text

		text = self._build_inventory_deletion_confirmation(tool_calls)
		if text:
			return text

		text = self._build_customer_order_prompt(tool_calls, classifier_output, user_input)
		if text:
			return text

		if self._is_storage_question(user_input):
			return None

		return self._build_inventory_fallback(tool_calls)

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
			fallback_reply = self._build_fallback_reply(tool_calls, user_input, classifier_output)
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
