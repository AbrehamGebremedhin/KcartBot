"""Intent classification tool for routing KCartBot conversations."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

from app.tools.base import ToolBase
from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntentDefinition:
    """Definition of an intent including expected slots and usage hints."""

    flow: str
    description: str
    required_slots: List[str]
    optional_slots: List[str]
    suggested_tools: List[str]


INTENT_REGISTRY: Dict[str, IntentDefinition] = {
    "intent.user.is_customer": IntentDefinition(
        flow="onboarding",
        description="User identifies themselves as a customer looking to place orders.",
        required_slots=[],
        optional_slots=[],
        suggested_tools=[],
    ),
    "intent.user.is_supplier": IntentDefinition(
        flow="onboarding",
        description="User identifies themselves as a supplier managing inventory.",
        required_slots=[],
        optional_slots=[],
        suggested_tools=[],
    ),
    "intent.user.has_account": IntentDefinition(
        flow="onboarding",
        description="User indicates they already have an existing account.",
        required_slots=[],
        optional_slots=[],
        suggested_tools=[],
    ),
    "intent.user.new_user": IntentDefinition(
        flow="onboarding",
        description="User indicates they are a new user without an existing account.",
        required_slots=[],
        optional_slots=[],
        suggested_tools=[],
    ),
    "intent.user.verify_account": IntentDefinition(
        flow="onboarding",
        description="User provides name and phone number to verify existing account.",
        required_slots=["user_name", "phone_number"],
        optional_slots=[],
        suggested_tools=["database_access"],
    ),
    "intent.customer.register": IntentDefinition(
        flow="customer",
        description="Capture a customer's name, phone number, and default delivery location during onboarding.",
        required_slots=["customer_name", "phone_number", "default_location"],
        optional_slots=[],
        suggested_tools=["database_access"],
    ),
    "intent.customer.check_availability": IntentDefinition(
        flow="customer",
        description="Customer asks if a specific product or item is available.",
        required_slots=["product_name"],
        optional_slots=["quantity", "delivery_date"],
        suggested_tools=["database_access", "vector_search"],
    ),
    "intent.customer.storage_advice": IntentDefinition(
        flow="customer",
        description="Guidance on how to store a specific item to maintain freshness.",
        required_slots=["product_name"],
        optional_slots=[],
        suggested_tools=["vector_search"],
    ),
    "intent.customer.nutrition_query": IntentDefinition(
        flow="customer",
        description="Compare nutritional properties such as calories between products.",
        required_slots=["product_a", "product_b"],
        optional_slots=["nutrient_metric"],
        suggested_tools=["vector_search"],
    ),
    "intent.customer.seasonal_query": IntentDefinition(
        flow="customer",
        description="Ask about seasonal availability of produce.",
        required_slots=[],
        optional_slots=["season", "location"],
        suggested_tools=["vector_search"],
    ),
    "intent.customer.what_is_in_season": IntentDefinition(
        flow="customer",
        description="Ask what produce is currently in season.",
        required_slots=[],
        optional_slots=["location"],
        suggested_tools=["vector_search"],
    ),
    "intent.customer.general_advisory": IntentDefinition(
        flow="customer",
        description="General product, food, or preparation questions that rely on knowledge retrieval.",
        required_slots=["question"],
        optional_slots=["related_product"],
        suggested_tools=["vector_search"],
    ),
    "intent.customer.place_order": IntentDefinition(
        flow="customer",
        description="Customer wants to place a new order specifying items and quantities.",
        required_slots=["order_items", "preferred_delivery_date"],
        optional_slots=["delivery_date", "supplier_name"],
        suggested_tools=["database_access"],
    ),
    "intent.customer.set_delivery_date": IntentDefinition(
        flow="customer",
        description="Customer provides or changes delivery date for an order.",
        required_slots=["delivery_date"],
        optional_slots=["order_reference"],
        suggested_tools=["database_access"],
    ),
    "intent.customer.set_delivery_location": IntentDefinition(
        flow="customer",
        description="Customer confirms or updates delivery address for an order.",
        required_slots=["delivery_location"],
        optional_slots=["order_reference"],
        suggested_tools=["database_access"],
    ),
    "intent.customer.confirm_payment": IntentDefinition(
        flow="customer",
        description="Customer confirms cash-on-delivery payment and expects confirmation.",
        required_slots=["order_reference"],
        optional_slots=["amount"],
        suggested_tools=["database_access"],
    ),
    "intent.customer.check_deliveries": IntentDefinition(
        flow="customer",
        description="Customer wants to check their scheduled deliveries or order status.",
        required_slots=[],
        optional_slots=["date", "order_reference"],
        suggested_tools=["database_access"],
    ),
    "intent.supplier.register": IntentDefinition(
        flow="supplier",
        description="Onboard a supplier by capturing business name and phone number.",
        required_slots=["supplier_name", "phone_number"],
        optional_slots=[],
        suggested_tools=["database_access"],
    ),
    "intent.supplier.add_product": IntentDefinition(
        flow="supplier",
        description="Supplier wants to add a new product listing.",
        required_slots=["product_name"],
        optional_slots=["category"],
        suggested_tools=["database_access"],
    ),
    "intent.supplier.set_quantity": IntentDefinition(
        flow="supplier",
        description="Supplier sets available quantity for an inventory item.",
        required_slots=["product_name", "quantity"],
        optional_slots=["unit"],
        suggested_tools=["database_access"],
    ),
    "intent.supplier.update_inventory": IntentDefinition(
        flow="supplier",
        description="Supplier wants to add more quantity to an existing inventory item.",
        required_slots=["product_name", "quantity"],
        optional_slots=["unit"],
        suggested_tools=["database_access"],
    ),
    "intent.supplier.set_delivery_dates": IntentDefinition(
        flow="supplier",
        description="Supplier provides delivery window for inventory availability.",
        required_slots=["delivery_dates"],
        optional_slots=["product_name"],
        suggested_tools=["database_access"],
    ),
    "intent.supplier.set_expiry_date": IntentDefinition(
        flow="supplier",
        description="Supplier gives an expiry date for products (optional).",
        required_slots=["expiry_date"],
        optional_slots=["product_name"],
        suggested_tools=["database_access"],
    ),
    "intent.supplier.set_price": IntentDefinition(
        flow="supplier",
        description="Supplier sets price per unit.",
        required_slots=["product_name", "unit_price"],
        optional_slots=["unit"],
        suggested_tools=["database_access"],
    ),
    "intent.supplier.request_pricing_insight": IntentDefinition(
        flow="supplier",
        description="Supplier wants competitor or historical price analysis before pricing.",
        required_slots=["product_name"],
        optional_slots=["location"],
        suggested_tools=["database_access"],
    ),
    "intent.supplier.generate_product_image": IntentDefinition(
        flow="supplier",
        description="Supplier requests marketing image generation for a product.",
        required_slots=["product_name"],
        optional_slots=["style"],
        suggested_tools=["image_generator"],
    ),
    "intent.supplier.check_stock": IntentDefinition(
        flow="supplier",
        description="Supplier wants overview of their inventory levels.",
        required_slots=[],
        optional_slots=["product_name"],
        suggested_tools=["database_access"],
    ),
    "intent.supplier.view_expiring_products": IntentDefinition(
        flow="supplier",
        description="Supplier wants items approaching expiry.",
        required_slots=[],
        optional_slots=["time_horizon"],
        suggested_tools=["database_access"],
    ),
    "intent.supplier.accept_flash_sale": IntentDefinition(
        flow="supplier",
        description="Supplier accepts proposed flash sale offer.",
        required_slots=["product_name"],
        optional_slots=["discount_rate", "duration"],
        suggested_tools=["flash_sale_manager", "database_access"],
    ),
    "intent.supplier.decline_flash_sale": IntentDefinition(
        flow="supplier",
        description="Supplier declines flash sale.",
        required_slots=["product_name"],
        optional_slots=["reason"],
        suggested_tools=["flash_sale_manager"],
    ),
    "intent.supplier.view_delivery_schedule": IntentDefinition(
        flow="supplier",
        description="Supplier asks for delivery schedule overview.",
        required_slots=[],
        optional_slots=["date_range"],
        suggested_tools=["database_access"],
    ),
    "intent.supplier.check_deliveries_by_date": IntentDefinition(
        flow="supplier",
        description="Supplier asks for deliveries on a specific date.",
        required_slots=["date"],
        optional_slots=[],
        suggested_tools=["database_access"],
    ),
    "intent.supplier.receive_order_notification": IntentDefinition(
        flow="supplier",
        description="Bot notifies supplier about a new incoming order.",
        required_slots=["order_reference"],
        optional_slots=["order_summary"],
        suggested_tools=["database_access"],
    ),
    "intent.supplier.accept_order": IntentDefinition(
        flow="supplier",
        description="Supplier accepts an order that was just announced.",
        required_slots=["order_reference"],
        optional_slots=["notes"],
        suggested_tools=["database_access"],
    ),
    "intent.supplier.decline_order": IntentDefinition(
        flow="supplier",
        description="Supplier declines an order due to capacity constraints or other reasons.",
        required_slots=["order_reference"],
        optional_slots=["reason"],
        suggested_tools=[],
    ),
}


INTENT_CATALOG_TEXT = "\n".join(
    [
        f"- {intent}: flow={definition.flow}, required={definition.required_slots}, optional={definition.optional_slots}"
        for intent, definition in INTENT_REGISTRY.items()
    ]
)


class IntentClassifierPayload(BaseModel):
    """Validated payload returned by the intent classifier model."""

    intent: Optional[str] = None
    flow: Optional[Literal["customer", "supplier", "onboarding", "unknown"]] = None,
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    filled_slots: Dict[str, Any] = Field(default_factory=dict)
    missing_slots: Optional[List[str]] = None
    rationale: Optional[str] = None

    @field_validator("intent", mode="before")
    @classmethod
    def _normalize_intent(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip()

    @field_validator("flow", mode="before")
    @classmethod
    def _normalize_flow(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip().lower()


CLASSIFIER_SYSTEM_PROMPT = f"""
You are the intent classification module for KCartBot. Analyse the most recent user utterance and map it to one of the supported intents.

IMPORTANT: If the user says something like "okay", "yes", "sure", "go ahead", "sounds good", etc., check the conversation context to understand what they are confirming:
- If previous context shows supplier onboarding in progress, classify as "intent.supplier.register"
- If previous context shows customer onboarding in progress, classify as "intent.customer.register"
- If confirming an order, classify as "intent.customer.confirm_order"
- If confirming a flash sale action, classify as appropriate flash sale intent
- Use the context to infer the actual intent behind the confirmation

Each intent belongs to exactly one flow (customer or supplier) and is described below:
{INTENT_CATALOG_TEXT}

For intent.customer.place_order, parse order details from the user's message and fill order_items as a list of objects with product_name, quantity, and unit. For example:
- "I want 2 kg mango" → {{"order_items": [{{"product_name": "mango", "quantity": 2, "unit": "kg"}}]}}
- "Order 5 liters milk and 3 kg tomatoes" → {{"order_items": [{{"product_name": "milk", "quantity": 5, "unit": "liter"}}, {{"product_name": "tomatoes", "quantity": 3, "unit": "kg"}}]}}

Return a compact JSON object with the following keys:
- intent: the best matching intent string from the catalog. Choose the most specific option.
- flow: "customer" or "supplier".
- confidence: float between 0 and 1 representing your confidence.
- filled_slots: object containing any slot values already provided by the user (keys in snake_case).
- missing_slots: list of slots still required before fulfilment.
- rationale: short natural language explanation (max 25 words).

If no intent reasonably matches AND there's no context to understand a confirmation, set intent to "intent.unknown" and flow to "unknown" with confidence under 0.4.

Always respond with JSON only. Avoid markdown fences or commentary.
"""


class IntentClassifierTool(ToolBase):
    """LangChain-compatible tool that classifies user utterances into intents."""

    def __init__(
        self,
        llm_service: Optional["LLMService"] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(
            name="intent_classifier",
            description=(
                "Classify the current user message into a supported intent. "
                "Call this before any other tool to decide the correct flow and required slots."
            ),
        )
        prompt = system_prompt or CLASSIFIER_SYSTEM_PROMPT
        if llm_service:
            self._llm = llm_service
            self._llm.update_system_prompt(prompt)
        else:
            from app.services.llm_service import LLMService
            self._llm = LLMService(system_prompt=prompt)

    async def run(self, input: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Classify the provided text and return structured metadata."""
        if isinstance(input, dict):
            utterance = input.get("text") or input.get("utterance") or input.get("message") or ""
        else:
            utterance = str(input or "")

        if not utterance.strip():
            return {
                "intent": "intent.unknown",
                "flow": "unknown",
                "confidence": 0.0,
                "filled_slots": {},
                "missing_slots": [],
                "rationale": "No user utterance provided.",
            }

        # Extract chat history from context for better classification
        chat_history = []
        if context and "chat_history" in context:
            chat_history = context["chat_history"]
            # Keep only last 3 exchanges to avoid token overflow
            if len(chat_history) > 6:
                chat_history = chat_history[-6:]

        # Format history for the classifier
        history_text = ""
        if chat_history:
            history_text = "Recent conversation:\n"
            for msg in chat_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"

        extra_context = ""
        if context:
            # Filter out chat_history from extra context to avoid duplication
            filtered_context = {k: v for k, v in context.items() if k != "chat_history"}
            if filtered_context:
                try:
                    extra_context = json.dumps(filtered_context, ensure_ascii=False)
                except Exception:
                    extra_context = str(filtered_context)

        prompt = (
            f"{history_text}\n"
            f"Current utterance: {utterance}\n"
            f"Session context: {extra_context or 'null'}\n"
            "Respond with JSON only."
        )

        try:
            response = await self._llm.acomplete(prompt)
            return self._parse_response(response)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Intent classification failed: %s", exc)
            return {
                "intent": "intent.unknown",
                "flow": "unknown",
                "confidence": 0.0,
                "filled_slots": {},
                "missing_slots": [],
                "rationale": "LLM classification error.",
            }

    @staticmethod
    def _parse_response(raw_text: str) -> Dict[str, Any]:
        """Parse JSON content from the LLM response."""
        if not raw_text:
            return {
                "intent": "intent.unknown",
                "flow": "unknown",
                "confidence": 0.0,
                "filled_slots": {},
                "missing_slots": [],
                "rationale": "Empty response from classifier.",
            }

        json_text = IntentClassifierTool._extract_json(raw_text)
        if not json_text:
            logger.warning("Classifier returned non-JSON payload: %s", raw_text)
            return {
                "intent": "intent.unknown",
                "flow": "unknown",
                "confidence": 0.0,
                "filled_slots": {},
                "missing_slots": [],
                "rationale": "Classifier response was not valid JSON.",
            }

        try:
            raw_payload = json.loads(json_text)
        except json.JSONDecodeError:
            logger.warning("Failed to decode classifier JSON: %s", json_text)
            return {
                "intent": "intent.unknown",
                "flow": "unknown",
                "confidence": 0.0,
                "filled_slots": {},
                "missing_slots": [],
                "rationale": "Classifier JSON parsing error.",
            }

        try:
            payload = IntentClassifierPayload.model_validate(raw_payload)
        except ValidationError as exc:
            logger.warning("Invalid classifier payload: %s", exc)
            return {
                "intent": "intent.unknown",
                "flow": "unknown",
                "confidence": 0.0,
                "filled_slots": {},
                "missing_slots": [],
                "rationale": "Classifier payload validation error.",
            }

        intent = payload.intent or "intent.unknown"
        definition = INTENT_REGISTRY.get(intent)
        filled_slots = payload.filled_slots or {}
        missing_slots = payload.missing_slots
        if missing_slots is None and definition:
            missing_slots = [
                slot for slot in definition.required_slots if not filled_slots.get(slot)
            ]

        flow = payload.flow or (definition.flow if definition else "unknown")
        confidence = payload.confidence if payload.confidence is not None else 0.0
        rationale = payload.rationale or ""

        return {
            "intent": intent,
            "flow": flow,
            "confidence": float(confidence),
            "filled_slots": filled_slots,
            "missing_slots": missing_slots or [],
            "rationale": rationale,
            "suggested_tools": definition.suggested_tools if definition else [],
        }

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Extract the first JSON object from the text."""
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return match.group(0) if match else None
