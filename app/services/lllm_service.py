from __future__ import annotations

import asyncio
import os
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Iterable, List, Mapping, Optional

from langchain_core.caches import BaseCache as _LCBaseCache  # type: ignore
from langchain_core.callbacks import Callbacks as _LCCallbacks  # type: ignore
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class LLMConfig:
    """Configuration for AsyncLLMService."""
    model: str = "gemini-2.0-flash-lite"  # Gemini model name
    temperature: float = 0.2
    max_retries: int = 3
    retry_backoff: float = 0.5  # seconds
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    extra_kwargs: Mapping[str, object] = field(default_factory=dict)


class LLMService:
    """Asynchronous LLM service using Google Gemini via google-generativeai."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        config: Optional[LLMConfig] = None,
        model: Optional[Any] = None,
        client: Optional[genai.Client] = None,
    ) -> None:
        self.config = config or LLMConfig()
        self.system_prompt = system_prompt or (
            """
            You are KCartBot, a conversational marketplace platform connecting customers with suppliers of fresh produce and groceries.

            ## Core Identity
            - **Name**: KCartBot
            - **Role**: Handle ALL interactions for customers and suppliers
            - **Tone**: Warm, helpful, efficient, knowledgeable about fresh produce
            - **Style**: Natural conversation, 1-3 sentences per response, avoid robotic language

            ## Critical Rules
            1. **Detect User Type First**: Determine if CUSTOMER or SUPPLIER in first interaction
            2. **Context Aware**: Remember conversation history and user state
            3. **Proactive**: Anticipate needs, suggest relevant options
            4. **Validate Data**: Confirm critical info (quantities, dates, prices, locations)
            5. **Natural Flow**: Don't list all options unless asked - guide conversationally

            ---

            ## CUSTOMER INTERACTIONS

            ### Registration (New Users)
            Collect: Name, Phone Number, Default Delivery Location
            - Keep it conversational: "Welcome to KCartBot! 🥬 What's your name?"
            - Validate phone format and location details

            ### Discovery & Advisory (RAG Knowledge)
            You know about:
            - Storage tips ("Store tomatoes at room temp, not fridge")
            - Nutrition ("Avocados have 3x calories of bananas - healthy fats!")
            - Seasonality ("Mangoes are in season now - sweetest in June-August")
            - Selection ("Ripe mangoes yield slightly to pressure, smell sweet at stem")
            - Preparation & recipes

            **Answer advisory questions directly and helpfully**, then offer to help them order.

            ### Ordering Flow
            1. **Parse Natural Language**: "5kg red onions and 2L milk" → extract items/quantities
            2. **Check Availability**: Query stock, confirm or suggest alternatives
            3. **Ask Delivery Date**: "When would you like delivery?"
            4. **Confirm Location**: "Send to [default address] or somewhere else?"
            5. **Show Summary**: 
            ```
            📦 Order Summary:
            - 5kg Red Onions - 275 ETB
            - 2L Milk - 130 ETB
            Total: 405 ETB | Delivery: [Date] at [Location]
            ```
            6. **Get Confirmation**: "Confirm this order?"

            ### Payment (COD Only)
            When user confirms:
            1. Say: "Payment is Cash on Delivery. Confirming your order..."
            2. [AUTO: 5-second pause]
            3. Auto-reply: "✅ Order Confirmed! Order #[ID]. Total: [Amount] ETB (pay on delivery). You'll get updates soon!"

            ---

            ## SUPPLIER INTERACTIONS

            ### Registration
            Collect: Name/Business Name, Phone Number
            - Simple and quick: "Welcome to KCartBot Supplier Hub! 🚜 What's your business name?"

            ### Product Addition
            When supplier says "Add tomatoes" or similar:
            1. **Quantity**: "How many kg available?"
            2. **Delivery Dates**: "When can customers receive these?"
            3. **Expiry (Optional)**: "Any expiry date? (Helps us suggest flash sales)"
            4. **Pricing Intelligence** (CRITICAL):
            ```
            "Let me check market rates...
            📊 Tomatoes Market Data:
            - Local Shops: ~50 ETB/kg
            - Supermarkets: ~65 ETB/kg  
            - Sweet spot: 55 ETB/kg moves fast
            
            What price do you want to set?"
            ```
            5. **Image Generation**: "Generate a fresh image for these tomatoes?"

            ### Stock Management
            - **Check Stock**: "You have: Tomatoes (50kg), Onions (30kg)..."
            - **Expiry Alerts** (Proactive): "⚠️ Your milk expires in 2 days. Run a 20% flash sale to clear it?"

            ### Logistics & Schedule
            - **Delivery Queries**: "Tuesday: 8 deliveries. Wednesday: 5 deliveries."
            - **Weekly Overview**: Show schedule table when asked

            ### Order Management
            - **New Order Notification**: "🔔 New Order! Customer needs 10kg tomatoes by Thursday. Accept?"
            - **Accept/Decline**: Track capacity, update supplier

            ---

            ## Response Guidelines
            - **Be concise**: 1-3 sentences unless detail needed
            - **Natural language**: "Great! Let's add those tomatoes" not "Processing product addition"
            - **Use emojis sparingly**: 🥬🍅✅🔔 for clarity, not decoration
            - **Confirm before acting**: Always get user confirmation for orders, price changes, etc.
            - **Handle errors gracefully**: "I didn't catch that. Could you specify the quantity in kg?"
            - **Suggest, don't overwhelm**: Offer next logical step, not full menu

            ## Knowledge Base Access
            When answering advisory questions:
            - Provide accurate produce knowledge (storage, nutrition, seasonality)
            - Cite confidence level if uncertain: "Typically..." or "Generally..."
            - Offer to help order items discussed
            - Suggest complementary products naturally

            ## Data Handling
            - **Currency**: Ethiopian Birr (ETB)
            - **Weight**: Kilograms (kg)
            - **Volume**: Liters (L)
            - **Date Format**: Natural ("next Tuesday") or DD/MM/YYYY
            - **Phone Format**: Validate Ethiopian numbers

            ## Example Interactions

            **Customer Advisory**:
            User: "How do I store tomatoes?"
            You: "Store tomatoes at room temperature, not in the fridge - cold kills their flavor! 🍅 They'll stay fresh for 3-5 days. Need to order some?"

            **Customer Order**:
            User: "I need 5kg onions and milk"
            You: "Got it! 5kg red onions and how much milk - 1L or 2L?"

            **Supplier Pricing**:
            User: "Add 100kg carrots"
            You: "Perfect! When can customers receive these carrots?"
            [After dates/expiry...]
            You: "📊 Market check: Carrots run 30-40 ETB/kg locally. 35 ETB moves well. Your price?"

            **Proactive Flash Sale**:
            You: "⚠️ Your spinach expires tomorrow. Flash sale at 40% off? Could clear your 15kg stock fast!"

            ---

            Remember: Be helpful, natural, and efficient. Guide users through flows conversationally, don't dump information. You're a knowledgeable friend helping them buy/sell fresh produce, not a robotic form-filler."""
        )
        if client is not None:
            self._client = client
        else:
            if not self.config.api_key:
                raise RuntimeError("GEMINI_API_KEY is not set in environment or config.")
            self._client = genai.Client(api_key=self.config.api_key)

        if model is not None:
            # Backward compatibility: allow passing pre-built GenerativeModel
            self._model_name = getattr(model, "model_name", self.config.model)
            self._legacy_model = model
        else:
            self._model_name = self.config.model
            self._legacy_model = None

    def _build_generation_config(self) -> types.GenerateContentConfig:
        base_kwargs = dict(self.config.extra_kwargs)
        # Ensure temperature always present for reproducibility
        base_kwargs.setdefault("temperature", self.config.temperature)
        try:
            return types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                **base_kwargs,
            )
        except TypeError:
            # Fall back to minimal supported configuration if extras are invalid
            return types.GenerateContentConfig(
                temperature=self.config.temperature,
                system_instruction=self.system_prompt,
            )

    def update_system_prompt(self, system_prompt: str) -> None:
        """Update the system prompt for future requests."""
        self.system_prompt = system_prompt or self.system_prompt

    def clone(self, system_prompt: Optional[str] = None) -> "LLMService":
        """Return a lightweight copy sharing the same underlying model/config."""
        return LLMService(
            system_prompt=system_prompt or self.system_prompt,
            config=self.config,
            model=self._legacy_model,
            client=self._client,
        )

    async def acomplete(
        self,
        prompt: str,
        *,
        history: Optional[Iterable[Mapping[str, str]]] = None,
        **kwargs,
    ) -> str:
        """Generate a single-shot completion asynchronously with retries."""
        messages = self._compose_messages(prompt, history)

        async def _call():
            # google-generativeai does not support async, so run in executor
            import concurrent.futures
            loop = asyncio.get_event_loop()
            def sync_call():
                config_obj = self._build_generation_config()
                if self._legacy_model is not None:
                    response = self._legacy_model.generate_content(
                        messages,
                        config=config_obj,
                    )
                else:
                    response = self._client.models.generate_content(
                        model=self._model_name,
                        contents=messages,
                        config=config_obj,
                    )
                return response.text if hasattr(response, "text") else str(response)
            return await loop.run_in_executor(None, sync_call)

        return await self._retry(_call, action="completion")

    async def astream(
        self,
        prompt: str,
        *,
        history: Optional[Iterable[Mapping[str, str]]] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream the model's response tokens asynchronously with retries."""
        messages = self._compose_messages(prompt, history)

        async def _gen():
            import concurrent.futures
            loop = asyncio.get_event_loop()
            def sync_stream():
                config_obj = self._build_generation_config()
                if self._legacy_model is not None:
                    generator = self._legacy_model.generate_content(
                        messages,
                        stream=True,
                        config=config_obj,
                    )
                else:
                    generator = self._client.models.generate_content(
                        model=self._model_name,
                        contents=messages,
                        stream=True,
                        config=config_obj,
                    )

                for chunk in generator:
                    yield chunk.text if hasattr(chunk, "text") else str(chunk)
            # Wrap sync generator as async
            for token in await loop.run_in_executor(None, lambda: list(sync_stream())):
                yield token

        for attempt in range(self.config.max_retries):
            try:
                async for token in _gen():
                    yield token
                break
            except Exception as e:
                if not self._should_retry(e, attempt):
                    self._raise_llm_error(e)
                delay = self.config.retry_backoff * (2**attempt)
                logger.warning(f"Retrying stream in {delay:.2f}s after error: {e}")
                await asyncio.sleep(delay)

    def _compose_messages(
        self,
        prompt: str,
        history: Optional[Iterable[Mapping[str, str]]],
    ) -> List[types.Content]:
        """Build Gemini content payload from history and prompt."""
        messages: List[types.Content] = []
        if history:
            for item in history:
                role = (item.get("role") or "").lower()
                content = item.get("content") or ""
                if not content:
                    continue
                mapped_role = "model" if role == "assistant" else "user"
                messages.append(
                    types.Content(role=mapped_role, parts=[types.Part(text=content)])
                )
        messages.append(
            types.Content(role="user", parts=[types.Part(text=prompt)])
        )
        return messages

    async def _retry(self, func, *, action: str):
        for attempt in range(self.config.max_retries):
            try:
                return await func()
            except Exception as e:
                if not self._should_retry(e, attempt):
                    self._raise_llm_error(e)
                delay = self.config.retry_backoff * (2**attempt)
                logger.warning(f"Retrying {action} in {delay:.2f}s after error: {e}")
                await asyncio.sleep(delay)
        raise RuntimeError(f"{action.capitalize()} failed after {self.config.max_retries} retries.")

    def _should_retry(self, e: Exception, attempt: int) -> bool:
        if attempt >= self.config.max_retries - 1:
            return False
        transient_errors = ("Connection refused", "Timeout", "Temporary failure", "429", "quota")
        return any(err.lower() in str(e).lower() for err in transient_errors)

    def _raise_llm_error(self, e: Exception) -> None:
        raise RuntimeError(
            f"LLM request failed: {e}. Verify your Gemini API key and model '{self.config.model}' is available."
        ) from e
