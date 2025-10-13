from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, AsyncIterator, Iterable, List, Mapping, Optional

import httpx

from langchain_core.caches import BaseCache as _LCBaseCache  # type: ignore
from langchain_core.callbacks import Callbacks as _LCCallbacks  # type: ignore

from app.core.config import get_settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class LLMConfig:
    """Configuration for AsyncLLMService."""
    model: str = "deepseek-chat"
    temperature: float = 0.2
    max_retries: int = 3
    retry_backoff: float = 0.5  # seconds
    api_key: Optional[str] = None
    base_url: str = "https://api.deepseek.com/v1"
    extra_kwargs: Mapping[str, object] = field(default_factory=dict)


class LLMService:
    """Asynchronous LLM service targeting the DeepSeek Chat API."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        config: Optional[LLMConfig] = None,
        http_client: Optional[httpx.AsyncClient] = None,
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
        if not self.config.api_key:
            settings = get_settings()
            self.config.api_key = getattr(settings, "deepseek_api_key", None) or getattr(settings, "gemini_api_key", None)

        if not self.config.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set in configuration or environment.")

        self._external_client = http_client
        self._request_timeout = 60.0

    def _headers(self) -> Mapping[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        prompt: str,
        history: Optional[Iterable[Mapping[str, str]]],
        *,
        stream: bool = False,
    ) -> Mapping[str, Any]:
        messages = self._compose_messages(prompt, history)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            **self.config.extra_kwargs,
        }
        if stream:
            payload["stream"] = True
        return payload

    async def _post_json(self, payload: Mapping[str, Any]) -> httpx.Response:
        url = f"{self.config.base_url}/chat/completions"
        headers = self._headers()
        timeout = httpx.Timeout(self._request_timeout)

        if self._external_client is not None:
            response = await self._external_client.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
        else:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=headers)

        response.raise_for_status()
        return response

    @asynccontextmanager
    async def _stream_request(self, payload: Mapping[str, Any]) -> AsyncIterator[httpx.Response]:
        url = f"{self.config.base_url}/chat/completions"
        headers = self._headers()
        timeout = httpx.Timeout(None, connect=self._request_timeout)

        if self._external_client is not None:
            async with self._external_client.stream(
                "POST",
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
            ) as response:
                response.raise_for_status()
                yield response
        else:
            client = httpx.AsyncClient(timeout=timeout)
            try:
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    yield response
            finally:
                await client.aclose()

    def update_system_prompt(self, system_prompt: str) -> None:
        """Update the system prompt for future requests."""
        self.system_prompt = system_prompt or self.system_prompt

    def clone(self, system_prompt: Optional[str] = None) -> "LLMService":
        """Return a lightweight copy sharing the same underlying model/config."""
        return LLMService(
            system_prompt=system_prompt or self.system_prompt,
            config=self.config,
            http_client=self._external_client,
        )

    async def acomplete(
        self,
        prompt: str,
        *,
        history: Optional[Iterable[Mapping[str, str]]] = None,
        **kwargs,
    ) -> str:
        """Generate a single-shot completion asynchronously with retries."""
        payload = self._build_payload(prompt, history)

        async def _call() -> str:
            response = await self._post_json(payload)
            data = response.json()
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
                raise RuntimeError(f"Unexpected DeepSeek response structure: {data}") from exc
            return content.strip()

        return await self._retry(_call, action="completion")

    async def astream(
        self,
        prompt: str,
        *,
        history: Optional[Iterable[Mapping[str, str]]] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream the model's response tokens asynchronously with retries."""
        payload = self._build_payload(prompt, history, stream=True)

        async def _gen():
            async with self._stream_request(payload) as stream:
                async for line in stream.aiter_lines():
                    if not line:
                        continue
                    if line.startswith(":"):
                        continue  # comment/heartbeat
                    if line.strip() == "data: [DONE]":
                        break
                    if not line.startswith("data: "):
                        continue
                    raw = line.split("data: ", 1)[1].strip()
                    if not raw:
                        continue
                    try:
                        event = json.loads(raw)
                        delta = event["choices"][0]["delta"].get("content")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        logger.debug("Skipping malformed DeepSeek stream chunk: %s", raw)
                        continue
                    if delta:
                        yield delta

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
    ) -> List[Mapping[str, str]]:
        messages: List[Mapping[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt.strip()})

        if history:
            for item in history:
                role = (item.get("role") or "user").lower()
                content = (item.get("content") or "").strip()
                if not content:
                    continue
                mapped_role = {
                    "assistant": "assistant",
                    "system": "system",
                }.get(role, "user")
                messages.append({"role": mapped_role, "content": content})

        messages.append({"role": "user", "content": prompt})
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
        transient_errors = (
            "connection refused",
            "timeout",
            "temporary failure",
            "429",
            "rate limit",
            "overloaded",
        )
        message = str(e).lower()
        if any(err in message for err in transient_errors):
            return True
        return isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError))

    def _raise_llm_error(self, e: Exception) -> None:
        raise RuntimeError(
            f"LLM request failed: {e}. Verify your DeepSeek API key and model '{self.config.model}' is available."
        ) from e
