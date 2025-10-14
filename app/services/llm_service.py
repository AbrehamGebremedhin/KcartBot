from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, AsyncGenerator, AsyncIterator, Dict, Iterable, List, Mapping, Optional

import httpx

from langchain_core.caches import BaseCache as _LCBaseCache  # type: ignore
from langchain_core.callbacks import Callbacks as _LCCallbacks  # type: ignore

from app.core.config import get_settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LLMServiceError(RuntimeError):
    """Raised when the upstream language model request definitively fails."""


@dataclass
class LLMConfig:
    """Configuration for AsyncLLMService."""
    model: str = "deepseek-chat"
    temperature: float = 0.2
    max_retries: int = 3
    retry_backoff: float = 0.5  # seconds
    api_key: Optional[str] = None
    base_url: str = "https://api.deepseek.com/v1"
    request_timeout: float = 45.0  # seconds
    slow_request_threshold: float = 8.0  # seconds
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
            You are KCartBot, a warm and efficient assistant for Ethiopia's fresh-goods marketplace. Keep every reply to one short paragraph and follow these guardrails:
            - Identify whether the user is a customer or supplier and stay within that flow.
            - Customers: gather missing name, phone, and delivery location if they're new. For orders capture items with kg or liter units, confirm availability, ask for delivery date/location, present a concise ETB summary, remind them payment is Cash on Delivery, then confirm once details and order items succeed.
            - Suppliers: help onboard quickly, collect product name, quantity, unit price, delivery schedule, and expiry one detail at a time, and share pricing guidance before submitting inventory updates.
            - Always use ETB currency and kg/liter units, reuse known context, ask for missing details individually, and keep the tone pragmatic and friendly.
            - When unsure, ask clarifying questions instead of guessing, and avoid multi-step instructions in a single reply.
            """
        )
        if not self.config.api_key:
            settings = get_settings()
            self.config.api_key = getattr(settings, "deepseek_api_key", None) or getattr(settings, "gemini_api_key", None)

        if not self.config.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set in configuration or environment.")

        self._external_client = http_client
        self._request_timeout = float(self.config.request_timeout or 45.0)
        self._slow_request_threshold = float(self.config.slow_request_threshold or 8.0)

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
        history_list = list(history or [])
        payload = self._build_payload(prompt, history_list)

        async def _call() -> str:
            response = await self._post_json(payload)
            data = response.json()
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
                raise RuntimeError(f"Unexpected DeepSeek response structure: {data}") from exc
            return content.strip()
        metrics: Dict[str, Any] = {}
        start = perf_counter()
        prompt_chars = len(prompt or "")
        history_entries = len(history_list)
        history_chars = sum(len(item.get("content", "")) for item in history_list)
        try:
            result = await self._retry(_call, action="completion", metrics=metrics)
        except LLMServiceError:
            duration = perf_counter() - start
            attempts = metrics.get("attempts", self.config.max_retries)
            logger.error(
                "LLM completion failed after %.2fs (attempts=%d, prompt_chars=%d, history_entries=%d, history_chars=%d, errors=%s)",
                duration,
                attempts,
                prompt_chars,
                history_entries,
                history_chars,
                metrics.get("errors"),
            )
            raise
        else:
            duration = perf_counter() - start
            attempts = metrics.get("attempts", 1)
            if duration >= self._slow_request_threshold or attempts > 1:
                logger.warning(
                    "LLM completion succeeded in %.2fs (attempts=%d, prompt_chars=%d, history_entries=%d, history_chars=%d)",
                    duration,
                    attempts,
                    prompt_chars,
                    history_entries,
                    history_chars,
                )
            return result

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
                logger.warning(
                    "Retrying stream in %.2fs after %s: %r",
                    delay,
                    type(e).__name__,
                    e,
                )
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

    async def _retry(self, func, *, action: str, metrics: Optional[Dict[str, Any]] = None):
        for attempt in range(self.config.max_retries):
            try:
                result = await func()
                if metrics is not None:
                    metrics["attempts"] = attempt + 1
                return result
            except Exception as e:
                if metrics is not None:
                    metrics["attempts"] = attempt + 1
                    metrics.setdefault("errors", []).append(repr(e) or str(e) or type(e).__name__)
                if not self._should_retry(e, attempt):
                    if metrics is not None:
                        metrics["failed"] = True
                    self._raise_llm_error(e)
                delay = self.config.retry_backoff * (2**attempt)
                logger.warning(
                    "Retrying %s in %.2fs after %s: %r",
                    action,
                    delay,
                    type(e).__name__,
                    e,
                )
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
        detail = str(e).strip()
        if not detail:
            detail = repr(e)
        if not detail:
            detail = type(e).__name__
        raise LLMServiceError(
            f"LLM request failed: {detail}. Verify your DeepSeek API key and model '{self.config.model}' is available."
        ) from e
