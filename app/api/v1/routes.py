"""Version 1 API routes."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from app.services.chat_service import ChatService
from app.services.llm_service import LLMServiceError
from app.core.rate_limiter import RateLimitStatus, rate_limiter


logger = logging.getLogger(__name__)


router = APIRouter()
_chat_service = ChatService()


class ChatRequest(BaseModel):
	session_id: str = Field(..., min_length=1, description="Opaque session identifier")
	message: str = Field(..., min_length=1, description="User utterance to send to the agent")
	context: Optional[Dict[str, Any]] = Field(default=None, description="Optional contextual metadata")


class ChatResponse(BaseModel):
	response: str
	intent: Optional[str] = None
	flow: Optional[str] = None
	classifier_output: Optional[Dict[str, Any]] = None
	tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
	trace: Optional[Dict[str, Any]] = None


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat_endpoint(
	payload: ChatRequest,
	rate_status: RateLimitStatus = Depends(rate_limiter),
) -> JSONResponse:
	"""Relay chat messages to the conversational agent."""
	try:
		result = await _chat_service.send_message(
			payload.session_id,
			payload.message,
			context=payload.context,
		)
	except ValidationError as exc:
		raise HTTPException(status_code=422, detail=exc.errors()) from exc
	except LLMServiceError as exc:
		logger.warning(
			"LLM unavailable for session",
			extra={
				"session_id": payload.session_id,
				"user_message": payload.message,
			},
		)
		raise HTTPException(status_code=503, detail=str(exc)) from exc
	except Exception as exc:  # pragma: no cover - defensive guard
		logger.exception(
			"Chat service failure for session",
			extra={
				"session_id": payload.session_id,
				"user_message": payload.message,
			},
		)
		raise HTTPException(status_code=500, detail="Chat service error") from exc

	sanitised = _sanitise_chat_result(result)
	headers = {
		"X-RateLimit-Limit": str(rate_status.limit),
		"X-RateLimit-Remaining": str(rate_status.remaining),
		"X-RateLimit-Reset": str(int(rate_status.reset_epoch)),
	}
	return JSONResponse(content=jsonable_encoder(sanitised), headers=headers)


def _sanitise_chat_result(result: Dict[str, Any]) -> Dict[str, Any]:
	"""Ensure response payload is JSON serialisable."""
	processed = dict(result)
	trace = processed.get("trace")
	if isinstance(trace, dict) and "raw" in trace:
		raw = trace.get("raw") or {}
		steps = raw.get("intermediate_steps") or []
		simplified_steps: List[Dict[str, Any]] = []
		for step in steps:
			if not isinstance(step, (list, tuple)) or len(step) != 2:
				continue
			action, observation = step
			simplified_steps.append(
				{
					"tool": getattr(action, "tool", None),
					"input": getattr(action, "tool_input", None),
					"observation": observation,
				}
			)
		trace["raw"] = {
			"output": raw.get("output"),
			"log": raw.get("log"),
			"intermediate_steps": simplified_steps,
		}
	return processed


__all__ = ["router"]
