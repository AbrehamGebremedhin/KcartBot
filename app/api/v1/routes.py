"""API routes for KCartBot v1 endpoints."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.chat_service import ChatService

router = APIRouter()

# Initialize chat service
chat_service = ChatService()


class ChatRequest(BaseModel):
    """Request model for chat messages."""

    message: str = Field(..., description="The user's message")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation continuity")
    user_context: Optional[Dict[str, Any]] = Field(None, description="Optional user context (user_id, language, etc.)")


class ChatResponse(BaseModel):
    """Response model for chat messages."""

    session_id: str = Field(..., description="Session ID for the conversation")
    response: str = Field(..., description="The assistant's response")
    chat_history: List[Dict[str, Any]] = Field(default_factory=list, description="Full conversation history for the session")
    intent_info: Optional[Dict[str, Any]] = Field(None, description="Intent classification information")
    timestamp: str = Field(..., description="Response timestamp")
    error: Optional[str] = Field(None, description="Error message if any")


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat_with_bot(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the KCartBot assistant.

    This endpoint processes user messages and returns responses based on the conversation context.
    Supports both customer and supplier flows with intent classification and tool integration.
    """
    try:
        result = await chat_service.process_message(
            user_message=request.message,
            session_id=request.session_id,
            user_context=request.user_context
        )

        return ChatResponse(**result)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(exc)}"
        )


__all__ = ["router"]
