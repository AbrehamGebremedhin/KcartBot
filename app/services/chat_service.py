"""Chat service for managing KCartBot conversations and sessions."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional
from uuid import uuid4

from app.agents.agent import Agent

logger = logging.getLogger(__name__)


class ChatService:
    """Service for managing chat conversations with the KCartBot agent."""

    def __init__(self) -> None:
        """Initialize the chat service with an agent instance."""
        self.agent = Agent()
        # In production, this would be a proper database/cache
        self._sessions: Dict[str, Dict[str, Any]] = {}

    async def process_message(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user message in the context of a conversation session.

        Args:
            user_message: The user's input message
            session_id: Optional session ID. If not provided, a new session is created.
            user_context: Optional user context (e.g., user_id, language preference)

        Returns:
            Dict containing response and session information
        """
        try:
            # Get or create session
            if not session_id:
                session_id = str(uuid4())
                self._sessions[session_id] = self._create_new_session(user_context or {}, session_id)
                logger.info(f"Created new session: {session_id}")
            elif session_id not in self._sessions:
                # Use the provided session_id to create a new session
                self._sessions[session_id] = self._create_new_session(user_context or {}, session_id)
                logger.info(f"Created new session with provided ID: {session_id}")
            else:
                logger.info(f"Using existing session: {session_id}")

            session_context = self._sessions[session_id]

            # Update session with user context if provided
            if user_context:
                session_context.update(user_context)

            # Process the message with the agent
            result = await self.agent.process_message(user_message, session_context)

            # Update the session with the new context
            self._sessions[session_id] = result.get("session_context", session_context)

            # Prepare response
            response = {
                "session_id": session_id,
                "response": result.get("response", ""),
                "chat_history": session_context.get("chat_history", []),
                "intent_info": result.get("intent_info", {}),
                "timestamp": self._get_current_timestamp(),
            }

            # Add any additional metadata
            if "error" in result:
                response["error"] = result["error"]

            return response

        except Exception as exc:
            logger.error(f"Error processing message: {exc}")
            return {
                "session_id": session_id or str(uuid4()),
                "response": "I'm sorry, I encountered an error. Please try again.",
                "chat_history": self._sessions.get(session_id, {}).get("chat_history", []) if session_id else [],
                "error": str(exc),
                "timestamp": self._get_current_timestamp(),
            }

    def _create_new_session(self, user_context: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new conversation session."""
        return {
            "session_id": session_id or str(uuid4()),
            "created_at": self._get_current_timestamp(),
            "chat_history": [],
            "current_intent": None,
            "current_flow": None,
            "filled_slots": {},
            "missing_slots": [],
            "user_id": user_context.get("user_id"),
            "user_role": user_context.get("user_role"),
            "preferred_language": user_context.get("preferred_language", "English"),
            "last_activity": self._get_current_timestamp(),
        }

    def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the context for a specific session."""
        return self._sessions.get(session_id)

    def update_session_context(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update the context for a specific session."""
        if session_id not in self._sessions:
            return False

        self._sessions[session_id].update(updates)
        self._sessions[session_id]["last_activity"] = self._get_current_timestamp()
        return True

    def end_session(self, session_id: str) -> bool:
        """End a conversation session."""
        if session_id in self._sessions:
            # Could add cleanup logic here
            del self._sessions[session_id]
            logger.info(f"Ended session: {session_id}")
            return True
        return False

    def list_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """List all active sessions (for debugging/admin purposes)."""
        # Return a copy without sensitive data
        return {
            sid: {
                k: v for k, v in session.items()
                if k not in ["chat_history"]  # Exclude potentially large chat history
            }
            for sid, session in self._sessions.items()
        }

    def cleanup_inactive_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up sessions that haven't been active for the specified hours."""
        import datetime

        now = datetime.datetime.utcnow()
        cutoff = now - datetime.timedelta(hours=max_age_hours)

        sessions_to_remove = []
        for session_id, session in self._sessions.items():
            last_activity = session.get("last_activity")
            if last_activity:
                try:
                    # Parse ISO timestamp
                    last_activity_dt = datetime.datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
                    if last_activity_dt < cutoff:
                        sessions_to_remove.append(session_id)
                except (ValueError, AttributeError):
                    # If we can't parse the timestamp, assume it's old
                    sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self._sessions[session_id]
            logger.info(f"Cleaned up inactive session: {session_id}")

        return len(sessions_to_remove)

    @staticmethod
    def _get_current_timestamp() -> str:
        """Get current timestamp in ISO format."""
        import datetime
        return datetime.datetime.utcnow().isoformat() + "Z"

    async def get_conversation_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of the conversation for a session."""
        session = self.get_session_context(session_id)
        if not session:
            return None

        chat_history = session.get("chat_history", [])
        message_count = len(chat_history)

        if message_count == 0:
            return {
                "session_id": session_id,
                "message_count": 0,
                "summary": "No messages yet",
                "current_intent": session.get("current_intent"),
                "current_flow": session.get("current_flow"),
            }

        # Get last few messages for summary
        recent_messages = chat_history[-6:]  # Last 3 exchanges

        # Create a simple summary
        user_messages = [msg for msg in recent_messages if msg.get("role") == "user"]
        assistant_messages = [msg for msg in recent_messages if msg.get("role") == "assistant"]

        summary = {
            "session_id": session_id,
            "message_count": message_count,
            "last_user_message": user_messages[-1]["content"] if user_messages else None,
            "last_assistant_message": assistant_messages[-1]["content"] if assistant_messages else None,
            "current_intent": session.get("current_intent"),
            "current_flow": session.get("current_flow"),
            "user_role": session.get("user_role"),
            "preferred_language": session.get("preferred_language"),
        }

        return summary
