"""Tool for resolving natural language dates to actual date objects."""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.llm_service import LLMService
from app.tools.base import ToolBase

logger = logging.getLogger(__name__)


class DateResolverTool(ToolBase):
    """Tool that resolves natural language dates to datetime.date objects."""

    def __init__(
        self,
        llm_service: Optional["LLMService"] = None,
    ):
        super().__init__(
            name="schedule_helper",
            description=(
                "Resolve natural language date expressions (e.g., 'today', 'tomorrow', 'next week') "
                "to actual date objects using datetime and LLM assistance."
            ),
        )
        self._llm_service = llm_service

    async def run(self, input: Any, context: Optional[Dict[str, Any]] = None) -> datetime.date:
        """Resolve the natural language date to a datetime.date object.

        Args:
            input: Natural language date string (e.g., "today", "next Monday").
            context: Optional context dictionary.

        Returns:
            datetime.date: The resolved date.

        Raises:
            ValueError: If the date cannot be resolved.
        """
        if isinstance(input, dict):
            date_text = input.get("date") or input.get("text") or str(input)
        else:
            date_text = str(input or "").strip()

        if not date_text:
            raise ValueError("No date text provided.")

        today = datetime.date.today()

        # For simple cases, handle directly
        date_text_lower = date_text.lower()
        if date_text_lower in ["today", "now"]:
            return today
        elif date_text_lower == "tomorrow":
            return today + datetime.timedelta(days=1)
        elif date_text_lower == "yesterday":
            return today - datetime.timedelta(days=1)

        # For more complex cases, use LLM
        from app.services.llm_service import LLMService
        llm = self._llm_service or LLMService()
        prompt = f"""
            Today is {today.strftime('%Y-%m-%d')} ({today.strftime('%A, %B %d, %Y')}).

            Resolve the following natural language date expression to an exact date in YYYY-MM-DD format.
            If it's relative to today, calculate accordingly.
            If it's ambiguous, make a reasonable assumption based on current context.

            Expression: "{date_text}"

            Return only the date in YYYY-MM-DD format, nothing else.
        """

        try:
            response = await llm.acomplete(prompt)
            resolved_date_str = response.strip()
            # Validate the format
            resolved_date = datetime.datetime.strptime(resolved_date_str, "%Y-%m-%d").date()
            return resolved_date
        except Exception as exc:
            logger.error("Failed to resolve date '%s': %s", date_text, exc)
            raise ValueError(f"Could not resolve date expression '{date_text}' to a valid date.") from exc
