from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, AliasChoices

from app.tools.base import ToolBase
from app.utils.schedule import parse_delivery_schedule, parse_expiry_date


class ScheduleToolRequest(BaseModel):
	operation: str = Field(description="Operation to execute: 'parse_schedule' or 'expiry_date'.")
	phrase: str = Field(
		description="Natural language phrase describing a schedule or expiry.",
		validation_alias=AliasChoices("phrase", "natural_language_phrase", "text"),
	)
	reference_date: Optional[str] = Field(
		default=None,
		description="Optional ISO date (YYYY-MM-DD) to anchor relative phrases.",
	)

	@field_validator("operation", mode="before")
	@classmethod
	def _normalise_operation(cls, value: Any) -> str:
		if not isinstance(value, str):
			raise TypeError("operation must be a string")
		return value.strip().lower()

	@field_validator("phrase", mode="before")
	@classmethod
	def _ensure_phrase(cls, value: Any) -> str:
		if not isinstance(value, str):
			raise TypeError("phrase must be a string")
		return value

	model_config = {"populate_by_name": True}


class ScheduleTool(ToolBase):
	"""Tool for normalising delivery schedules and resolving expiry phrases."""

	def __init__(self) -> None:
		super().__init__(
			name="schedule_helper",
			description=(
				"Interpret natural-language scheduling phrases. "
				"Operations: 'parse_schedule' (returns normalised delivery schedule) and "
				"'expiry_date' (resolves to a concrete calendar date)."
			),
		)

	async def run(self, input: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		try:
			if isinstance(input, str):
				try:
					input = json.loads(input)
				except json.JSONDecodeError:
					return {"error": "Invalid JSON payload for schedule_helper tool."}
			request = ScheduleToolRequest.model_validate(input)
		except ValidationError as exc:
			return {"error": "Invalid schedule_helper request.", "details": exc.errors()}
		except Exception as exc:  # pragma: no cover - defensive
			return {"error": f"Unexpected validation error: {exc}"}

		reference_date = self._resolve_reference_date(request.reference_date, context)
		if isinstance(reference_date, str):
			return {"error": reference_date}

		operation = request.operation
		phrase = request.phrase

		if operation == "parse_schedule":
			result = parse_delivery_schedule(phrase, reference_date)
			return result.as_dict()

		if operation == "expiry_date":
			result = parse_expiry_date(phrase, reference_date)
			return result.as_dict()

		return {
			"error": f"Unknown operation: {operation}",
			"valid_operations": ["parse_schedule", "expiry_date"],
		}

	@staticmethod
	def _resolve_reference_date(
		reference_raw: Optional[str],
		context: Optional[Dict[str, Any]],
	) -> Optional[date] | str:
		today = datetime.now(timezone.utc).date()
		if reference_raw is not None:
			try:
				iso_value = reference_raw.split("T")[0]
				parsed = date.fromisoformat(iso_value)
				return parsed if parsed >= today else today
			except ValueError:
				return "reference_date must be a valid ISO date (YYYY-MM-DD)."

		if context:
			for key in ("current_date", "reference_date", "today"):
				value = context.get(key)
				if isinstance(value, date):
					return value if value >= today else today
				if isinstance(value, str):
					try:
						iso_value = value.split("T")[0]
						parsed = date.fromisoformat(iso_value)
						return parsed if parsed >= today else today
					except ValueError:
						continue

		return today
