"""Utility helpers for interpreting natural language delivery schedules."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

DAYS_ORDER: List[str] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
FULL_WEEK: str = ",".join(DAYS_ORDER)
DAY_TO_INDEX: Dict[str, int] = {label: idx for idx, label in enumerate(DAYS_ORDER)}

DAY_ALIASES: Dict[str, str] = {
    "mon": "Mon",
    "monday": "Mon",
    "tue": "Tue",
    "tues": "Tue",
    "tuesday": "Tue",
    "wed": "Wed",
    "weds": "Wed",
    "wednesday": "Wed",
    "thu": "Thu",
    "thur": "Thu",
    "thurs": "Thu",
    "thursday": "Thu",
    "fri": "Fri",
    "friday": "Fri",
    "sat": "Sat",
    "saturday": "Sat",
    "sun": "Sun",
    "sunday": "Sun",
}

PHRASE_MAP: Dict[str, str] = {
    "daily": FULL_WEEK,
    "everyday": FULL_WEEK,
    "every day": FULL_WEEK,
    "all week": FULL_WEEK,
    "all days": FULL_WEEK,
    "whole week": FULL_WEEK,
    "7 days": FULL_WEEK,
    "seven days": FULL_WEEK,
    "weekdays": "Mon,Tue,Wed,Thu,Fri",
    "week days": "Mon,Tue,Wed,Thu,Fri",
    "weekday": "Mon,Tue,Wed,Thu,Fri",
    "week ends": "Sat,Sun",
    "weekend": "Sat,Sun",
    "weekend only": "Sat,Sun",
    "weekend-only": "Sat,Sun",
    "weekends": "Sat,Sun",
    "weeknights": "Mon,Tue,Wed,Thu,Fri",
    "mon-fri": "Mon,Tue,Wed,Thu,Fri",
    "monday to friday": "Mon,Tue,Wed,Thu,Fri",
}


@dataclass
class ScheduleParseResult:
    """Structured representation of a parsed delivery schedule phrase."""

    input_phrase: str
    normalized_days: Optional[str]
    labels: List[str]
    schedule_type: str
    start_date: Optional[date]
    end_date: Optional[date]
    notes: List[str]

    def as_dict(self) -> Dict[str, Optional[str]]:
        """Return a JSON-friendly representation of the parse result."""

        return {
            "input": self.input_phrase,
            "normalized_days": self.normalized_days,
            "labels": self.labels,
            "schedule_type": self.schedule_type,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "notes": self.notes,
        }


@dataclass
class ExpiryParseResult:
    """Structured representation of a parsed expiry phrase."""

    input_phrase: str
    reference_date: date
    resolved_date: Optional[date]
    notes: List[str]

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {
            "input": self.input_phrase,
            "reference_date": self.reference_date.isoformat(),
            "resolved_date": self.resolved_date.isoformat() if self.resolved_date else None,
            "notes": self.notes,
        }


def _deduplicate_preserve_order(items: List[str]) -> List[str]:
    """Remove duplicates while preserving the order of items."""

    seen: set[str] = set()
    unique: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _weekday_range(start_label: str, end_label: str) -> List[str]:
    """Return every day label between the given start and end labels."""

    try:
        start_index = DAYS_ORDER.index(start_label)
        end_index = DAYS_ORDER.index(end_label)
    except ValueError:
        return []

    if start_index <= end_index:
        return DAYS_ORDER[start_index : end_index + 1]

    # Wrap around (e.g. Thu-Mon)
    return DAYS_ORDER[start_index:] + DAYS_ORDER[: end_index + 1]


def _week_bounds(reference: date, offset_weeks: int = 0) -> Tuple[date, date]:
    """Return start/end dates for the week at the given offset from reference."""

    start = reference - timedelta(days=reference.weekday()) + timedelta(days=offset_weeks * 7)
    end = start + timedelta(days=6)
    return start, end


def parse_delivery_schedule(phrase: str, reference_date: Optional[date] = None) -> ScheduleParseResult:
    """Parse human-friendly availability text into structured delivery schedule data."""

    reference = reference_date or datetime.now(timezone.utc).date()
    text = (phrase or "").strip().lower()

    if not text:
        return ScheduleParseResult(
            input_phrase=phrase or "",
            normalized_days=None,
            labels=[],
            schedule_type="unknown",
            start_date=None,
            end_date=None,
            notes=["Empty phrase"],
        )

    notes: List[str] = []

    if text in PHRASE_MAP:
        mapping = PHRASE_MAP[text]
        labels = mapping.split(",") if mapping else []
        return ScheduleParseResult(
            input_phrase=phrase,
            normalized_days=mapping or None,
            labels=labels,
            schedule_type="daily" if mapping == FULL_WEEK else "weekly",
            start_date=None,
            end_date=None,
            notes=notes,
        )

    if text in {"next week", "upcoming week"}:
        start, end = _week_bounds(reference, offset_weeks=1)
        return ScheduleParseResult(
            input_phrase=phrase,
            normalized_days=FULL_WEEK,
            labels=DAYS_ORDER.copy(),
            schedule_type="relative_week",
            start_date=start,
            end_date=end,
            notes=[f"Interpreted as week of {start.isoformat()}"]
        )

    if text in {"this week", "current week"}:
        start, end = _week_bounds(reference, offset_weeks=0)
        return ScheduleParseResult(
            input_phrase=phrase,
            normalized_days=FULL_WEEK,
            labels=DAYS_ORDER.copy(),
            schedule_type="relative_week",
            start_date=start,
            end_date=end,
            notes=[f"Interpreted as current week starting {start.isoformat()}"]
        )

    if text in {"next weekend", "upcoming weekend"}:
        start, _ = _week_bounds(reference, offset_weeks=1)
        return ScheduleParseResult(
            input_phrase=phrase,
            normalized_days="Sat,Sun",
            labels=["Sat", "Sun"],
            schedule_type="relative_weekend",
            start_date=start + timedelta(days=5),
            end_date=start + timedelta(days=6),
            notes=["Weekend of upcoming week"],
        )

    if text in {"weekend", "this weekend"}:
        start, _ = _week_bounds(reference, offset_weeks=0)
        return ScheduleParseResult(
            input_phrase=phrase,
            normalized_days="Sat,Sun",
            labels=["Sat", "Sun"],
            schedule_type="relative_weekend",
            start_date=start + timedelta(days=5),
            end_date=start + timedelta(days=6),
            notes=["Weekend of current week"],
        )

    range_match = re.fullmatch(r"(?P<start>[^-]+?)(?:\s*(?:to|-)\s*)(?P<end>.+)", text)
    if range_match:
        start_key = range_match.group("start").strip()
        end_key = range_match.group("end").strip()
        start_label = DAY_ALIASES.get(start_key)
        end_label = DAY_ALIASES.get(end_key)
        if start_label and end_label:
            labels = _weekday_range(start_label, end_label)
            if labels:
                return ScheduleParseResult(
                    input_phrase=phrase,
                    normalized_days=",".join(labels),
                    labels=labels,
                    schedule_type="range",
                    start_date=None,
                    end_date=None,
                    notes=notes,
                )

    pattern = r"(mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)"
    matches = re.findall(pattern, text)
    if matches:
        labels = _deduplicate_preserve_order([DAY_ALIASES[m.lower()] for m in matches])
        if labels:
            return ScheduleParseResult(
                input_phrase=phrase,
                normalized_days=",".join(labels),
                labels=labels,
                schedule_type="list",
                start_date=None,
                end_date=None,
                notes=notes,
            )

    tokens = [token.strip() for token in re.split(r"[,/&\s]+", text) if token.strip()]
    mapped_tokens = [DAY_ALIASES[token] for token in tokens if token in DAY_ALIASES]
    if mapped_tokens:
        labels = _deduplicate_preserve_order(mapped_tokens)
        return ScheduleParseResult(
            input_phrase=phrase,
            normalized_days=",".join(labels),
            labels=labels,
            schedule_type="list",
            start_date=None,
            end_date=None,
            notes=notes,
        )

    notes.append("Could not confidently parse phrase")
    return ScheduleParseResult(
        input_phrase=phrase,
        normalized_days=None,
        labels=[],
        schedule_type="unknown",
        start_date=None,
        end_date=None,
        notes=notes,
    )


def parse_expiry_date(phrase: str, reference_date: Optional[date] = None) -> ExpiryParseResult:
    """Resolve natural-language expiry phrases into a concrete calendar date."""

    reference = reference_date or datetime.now(timezone.utc).date()
    raw_text = (phrase or "").strip()
    lower_text = raw_text.lower()
    notes: List[str] = []

    if not raw_text:
        return ExpiryParseResult(
            input_phrase="",
            reference_date=reference,
            resolved_date=None,
            notes=["Empty phrase"],
        )

    # Attempt ISO date parsing (with optional datetime component)
    try:
        iso_candidate = raw_text.split("T")[0]
        resolved = date.fromisoformat(iso_candidate)
        notes.append("Parsed as ISO date")
        return ExpiryParseResult(raw_text, reference, resolved, notes)
    except ValueError:
        pass

    simple_offsets = {
        "today": 0,
        "now": 0,
        "tomorrow": 1,
        "day after tomorrow": 2,
    }
    if lower_text in simple_offsets:
        resolved = reference + timedelta(days=simple_offsets[lower_text])
        notes.append(f"Resolved using simple offset ({lower_text})")
        return ExpiryParseResult(raw_text, reference, resolved, notes)

    in_match = re.fullmatch(r"in\s+(\d+)\s+(day|days|week|weeks)", lower_text)
    if in_match:
        value = int(in_match.group(1))
        unit = in_match.group(2)
        multiplier = 7 if "week" in unit else 1
        resolved = reference + timedelta(days=value * multiplier)
        notes.append(f"Resolved as {value} {unit} from reference")
        return ExpiryParseResult(raw_text, reference, resolved, notes)

    if lower_text in {"next week", "upcoming week"}:
        start, _ = _week_bounds(reference, offset_weeks=1)
        notes.append("Interpreted as start of next week")
        return ExpiryParseResult(raw_text, reference, start, notes)

    if lower_text in {"this week", "current week"}:
        start, _ = _week_bounds(reference, offset_weeks=0)
        notes.append("Interpreted as start of current week")
        return ExpiryParseResult(raw_text, reference, start, notes)

    if lower_text in {"next weekend", "upcoming weekend"}:
        start, _ = _week_bounds(reference, offset_weeks=1)
        resolved = start + timedelta(days=5)
        notes.append("Interpreted as Saturday of upcoming weekend")
        return ExpiryParseResult(raw_text, reference, resolved, notes)

    if lower_text in {"weekend", "this weekend"}:
        start, _ = _week_bounds(reference, offset_weeks=0)
        resolved = start + timedelta(days=5)
        notes.append("Interpreted as Saturday of current weekend")
        return ExpiryParseResult(raw_text, reference, resolved, notes)

    schedule = parse_delivery_schedule(raw_text, reference)
    if schedule.start_date:
        notes.append("Using schedule start date")
        return ExpiryParseResult(raw_text, reference, schedule.start_date, notes)

    if schedule.labels:
        first_label = schedule.labels[0]
        target_index = DAY_TO_INDEX.get(first_label)
        if target_index is not None:
            delta = (target_index - reference.weekday()) % 7
            if delta == 0:
                delta = 7
            resolved = reference + timedelta(days=delta)
            notes.append(f"Next occurrence of {first_label}")
            return ExpiryParseResult(raw_text, reference, resolved, notes)

    notes.extend(schedule.notes)
    notes.append("Unable to resolve to a concrete date")
    return ExpiryParseResult(raw_text, reference, None, notes)
