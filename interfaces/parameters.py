"""Shared request-parameter parsing for public entry points."""

from __future__ import annotations

from datetime import datetime, timezone

WEATHER_MODES = ("forecast", "observed")
ILLUMINATION_MODES = ("solar", "lunar")


def parse_at_time(value: str | None) -> datetime | None:
    if value is None or not value.strip():
        return None

    parsed_value = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed_value.tzinfo is None:
        return parsed_value.replace(tzinfo=timezone.utc)
    return parsed_value.astimezone(timezone.utc)


def parse_time_window_hours(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None

    hours: list[int] = []
    for part in value.split(","):
        normalized_part = part.strip()
        if normalized_part:
            hours.append(int(normalized_part))
    return tuple(hours)


def parse_csv_values(value: str | None) -> tuple[str, ...] | None:
    if value is None or not value.strip():
        return None

    items = tuple(part.strip() for part in value.split(",") if part.strip())
    return items or None


def normalize_illumination(value: str) -> str:
    illumination = str(value).strip().lower()
    if illumination not in ILLUMINATION_MODES:
        raise ValueError(f"Invalid illumination: {value}. Expected one of {', '.join(ILLUMINATION_MODES)}")
    return illumination


def normalize_weather_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in WEATHER_MODES:
        raise ValueError(f"Invalid mode: {value}. Expected one of {', '.join(WEATHER_MODES)}")
    return mode
