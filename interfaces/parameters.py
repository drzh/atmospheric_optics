"""Shared request-parameter parsing for public entry points."""

from __future__ import annotations

from datetime import datetime, timezone

WEATHER_MODES = ("forecast", "observed")
ILLUMINATION_MODES = ("solar", "lunar")
DEFAULT_ILLUMINATION = "solar,lunar"


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


def parse_illumination_modes(value: str | None) -> tuple[str, ...]:
    if value is None or not str(value).strip():
        return tuple(ILLUMINATION_MODES)

    modes: list[str] = []
    invalid: list[str] = []
    for part in str(value).split(","):
        mode = part.strip().lower()
        if not mode:
            continue
        if mode not in ILLUMINATION_MODES:
            invalid.append(mode)
            continue
        if mode not in modes:
            modes.append(mode)
    if invalid:
        expected = ", ".join(ILLUMINATION_MODES + (DEFAULT_ILLUMINATION,))
        raise ValueError(f"Invalid illumination: {value}. Expected one of {expected}")
    if not modes:
        return tuple(ILLUMINATION_MODES)
    return tuple(modes)


def normalize_illumination(value: str | None) -> str:
    return ",".join(parse_illumination_modes(value))


def normalize_weather_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in WEATHER_MODES:
        raise ValueError(f"Invalid mode: {value}. Expected one of {', '.join(WEATHER_MODES)}")
    return mode
