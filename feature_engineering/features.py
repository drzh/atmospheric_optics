"""Feature engineering for atmospheric optics rules."""

from __future__ import annotations

import math
from typing import Mapping

OPTICAL_THICKNESS_PROXY_SCALE = 0.5


def compute_features(
    weather: Mapping[str, float],
    solar: Mapping[str, float],
) -> dict[str, float]:
    """Compute rule-friendly features from weather and solar inputs."""

    solar_elevation = _coerce_float(solar.get("elevation"))
    cirrus_coverage = _normalize_fraction(
        weather.get("cirrus_coverage", weather.get("cloud_cover_high"))
    )
    cloud_optical_thickness = _normalize_fraction(weather.get("cloud_optical_thickness"))
    if not math.isfinite(cloud_optical_thickness) and math.isfinite(cirrus_coverage):
        cloud_optical_thickness = _clamp_unit_interval(
            cirrus_coverage * OPTICAL_THICKNESS_PROXY_SCALE
        )

    return {
        "cirrus_coverage": cirrus_coverage,
        "cloud_optical_thickness": cloud_optical_thickness,
        "temp_250": _coerce_float(weather.get("temp_250")),
        "humidity_250": _normalize_fraction(weather.get("humidity_250")),
        "solar_elevation": solar_elevation,
        "sun_visible": 1.0 if math.isfinite(solar_elevation) and solar_elevation > 0.0 else 0.0,
        "precipitation": _nonnegative_or_nan(weather.get("precipitation")),
    }


def _coerce_float(value: object) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _normalize_fraction(value: object) -> float:
    coerced_value = _coerce_float(value)
    if not math.isfinite(coerced_value):
        return math.nan
    if coerced_value > 1.0:
        coerced_value /= 100.0
    return _clamp_unit_interval(coerced_value)


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, value))


def _nonnegative_or_nan(value: object) -> float:
    coerced_value = _coerce_float(value)
    if not math.isfinite(coerced_value):
        return math.nan
    return max(0.0, coerced_value)
