"""Derived alignment and cloud-structure features for atmospheric optics."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

PLATE_ALIGNMENT_DECAY_K = 1.0
VERTICAL_VELOCITY_DECAY_K = 1.0
DEFAULT_PLATE_ALIGNMENT = 0.5
DEFAULT_CLOUD_VARIABILITY = 0.3
DEFAULT_WIND_STABILITY = 0.5


def compute_plate_alignment(
    wind_shear_250: float,
    vertical_velocity_variance: float = math.nan,
    decay_k: float = PLATE_ALIGNMENT_DECAY_K,
    vertical_velocity_decay_k: float = VERTICAL_VELOCITY_DECAY_K,
    default: float = DEFAULT_PLATE_ALIGNMENT,
) -> float:
    """Estimate crystal plate alignment from upper-air wind shear."""

    if not math.isfinite(wind_shear_250):
        return _clamp_unit_interval(default)
    alignment = math.exp(-max(0.0, wind_shear_250) * max(0.0, decay_k))
    if math.isfinite(vertical_velocity_variance):
        alignment *= math.exp(
            -max(0.0, vertical_velocity_variance) * max(0.0, vertical_velocity_decay_k)
        )
    return _clamp_unit_interval(alignment)


def compute_cloud_variability(
    cloud_cover_input: object,
    default: float = DEFAULT_CLOUD_VARIABILITY,
) -> float:
    """Estimate structured cloud variability from a 3x3 neighborhood or fallback value."""

    if isinstance(cloud_cover_input, Sequence) and not isinstance(cloud_cover_input, (str, bytes, bytearray)):
        values = [
            _normalize_fraction(value)
            for value in cloud_cover_input
            if math.isfinite(_normalize_fraction(value))
        ]
        if len(values) >= 2:
            return _clamp_unit_interval(statistics.pstdev(values))

    scalar_value = _normalize_fraction(cloud_cover_input)
    if math.isfinite(scalar_value):
        return scalar_value

    return _clamp_unit_interval(default)


def compute_wind_stability(
    wind_shear_250: float,
    vertical_velocity_variance: float = math.nan,
    *,
    default: float = DEFAULT_WIND_STABILITY,
) -> float:
    """Estimate how stable the upper-air flow remains for aligned crystals."""

    if not math.isfinite(wind_shear_250) and not math.isfinite(vertical_velocity_variance):
        return _clamp_unit_interval(default)

    wind_shear = max(0.0, wind_shear_250) if math.isfinite(wind_shear_250) else 0.0
    vertical_velocity_term = (
        max(0.0, vertical_velocity_variance)
        if math.isfinite(vertical_velocity_variance)
        else 0.0
    )
    return _clamp_unit_interval(math.exp(-(wind_shear + vertical_velocity_term)))


def _normalize_fraction(value: object) -> float:
    if value is None:
        return math.nan
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(numeric_value):
        return math.nan
    if numeric_value > 1.0:
        numeric_value /= 100.0
    return _clamp_unit_interval(numeric_value)


def _clamp_unit_interval(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
