"""Shared helpers for the quantitative atmospheric optics models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ModelWeights:
    physical: float = 1.2
    visibility: float = 1.0
    geometry: float = 1.1


DEFAULT_WEIGHTS = ModelWeights()


def sigmoid(x: float, k: float = 1.0, x0: float = 0.0) -> float:
    """Return a smooth threshold response."""

    if not math.isfinite(x) or not math.isfinite(k) or not math.isfinite(x0):
        return 0.0
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


def gaussian(x: float, mu: float, sigma: float) -> float:
    """Return a peaked preference curve."""

    if not math.isfinite(x) or not math.isfinite(mu) or not math.isfinite(sigma) or sigma <= 0.0:
        return 0.0
    return math.exp(-((x - mu) ** 2) / (2.0 * (sigma**2)))


def clamp(value: float) -> float:
    """Clamp any numeric value to the unit interval."""

    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def numeric_feature(features: Mapping[str, float], key: str, default: float = math.nan) -> float:
    """Read a numeric feature without assuming it is present."""

    value = features.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def unit_feature(features: Mapping[str, float], key: str, default: float = 0.0) -> float:
    """Read and clamp a normalized feature."""

    value = numeric_feature(features, key, default=math.nan)
    if not math.isfinite(value):
        return clamp(default)
    return clamp(value)


def nonnegative_feature(features: Mapping[str, float], key: str, default: float = 0.0) -> float:
    """Read a non-negative scalar feature such as precipitation."""

    value = numeric_feature(features, key, default=math.nan)
    if not math.isfinite(value):
        return max(0.0, default)
    return max(0.0, value)


def combine_probability(
    physical: float,
    visibility: float,
    geometry: float,
    weights: ModelWeights = DEFAULT_WEIGHTS,
) -> float:
    """Combine model components with the recommended weighting adjustment."""

    physical_term = clamp(physical) ** weights.physical
    visibility_term = clamp(visibility) ** weights.visibility
    geometry_term = clamp(geometry) ** weights.geometry
    return clamp(physical_term * visibility_term * geometry_term)
