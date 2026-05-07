"""Deterministic GOES texture classification helpers."""

from __future__ import annotations

import math
from statistics import pstdev
from typing import Iterable


def classify_texture(values: Iterable[object], *, optical_thickness: float | None = None) -> str | None:
    """Classify cloud morphology from a small satellite cloud-fraction window."""

    numeric_values = [_coerce_fraction(value) for value in values]
    finite_values = [value for value in numeric_values if math.isfinite(value)]
    if len(finite_values) < 2:
        return None

    variability = pstdev(finite_values)
    mean_cover = sum(finite_values) / len(finite_values)
    finite_optical_thickness = _coerce_fraction(optical_thickness)

    if variability >= 0.28:
        return "cellular"
    if variability >= 0.18:
        return "granular" if mean_cover < 0.55 else "cellular"
    if math.isfinite(finite_optical_thickness) and finite_optical_thickness <= 0.2 and mean_cover <= 0.55:
        return "fibrous"
    return "smooth"


def _coerce_fraction(value: object) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(numeric_value):
        return math.nan
    if numeric_value > 1.0:
        numeric_value /= 100.0
    return max(0.0, min(1.0, numeric_value))
