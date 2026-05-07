"""Temporal smoothing helpers for atmospheric optics prediction."""

from __future__ import annotations

import math
from typing import Sequence


ASYMMETRIC_ONSET_SCALE_HOURS = 0.5
ASYMMETRIC_DECAY_SCALE_HOURS = 1.5
PEAK_PRESERVATION_THRESHOLD = 0.6


def smooth_probabilities(
    hours: Sequence[int | float],
    probabilities: Sequence[float],
    *,
    preserve_peak: bool = True,
    preserve_peak_threshold: float = PEAK_PRESERVATION_THRESHOLD,
) -> list[float]:
    """Smooth a short timeline while preserving strong peaks in place."""

    offsets = _normalized_hours(hours)
    values = _normalized_probabilities(probabilities)
    if len(offsets) != len(values):
        raise ValueError("hours and probabilities must contain the same number of elements")
    if len(values) < 2:
        return values

    smoothed: list[float] = []
    for target_hour in offsets:
        weighted_sum = 0.0
        total_weight = 0.0
        for source_hour, probability in zip(offsets, values):
            distance = target_hour - source_hour
            weight = asymmetric_kernel(distance)
            weighted_sum += probability * weight
            total_weight += weight
        smoothed.append(_clamp_unit_interval(weighted_sum / total_weight if total_weight else 0.0))

    if preserve_peak:
        raw_peak_index = peak_index(values)
        if raw_peak_index is not None and values[raw_peak_index] >= preserve_peak_threshold:
            smoothed[raw_peak_index] = max(
                smoothed[raw_peak_index],
                values[raw_peak_index],
            )

    return [_clamp_unit_interval(value) for value in smoothed]


def asymmetric_kernel(dt: float) -> float:
    """Return a sharper backward-looking kernel and slower forward decay."""

    if not math.isfinite(dt):
        return 0.0
    scale = ASYMMETRIC_ONSET_SCALE_HOURS if dt < 0.0 else ASYMMETRIC_DECAY_SCALE_HOURS
    scale = max(scale, 1.0e-6)
    return math.exp(-(abs(dt) / scale))


def temporal_stability(hours: Sequence[int | float], probabilities: Sequence[float]) -> float:
    """Return a 0-1 penalty term for rapid timeline oscillations."""

    offsets = _normalized_hours(hours)
    values = _normalized_probabilities(probabilities)
    if len(offsets) != len(values):
        raise ValueError("hours and probabilities must contain the same number of elements")
    if len(values) < 2:
        return 1.0

    absolute_rates: list[float] = []
    signed_rates: list[float] = []
    for previous_hour, current_hour, previous_value, current_value in zip(
        offsets,
        offsets[1:],
        values,
        values[1:],
    ):
        delta_hours = max(current_hour - previous_hour, 1.0e-6)
        rate = (current_value - previous_value) / delta_hours
        absolute_rates.append(abs(rate))
        signed_rates.append(rate)

    curvature_terms: list[float] = []
    for index in range(1, len(values) - 1):
        left_delta_hours = max(offsets[index] - offsets[index - 1], 1.0e-6)
        right_delta_hours = max(offsets[index + 1] - offsets[index], 1.0e-6)
        left_slope = (values[index] - values[index - 1]) / left_delta_hours
        right_slope = (values[index + 1] - values[index]) / right_delta_hours
        curvature_terms.append(abs(right_slope - left_slope))

    direction_changes = 0.0
    if len(signed_rates) > 1:
        direction_changes = (
            sum(1 for left_rate, right_rate in zip(signed_rates, signed_rates[1:]) if left_rate * right_rate < 0.0)
            / (len(signed_rates) - 1)
        )

    mean_rate = sum(absolute_rates) / len(absolute_rates)
    mean_curvature = sum(curvature_terms) / len(curvature_terms) if curvature_terms else 0.0
    rate_penalty = min(mean_rate / 0.22, 1.0)
    curvature_penalty = min(mean_curvature / 0.18, 1.0)
    stability = 1.0 - (0.55 * rate_penalty) - (0.30 * curvature_penalty) - (0.15 * direction_changes)
    return _clamp_unit_interval(stability)


def temporal_consistency(probabilities: Sequence[float]) -> float:
    """Quantify how stable a short forecast timeline remains overall."""

    values = [value for value in _normalized_probabilities(probabilities) if math.isfinite(value)]
    if len(values) < 2:
        return 1.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return _clamp_unit_interval(1.0 - variance)


def resolve_peak_index(
    raw_probabilities: Sequence[float],
    smoothed_probabilities: Sequence[float],
    *,
    preserve_peak_threshold: float = PEAK_PRESERVATION_THRESHOLD,
) -> int | None:
    """Prefer the raw peak time when the unsmoothed signal is already strong."""

    raw_values = _normalized_probabilities(raw_probabilities)
    smoothed_values = _normalized_probabilities(smoothed_probabilities)
    raw_peak = peak_index(raw_values)
    if raw_peak is not None and raw_values[raw_peak] > preserve_peak_threshold:
        return raw_peak
    return peak_index(smoothed_values)


def peak_index(probabilities: Sequence[float]) -> int | None:
    values = _normalized_probabilities(probabilities)
    if not values:
        return None
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def _normalized_hours(hours: Sequence[int | float]) -> list[float]:
    return [float(hour) for hour in hours]


def _normalized_probabilities(probabilities: Sequence[float]) -> list[float]:
    values: list[float] = []
    for probability in probabilities:
        try:
            numeric_probability = float(probability)
        except (TypeError, ValueError):
            numeric_probability = 0.0
        values.append(_clamp_unit_interval(numeric_probability))
    return values


def _clamp_unit_interval(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
