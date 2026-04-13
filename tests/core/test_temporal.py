from __future__ import annotations

import pytest

from core.temporal import (
    peak_index,
    resolve_peak_index,
    smooth_probabilities,
    temporal_consistency,
    temporal_stability,
)


def test_smooth_probabilities_preserves_strong_peak_index() -> None:
    hours = [0, 1, 2, 3]
    probabilities = [0.2, 0.35, 0.92, 0.5]

    smoothed = smooth_probabilities(hours, probabilities, preserve_peak=True)

    assert peak_index(smoothed) == 2
    assert smoothed[2] >= probabilities[2]


def test_asymmetric_smoothing_limits_backward_peak_drift() -> None:
    hours = [0, 1, 2, 3]
    probabilities = [0.1, 0.2, 0.85, 0.6]

    smoothed = smooth_probabilities(hours, probabilities, preserve_peak=True)

    assert smoothed[1] < smoothed[2]
    assert resolve_peak_index(probabilities, smoothed) == 2


def test_temporal_metrics_penalize_oscillation() -> None:
    hours = [0, 1, 2, 3]
    stable = [0.2, 0.24, 0.27, 0.3]
    oscillatory = [0.2, 0.7, 0.15, 0.8]

    assert temporal_stability(hours, stable) > temporal_stability(hours, oscillatory)
    assert temporal_consistency(stable) > temporal_consistency(oscillatory)
    assert 0.0 <= temporal_stability(hours, oscillatory) <= 1.0
    assert 0.0 <= temporal_consistency(oscillatory) <= 1.0
