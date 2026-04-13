"""Derived cirrus and ice-support features for atmospheric optics."""

from __future__ import annotations

import math

THIN_CIRRUS_DECAY_K = 2.0


def compute_thin_cirrus(
    cirrus_coverage: float,
    cloud_optical_thickness: float,
    decay_k: float = THIN_CIRRUS_DECAY_K,
) -> float:
    """Estimate the fraction of optically thin cirrus support."""

    if not math.isfinite(cirrus_coverage) or not math.isfinite(cloud_optical_thickness):
        return math.nan
    return _clamp_unit_interval(cirrus_coverage) * math.exp(
        -max(0.0, decay_k) * _clamp_unit_interval(cloud_optical_thickness)
    )


def compute_ice_presence(
    thin_cirrus: float,
    ice_cloud_fraction: float = math.nan,
    *,
    ice_300mb: float = math.nan,
    ice_250mb: float = math.nan,
    ice_200mb: float = math.nan,
) -> float:
    """Estimate ice presence from multi-layer upper-level ice support."""

    multi_layer_weights = (
        (0.3, ice_300mb),
        (0.4, ice_250mb),
        (0.3, ice_200mb),
    )
    available_layer_weights = [weight for weight, value in multi_layer_weights if math.isfinite(value)]
    if available_layer_weights:
        weighted_support = sum(
            weight * _clamp_unit_interval(value)
            for weight, value in multi_layer_weights
            if math.isfinite(value)
        )
        return _clamp_unit_interval(weighted_support / sum(available_layer_weights))

    if not math.isfinite(thin_cirrus):
        return math.nan
    ice_presence = _clamp_unit_interval(thin_cirrus)
    if math.isfinite(ice_cloud_fraction):
        ice_presence *= _clamp_unit_interval(ice_cloud_fraction)
    return _clamp_unit_interval(ice_presence)


def _clamp_unit_interval(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
