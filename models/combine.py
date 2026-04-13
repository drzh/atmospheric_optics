"""Probability combination helper for atmospheric optics models."""

from __future__ import annotations

import math
from dataclasses import dataclass

from models.probability import DEFAULT_WEIGHTS, ModelWeights, clamp, sigmoid

LOG_COMBINE_EPSILON = 1e-6


@dataclass(frozen=True)
class ModelComponents:
    probability: float
    physical: float
    visibility: float
    geometry: float


def combine_log(
    physical: float,
    visibility: float,
    geometry: float,
    weights: ModelWeights = DEFAULT_WEIGHTS,
    epsilon: float = LOG_COMBINE_EPSILON,
    *,
    return_components: bool = False,
) -> float | ModelComponents:
    """Combine model components with the weighted log formulation."""

    clamped_physical = clamp(physical)
    clamped_visibility = clamp(visibility)
    clamped_geometry = clamp(geometry)
    safe_epsilon = epsilon if math.isfinite(epsilon) and epsilon > 0.0 else LOG_COMBINE_EPSILON
    combined_score = (
        (weights.physical * math.log(clamped_physical + safe_epsilon))
        + (weights.visibility * math.log(clamped_visibility + safe_epsilon))
        + (weights.geometry * math.log(clamped_geometry + safe_epsilon))
    )
    probability = clamp(sigmoid(combined_score))
    if return_components:
        return ModelComponents(
            probability=probability,
            physical=clamped_physical,
            visibility=clamped_visibility,
            geometry=clamped_geometry,
        )
    return probability
