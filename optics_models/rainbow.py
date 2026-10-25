"""Rainbow probability rules."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from optics_models.combination import ModelComponents, combine_log
from optics_models.probability import (
    DEFAULT_WEIGHTS,
    ModelWeights,
    gaussian,
    nonnegative_feature,
    numeric_feature,
    sigmoid,
    unit_feature,
)


@dataclass(frozen=True)
class RainbowParameters:
    precipitation_sigmoid_k: float = 5.0
    precipitation_sigmoid_x0: float = 0.1
    geometry_mu: float = 20.0
    geometry_sigma: float = 15.0
    weights: ModelWeights = field(default_factory=lambda: DEFAULT_WEIGHTS)


DEFAULT_RAINBOW_PARAMETERS = RainbowParameters()
PRIMARY_BOW_MAX_ELEVATION = 42.0


def _source_elevation(features: Mapping[str, float]) -> float:
    elevation = numeric_feature(features, "source_elevation")
    if math.isfinite(elevation):
        return elevation
    return numeric_feature(features, "solar_elevation")


def predict_rainbow(
    features: Mapping[str, float],
    parameters: RainbowParameters = DEFAULT_RAINBOW_PARAMETERS,
    *,
    return_components: bool = False,
) -> float | ModelComponents:
    """Predict the probability of rainbow conditions."""

    precipitation = nonnegative_feature(features, "precipitation")
    dry_baseline = sigmoid(0.0, k=parameters.precipitation_sigmoid_k, x0=parameters.precipitation_sigmoid_x0)
    rain_response = sigmoid(
        precipitation,
        k=parameters.precipitation_sigmoid_k,
        x0=parameters.precipitation_sigmoid_x0,
    )
    # A logistic offset alone assigns substantial support to completely dry air.
    physical = max(0.0, rain_response - dry_baseline) / max(1e-9, 1.0 - dry_baseline)
    visibility = (
        unit_feature(features, "source_visible", default=unit_feature(features, "sun_visible"))
        * math.exp(-unit_feature(features, "cloud_optical_thickness"))
        * unit_feature(features, "brightness_factor", default=1.0)
    )
    source_elevation = _source_elevation(features)
    geometry = gaussian(
        source_elevation,
        mu=parameters.geometry_mu,
        sigma=parameters.geometry_sigma,
    ) if 0.0 <= source_elevation < PRIMARY_BOW_MAX_ELEVATION else 0.0

    return combine_log(
        physical,
        visibility,
        geometry,
        parameters.weights,
        return_components=return_components,
    )
