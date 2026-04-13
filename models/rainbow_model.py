"""Rainbow probability rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from models.combine import ModelComponents, combine_log
from models.probability import (
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


def predict_rainbow(
    features: Mapping[str, float],
    parameters: RainbowParameters = DEFAULT_RAINBOW_PARAMETERS,
    *,
    return_components: bool = False,
) -> float | ModelComponents:
    """Predict the probability of rainbow conditions."""

    physical = sigmoid(
        nonnegative_feature(features, "precipitation"),
        k=parameters.precipitation_sigmoid_k,
        x0=parameters.precipitation_sigmoid_x0,
    )
    visibility = unit_feature(features, "sun_visible")
    geometry = gaussian(
        numeric_feature(features, "solar_elevation"),
        mu=parameters.geometry_mu,
        sigma=parameters.geometry_sigma,
    )

    return combine_log(
        physical,
        visibility,
        geometry,
        parameters.weights,
        return_components=return_components,
    )
