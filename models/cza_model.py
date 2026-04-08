"""Circumzenithal arc probability rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from models.probability import (
    DEFAULT_WEIGHTS,
    ModelWeights,
    combine_probability,
    gaussian,
    numeric_feature,
    sigmoid,
    unit_feature,
)


@dataclass(frozen=True)
class CzaParameters:
    ice_sigmoid_k: float = 0.2
    ice_sigmoid_x0: float = 20.0
    geometry_mu: float = 22.0
    geometry_sigma: float = 5.0
    weights: ModelWeights = field(default_factory=lambda: DEFAULT_WEIGHTS)


DEFAULT_CZA_PARAMETERS = CzaParameters()


def predict_cza(
    features: Mapping[str, float],
    parameters: CzaParameters = DEFAULT_CZA_PARAMETERS,
) -> float:
    """Predict the probability of a circumzenithal arc."""

    p_ice = sigmoid(
        -numeric_feature(features, "temp_250"),
        k=parameters.ice_sigmoid_k,
        x0=parameters.ice_sigmoid_x0,
    )
    p_cirrus = unit_feature(features, "cirrus_coverage")
    physical = p_ice * p_cirrus

    sun_visible = unit_feature(features, "sun_visible")
    cloud_optical_thickness = unit_feature(features, "cloud_optical_thickness")
    visibility = sun_visible * (1.0 - cloud_optical_thickness)

    geometry = gaussian(
        numeric_feature(features, "solar_elevation"),
        mu=parameters.geometry_mu,
        sigma=parameters.geometry_sigma,
    )

    return combine_probability(physical, visibility, geometry, parameters.weights)
