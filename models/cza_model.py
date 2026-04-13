"""Circumzenithal arc probability rules."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from models.combine import ModelComponents, combine_log
from models.probability import (
    DEFAULT_WEIGHTS,
    ModelWeights,
    gaussian,
    numeric_feature,
    unit_feature,
)


@dataclass(frozen=True)
class CzaParameters:
    geometry_mu: float = 22.0
    geometry_sigma: float = 5.0
    weights: ModelWeights = field(default_factory=lambda: DEFAULT_WEIGHTS)


DEFAULT_CZA_PARAMETERS = CzaParameters()


def predict_cza(
    features: Mapping[str, float],
    parameters: CzaParameters = DEFAULT_CZA_PARAMETERS,
    *,
    return_components: bool = False,
) -> float | ModelComponents:
    """Predict the probability of a circumzenithal arc."""

    physical = unit_feature(features, "ice_presence") * unit_feature(features, "thin_cirrus")
    visibility = unit_feature(features, "sun_visible") * math.exp(
        -unit_feature(features, "cloud_optical_thickness")
    )
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
