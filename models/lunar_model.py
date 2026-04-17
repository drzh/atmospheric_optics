"""Lunar-specific atmospheric optics rules."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from models.combine import ModelComponents, combine_log
from models.probability import DEFAULT_WEIGHTS, ModelWeights, gaussian, unit_feature


@dataclass(frozen=True)
class LunarCoronaParameters:
    optical_thickness_mu: float = 0.3
    optical_thickness_sigma: float = 0.2
    variability_mu: float = 0.3
    variability_sigma: float = 0.2
    cloud_penalty: float = 0.3
    weights: ModelWeights = field(default_factory=lambda: DEFAULT_WEIGHTS)


DEFAULT_LUNAR_CORONA_PARAMETERS = LunarCoronaParameters()


def predict_lunar_corona(
    features: Mapping[str, float],
    parameters: LunarCoronaParameters = DEFAULT_LUNAR_CORONA_PARAMETERS,
    *,
    return_components: bool = False,
) -> float | ModelComponents:
    """Predict the probability of a lunar corona."""

    physical = gaussian(
        unit_feature(features, "cloud_optical_thickness"),
        mu=parameters.optical_thickness_mu,
        sigma=parameters.optical_thickness_sigma,
    ) * gaussian(
        unit_feature(features, "cloud_variability", default=0.3),
        mu=parameters.variability_mu,
        sigma=parameters.variability_sigma,
    )
    visibility = (
        unit_feature(features, "source_visible", default=unit_feature(features, "moon_visible"))
        * math.exp(-(parameters.cloud_penalty * unit_feature(features, "cloud_optical_thickness")))
        * unit_feature(features, "brightness_factor", default=1.0)
    )
    return combine_log(
        physical,
        visibility,
        1.0,
        parameters.weights,
        return_components=return_components,
    )
