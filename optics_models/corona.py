"""Shared solar and lunar thin-cloud diffraction suitability model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from optics_models.combination import ModelComponents, combine_log
from optics_models.probability import DEFAULT_WEIGHTS, ModelWeights, gaussian, numeric_feature, unit_feature


@dataclass(frozen=True)
class CoronaParameters:
    optical_thickness_mu: float = 0.3
    optical_thickness_sigma: float = 0.2
    variability_mu: float = 0.3
    variability_sigma: float = 0.2
    cloud_penalty: float = 0.3
    weights: ModelWeights = field(default_factory=lambda: DEFAULT_WEIGHTS)


DEFAULT_CORONA_PARAMETERS = CoronaParameters()


def predict_corona(
    features: Mapping[str, float],
    parameters: CoronaParameters = DEFAULT_CORONA_PARAMETERS,
    *,
    geometry: float = 1.0,
    return_components: bool = False,
) -> float | ModelComponents:
    """Estimate diffraction support; particle-size uniformity remains a proxy."""

    optical_thickness = numeric_feature(features, "cloud_optical_thickness")
    physical = gaussian(
        unit_feature(features, "cloud_optical_thickness"),
        mu=parameters.optical_thickness_mu,
        sigma=parameters.optical_thickness_sigma,
    ) * gaussian(
        unit_feature(features, "cloud_variability", default=0.3),
        mu=parameters.variability_mu,
        sigma=parameters.variability_sigma,
    )
    if not math.isfinite(optical_thickness) or optical_thickness <= 0.0:
        physical = 0.0
    visibility = (
        unit_feature(features, "source_visible", default=unit_feature(features, "moon_visible", default=unit_feature(features, "sun_visible")))
        * math.exp(-(parameters.cloud_penalty * unit_feature(features, "cloud_optical_thickness")))
        * unit_feature(features, "brightness_factor", default=1.0)
    )
    return combine_log(physical, visibility, geometry, parameters.weights, return_components=return_components)


def predict_solar_corona(
    features: Mapping[str, float],
    parameters: CoronaParameters = DEFAULT_CORONA_PARAMETERS,
    *,
    return_components: bool = False,
) -> float | ModelComponents:
    elevation = numeric_feature(features, "source_elevation")
    if not math.isfinite(elevation):
        elevation = numeric_feature(features, "solar_elevation")
    solar_features = dict(features)
    solar_features["source_visible"] = unit_feature(features, "source_visible", default=unit_feature(features, "sun_visible"))
    return predict_corona(
        solar_features, parameters, geometry=1.0 if 0.0 <= elevation <= 90.0 else 0.0,
        return_components=return_components,
    )
