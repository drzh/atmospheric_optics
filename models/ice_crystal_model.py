"""Additional ice-crystal optical phenomenon probability rules."""

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
    sigmoid,
    unit_feature,
)


def _source_elevation(features: Mapping[str, float]) -> float:
    source_elevation = numeric_feature(features, "source_elevation")
    if math.isfinite(source_elevation):
        return source_elevation
    return numeric_feature(features, "solar_elevation")


def _ice_crystal_support(features: Mapping[str, float]) -> float:
    return unit_feature(features, "ice_presence") * unit_feature(features, "thin_cirrus")


def _source_visibility(features: Mapping[str, float], *, cloud_penalty: float = 1.0) -> float:
    return (
        unit_feature(features, "source_visible", default=unit_feature(features, "sun_visible"))
        * math.exp(
        -(cloud_penalty * unit_feature(features, "cloud_optical_thickness"))
        )
        * unit_feature(features, "brightness_factor", default=1.0)
    )


@dataclass(frozen=True)
class CircumhorizontalArcParameters:
    geometry_threshold_k: float = 0.6
    geometry_threshold_x0: float = 58.0
    geometry_mu: float = 66.0
    geometry_sigma: float = 8.0
    weights: ModelWeights = field(default_factory=lambda: DEFAULT_WEIGHTS)


@dataclass(frozen=True)
class UpperTangentArcParameters:
    geometry_mu: float = 12.0
    geometry_sigma: float = 7.0
    geometry_max_elevation_k: float = 0.35
    geometry_max_elevation_x0: float = 32.0
    weights: ModelWeights = field(default_factory=lambda: DEFAULT_WEIGHTS)


@dataclass(frozen=True)
class SunPillarParameters:
    cloud_penalty: float = 0.5
    geometry_mu: float = 2.0
    geometry_sigma: float = 4.0
    geometry_max_elevation_k: float = 0.4
    geometry_max_elevation_x0: float = 14.0
    weights: ModelWeights = field(default_factory=lambda: DEFAULT_WEIGHTS)


DEFAULT_CIRCUMHORIZONTAL_ARC_PARAMETERS = CircumhorizontalArcParameters()
DEFAULT_UPPER_TANGENT_ARC_PARAMETERS = UpperTangentArcParameters()
DEFAULT_SUN_PILLAR_PARAMETERS = SunPillarParameters()


def predict_circumhorizontal_arc(
    features: Mapping[str, float],
    parameters: CircumhorizontalArcParameters = DEFAULT_CIRCUMHORIZONTAL_ARC_PARAMETERS,
    *,
    return_components: bool = False,
) -> float | ModelComponents:
    """Predict the probability of a circumhorizontal arc."""

    physical = _ice_crystal_support(features)
    visibility = _source_visibility(features)
    source_elevation = _source_elevation(features)
    geometry = sigmoid(
        source_elevation,
        k=parameters.geometry_threshold_k,
        x0=parameters.geometry_threshold_x0,
    ) * gaussian(
        source_elevation,
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


def predict_upper_tangent_arc(
    features: Mapping[str, float],
    parameters: UpperTangentArcParameters = DEFAULT_UPPER_TANGENT_ARC_PARAMETERS,
    *,
    return_components: bool = False,
) -> float | ModelComponents:
    """Predict the probability of an upper tangent arc."""

    physical = _ice_crystal_support(features)
    visibility = _source_visibility(features)
    source_elevation = _source_elevation(features)
    geometry = gaussian(
        source_elevation,
        mu=parameters.geometry_mu,
        sigma=parameters.geometry_sigma,
    ) * (1.0 - sigmoid(
        source_elevation,
        k=parameters.geometry_max_elevation_k,
        x0=parameters.geometry_max_elevation_x0,
    ))
    return combine_log(
        physical,
        visibility,
        geometry,
        parameters.weights,
        return_components=return_components,
    )


def predict_sun_pillar(
    features: Mapping[str, float],
    parameters: SunPillarParameters = DEFAULT_SUN_PILLAR_PARAMETERS,
    *,
    return_components: bool = False,
) -> float | ModelComponents:
    """Predict the probability of a sun pillar."""

    physical = unit_feature(features, "ice_presence") * max(
        unit_feature(features, "thin_cirrus"),
        unit_feature(features, "humidity_250"),
    )
    physical *= 0.7 + (0.3 * unit_feature(features, "wind_stability", default=0.5))
    visibility = _source_visibility(features, cloud_penalty=parameters.cloud_penalty)
    source_elevation = _source_elevation(features)
    geometry = gaussian(
        source_elevation,
        mu=parameters.geometry_mu,
        sigma=parameters.geometry_sigma,
    ) * (1.0 - sigmoid(
        source_elevation,
        k=parameters.geometry_max_elevation_k,
        x0=parameters.geometry_max_elevation_x0,
    ))
    return combine_log(
        physical,
        visibility,
        geometry,
        parameters.weights,
        return_components=return_components,
    )
