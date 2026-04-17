"""Halo and sundog probability rules."""

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
class HaloParameters:
    geometry_mu: float = 20.0
    geometry_sigma: float = 20.0
    weights: ModelWeights = field(default_factory=lambda: DEFAULT_WEIGHTS)


@dataclass(frozen=True)
class ParheliaParameters:
    geometry_mu: float = 20.0
    geometry_sigma: float = 15.0
    plate_alignment_default: float = 0.5
    weights: ModelWeights = field(default_factory=lambda: DEFAULT_WEIGHTS)


DEFAULT_HALO_PARAMETERS = HaloParameters()
DEFAULT_PARHELIA_PARAMETERS = ParheliaParameters()


def _source_elevation(features: Mapping[str, float]) -> float:
    elevation = numeric_feature(features, "source_elevation")
    if math.isfinite(elevation):
        return elevation
    return numeric_feature(features, "solar_elevation")


def _ice_crystal_terms(features: Mapping[str, float]) -> tuple[float, float]:
    physical = unit_feature(features, "ice_presence") * unit_feature(features, "thin_cirrus")
    visibility = (
        unit_feature(features, "source_visible", default=unit_feature(features, "sun_visible"))
        * math.exp(
        -unit_feature(features, "cloud_optical_thickness")
        )
        * unit_feature(features, "brightness_factor", default=1.0)
    )
    return physical, visibility


def predict_halo(
    features: Mapping[str, float],
    parameters: HaloParameters = DEFAULT_HALO_PARAMETERS,
    *,
    return_components: bool = False,
) -> float | ModelComponents:
    """Predict the probability of halo conditions."""

    physical, visibility = _ice_crystal_terms(features)
    geometry = gaussian(
        _source_elevation(features),
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


def predict_parhelia(
    features: Mapping[str, float],
    parameters: ParheliaParameters = DEFAULT_PARHELIA_PARAMETERS,
    *,
    return_components: bool = False,
) -> float | ModelComponents:
    """Predict the probability of sundog conditions."""

    physical, visibility = _ice_crystal_terms(features)
    alignment = (
        0.7 * unit_feature(features, "plate_alignment", default=parameters.plate_alignment_default)
        + 0.3 * unit_feature(features, "wind_stability", default=parameters.plate_alignment_default)
    )
    physical *= alignment
    geometry = gaussian(
        _source_elevation(features),
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
