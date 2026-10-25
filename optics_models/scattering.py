"""Scattering-driven optical phenomenon probability rules."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from optics_models.combination import ModelComponents, combine_log
from optics_models.rainbow import PRIMARY_BOW_MAX_ELEVATION
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
class CrepuscularRaysParameters:
    cloud_coverage_mu: float = 0.5
    cloud_coverage_sigma: float = 0.3
    cloud_variability_mu: float = 0.5
    cloud_variability_sigma: float = 0.3
    geometry_mu: float = 6.0
    geometry_sigma: float = 6.0
    cloud_visibility_penalty: float = 0.5
    weights: ModelWeights = field(default_factory=lambda: DEFAULT_WEIGHTS)


@dataclass(frozen=True)
class FogbowParameters:
    visibility_mu: float = 0.8
    visibility_sigma: float = 0.8
    rain_suppression_k: float = 8.0
    rain_suppression_x0: float = 0.6
    geometry_mu: float = 15.0
    geometry_sigma: float = 12.0
    cloud_penalty: float = 0.25
    weights: ModelWeights = field(default_factory=lambda: DEFAULT_WEIGHTS)


DEFAULT_CREPUSCULAR_RAYS_PARAMETERS = CrepuscularRaysParameters()
DEFAULT_FOGBOW_PARAMETERS = FogbowParameters()


def predict_anticrepuscular_rays(
    features: Mapping[str, float],
    parameters: CrepuscularRaysParameters = DEFAULT_CREPUSCULAR_RAYS_PARAMETERS,
    *,
    return_components: bool = False,
) -> float | ModelComponents:
    """Combine source-side cloud gaps with a clearer opposite viewing direction."""

    source_cloud = unit_feature(features, "source_cloud_fraction")
    opposite_cloud = numeric_feature(features, "antisolar_cloud_fraction")
    # A mixture of cloud and gaps supplies the normalized shadow-pattern support.
    physical = 4.0 * source_cloud * (1.0 - source_cloud)
    clearance = 1.0 - opposite_cloud if math.isfinite(opposite_cloud) and 0.0 <= opposite_cloud <= 1.0 else 0.0
    visibility = (
        clearance * unit_feature(features, "sun_visible")
        * math.exp(-parameters.cloud_visibility_penalty * unit_feature(features, "cloud_optical_thickness"))
    )
    elevation = numeric_feature(features, "solar_elevation")
    geometry = gaussian(elevation, mu=parameters.geometry_mu, sigma=parameters.geometry_sigma) if -6.0 <= elevation <= 90.0 else 0.0
    return combine_log(physical, visibility, geometry, parameters.weights, return_components=return_components)


def predict_crepuscular_rays(
    features: Mapping[str, float],
    parameters: CrepuscularRaysParameters = DEFAULT_CREPUSCULAR_RAYS_PARAMETERS,
    *,
    return_components: bool = False,
) -> float | ModelComponents:
    """Predict the probability of crepuscular rays."""

    cirrus_coverage = unit_feature(features, "cirrus_coverage")
    cloud_variability = unit_feature(features, "cloud_variability", default=0.3)
    physical = gaussian(
        cirrus_coverage,
        mu=parameters.cloud_coverage_mu,
        sigma=parameters.cloud_coverage_sigma,
    ) * gaussian(
        cloud_variability,
        mu=parameters.cloud_variability_mu,
        sigma=parameters.cloud_variability_sigma,
    )
    visibility = unit_feature(features, "sun_visible") * math.exp(
        -(parameters.cloud_visibility_penalty * unit_feature(features, "cloud_optical_thickness"))
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


def predict_fogbow(
    features: Mapping[str, float],
    parameters: FogbowParameters = DEFAULT_FOGBOW_PARAMETERS,
    *,
    return_components: bool = False,
) -> float | ModelComponents:
    """Predict the probability of a fogbow."""

    fog_presence = unit_feature(features, "fog_presence")
    surface_visibility = numeric_feature(features, "surface_visibility")
    visibility_support = (
        gaussian(surface_visibility, mu=parameters.visibility_mu, sigma=parameters.visibility_sigma)
        if math.isfinite(surface_visibility) and surface_visibility >= 0.0
        else 1.0
    )
    dry_air_support = 1.0 - sigmoid(
        nonnegative_feature(features, "precipitation"),
        k=parameters.rain_suppression_k,
        x0=parameters.rain_suppression_x0,
    )
    # Low visibility also occurs in dust, smoke, and haze; require droplet evidence.
    physical = fog_presence * visibility_support * max(0.0, dry_air_support)
    visibility = unit_feature(features, "sun_visible") * math.exp(
        -(parameters.cloud_penalty * unit_feature(features, "cloud_optical_thickness"))
    )
    solar_elevation = numeric_feature(features, "solar_elevation")
    geometry = gaussian(
        solar_elevation,
        mu=parameters.geometry_mu,
        sigma=parameters.geometry_sigma,
    ) if 0.0 <= solar_elevation < PRIMARY_BOW_MAX_ELEVATION else 0.0
    return combine_log(
        physical,
        visibility,
        geometry,
        parameters.weights,
        return_components=return_components,
    )
