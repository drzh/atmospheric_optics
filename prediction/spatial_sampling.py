"""Spatial sampling helpers for atmospheric optics prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


DEFAULT_SPATIAL_RADIUS_KM = 30.0
DEFAULT_SPATIAL_AGGREGATION = "weighted_blend"
SPATIAL_RADIUS_KM: dict[str, float] = {
    "halo": 40.0,
    "parhelia": 40.0,
    "cza": 35.0,
    "circumhorizontal_arc": 45.0,
    "upper_tangent_arc": 35.0,
    "crepuscular_rays": 30.0,
    "rainbow": 25.0,
    "sun_pillar": 15.0,
    "fogbow": 10.0,
    "lunar_halo": 40.0,
    "paraselenae": 40.0,
    "lunar_pillar": 15.0,
    "lunar_corona": 15.0,
    "moonbow": 25.0,
}
DIRECTIONAL_PHENOMENA = {
    "rainbow",
    "crepuscular_rays",
    "sun_pillar",
    "moonbow",
    "lunar_pillar",
}


@dataclass(frozen=True)
class SpatialSample:
    lat: float
    lon: float
    dx_km: float
    dy_km: float
    distance_km: float


def km_to_lat(km: float) -> float:
    """Convert kilometers to an approximate latitude offset."""

    return float(km) / 111.0


def km_to_lon(km: float, lat: float) -> float:
    """Convert kilometers to an approximate longitude offset at the given latitude."""

    cos_lat = math.cos(math.radians(float(lat)))
    if abs(cos_lat) < 1.0e-6:
        return 0.0
    return float(km) / (111.0 * cos_lat)


def adaptive_radius(base_radius: float, cloud_variability: float, wind_shear: float) -> float:
    """Scale the influence radius using variability and shear."""

    normalized_cloud_variability = _nonnegative_scalar(cloud_variability)
    normalized_wind_shear = _nonnegative_scalar(wind_shear)
    factor = 1.0 + (0.5 * normalized_cloud_variability) + (0.3 * normalized_wind_shear)
    return max(5.0, min(float(base_radius) * factor, 80.0))


def generate_grid(
    lat: float,
    lon: float,
    radius_km: float,
    spatial_resolution_km: float | None = None,
) -> list[tuple[float, float]]:
    """Return a 3x3 km-based sampling grid centered on the requested point."""

    return [
        (sample.lat, sample.lon)
        for sample in generate_samples(
            lat,
            lon,
            radius_km,
            spatial_resolution_km=spatial_resolution_km,
        )
    ]


def generate_samples(
    lat: float,
    lon: float,
    radius_km: float,
    spatial_resolution_km: float | None = None,
) -> list[SpatialSample]:
    """Return a 3x3 sampling grid with point geometry metadata."""

    sample_spacing_km = _sample_spacing(radius_km, spatial_resolution_km)
    dlat = km_to_lat(sample_spacing_km)
    dlon = km_to_lon(sample_spacing_km, lat)
    samples: list[SpatialSample] = []
    for lat_index in (-1, 0, 1):
        for lon_index in (-1, 0, 1):
            dx_km = float(lon_index) * sample_spacing_km
            dy_km = float(lat_index) * sample_spacing_km
            samples.append(
                SpatialSample(
                    lat=round(_clamp_latitude(lat + (lat_index * dlat)), 6),
                    lon=round(_normalize_longitude(lon + (lon_index * dlon)), 6),
                    dx_km=dx_km,
                    dy_km=dy_km,
                    distance_km=math.hypot(dx_km, dy_km),
                )
            )
    return samples


def radius_for(phenomenon: str) -> float:
    return SPATIAL_RADIUS_KM.get(phenomenon, DEFAULT_SPATIAL_RADIUS_KM)


def aggregation_for(phenomenon: str) -> str:
    del phenomenon
    return DEFAULT_SPATIAL_AGGREGATION


def spatial_weight(distance_km: float, sigma_km: float) -> float:
    """Return a Gaussian distance-decay weight."""

    if not math.isfinite(distance_km) or not math.isfinite(sigma_km) or sigma_km <= 0.0:
        return 0.0
    return math.exp(-((distance_km**2) / (2.0 * (sigma_km**2))))


def directional_weight(dx_km: float, dy_km: float, solar_azimuth: float) -> float:
    """Bias samples toward the solar azimuth for directional phenomena."""

    if not math.isfinite(dx_km) or not math.isfinite(dy_km):
        return 1.0
    if abs(dx_km) < 1.0e-6 and abs(dy_km) < 1.0e-6:
        return 1.0
    if not math.isfinite(solar_azimuth):
        return 1.0

    angle = math.atan2(dy_km, dx_km)
    solar_angle = math.radians(90.0 - solar_azimuth)
    return _clamp_unit_interval(0.5 + (0.5 * math.cos(angle - solar_angle)))


def apply_spatial_weights(
    phenomenon: str,
    probabilities: Sequence[float],
    samples: Sequence[SpatialSample],
    *,
    radius_km: float,
    solar_azimuth: float = math.nan,
) -> list[float]:
    """Apply distance and optional directional weighting to point probabilities."""

    values = _normalized_probabilities(probabilities)
    if len(values) != len(samples):
        raise ValueError("probabilities and samples must contain the same number of elements")

    sigma_km = max(radius_km / 2.0, 1.0e-6)
    weighted_probabilities: list[float] = []
    for sample, probability in zip(samples, values):
        if not math.isfinite(probability):
            weighted_probabilities.append(math.nan)
            continue
        weight = spatial_weight(sample.distance_km, sigma_km)
        if phenomenon in DIRECTIONAL_PHENOMENA:
            weight *= directional_weight(sample.dx_km, sample.dy_km, solar_azimuth)
        weighted_probabilities.append(_clamp_unit_interval(probability * weight))
    return weighted_probabilities


def aggregate_probabilities(probabilities: Sequence[float], method: str = DEFAULT_SPATIAL_AGGREGATION) -> float:
    """Blend weighted mean and weighted max into one spatially aware signal."""

    del method
    values = _normalized_probabilities(probabilities)
    valid_values = [value for value in values if math.isfinite(value)]
    if not valid_values:
        return 0.0

    mean_probability = sum(valid_values) / len(valid_values)
    max_probability = max(valid_values)
    return _clamp_unit_interval((0.6 * mean_probability) + (0.4 * max_probability))


def spatial_context(
    phenomenon: str,
    probabilities: Sequence[float],
    *,
    radius_km: float | None = None,
    aggregation: str | None = None,
) -> dict[str, float | str]:
    """Summarize the current weighted spatial field for one phenomenon."""

    values = _normalized_probabilities(probabilities)
    valid_values = [value for value in values if math.isfinite(value)]
    resolved_radius_km = radius_km if radius_km is not None else radius_for(phenomenon)
    resolved_aggregation = aggregation if aggregation is not None else aggregation_for(phenomenon)
    if not valid_values:
        return {
            "radius_km": resolved_radius_km,
            "aggregation": resolved_aggregation,
            "center_probability": 0.0,
            "mean_probability": 0.0,
            "max_probability": 0.0,
            "min_probability": 0.0,
            "spatial_variance": 0.0,
            "spatial_consistency": 0.0,
            "spatial_gradient": 0.0,
            "edge_signal": 0.0,
        }

    center_probability = _center_probability(values, fallback=valid_values[len(valid_values) // 2])
    mean_probability = sum(valid_values) / len(valid_values)
    max_probability = max(valid_values)
    min_probability = min(valid_values)
    variance = _variance(valid_values)
    return {
        "radius_km": resolved_radius_km,
        "aggregation": resolved_aggregation,
        "center_probability": center_probability,
        "mean_probability": mean_probability,
        "max_probability": max_probability,
        "min_probability": min_probability,
        "spatial_variance": variance,
        "spatial_consistency": _clamp_unit_interval(1.0 - min(variance, 1.0)),
        "spatial_gradient": _clamp_unit_interval(max_probability - min_probability),
        "edge_signal": _clamp_unit_interval(max_probability - center_probability),
    }


def adjust_confidence(base_confidence: float, probabilities: Sequence[float]) -> float:
    """Reduce confidence when neighboring gridpoints disagree strongly."""

    context = spatial_context("", probabilities)
    spatial_consistency = float(context.get("spatial_consistency", 0.0))
    return _clamp_unit_interval(_clamp_unit_interval(base_confidence) * (0.7 + (0.3 * spatial_consistency)))


def _sample_spacing(radius_km: float, spatial_resolution_km: float | None) -> float:
    resolved_radius = max(0.0, float(radius_km))
    if spatial_resolution_km is None or not math.isfinite(spatial_resolution_km):
        return resolved_radius
    return max(1.0, min(resolved_radius, float(spatial_resolution_km)))


def _nonnegative_scalar(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, float(value))


def _normalized_probabilities(probabilities: Sequence[float]) -> list[float]:
    values: list[float] = []
    for probability in probabilities:
        try:
            numeric_probability = float(probability)
        except (TypeError, ValueError):
            numeric_probability = math.nan
        if math.isfinite(numeric_probability):
            values.append(_clamp_unit_interval(numeric_probability))
        else:
            values.append(math.nan)
    return values


def _center_probability(values: Sequence[float], *, fallback: float) -> float:
    if len(values) > 4 and math.isfinite(values[4]):
        return _clamp_unit_interval(values[4])
    for value in values:
        if math.isfinite(value):
            return _clamp_unit_interval(value)
    return _clamp_unit_interval(fallback)


def _variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    return sum((value - mean_value) ** 2 for value in values) / len(values)


def _clamp_latitude(latitude: float) -> float:
    return max(-90.0, min(90.0, latitude))


def _normalize_longitude(longitude: float) -> float:
    return ((longitude + 180.0) % 360.0) - 180.0


def _clamp_unit_interval(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
