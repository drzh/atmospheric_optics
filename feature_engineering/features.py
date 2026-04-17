"""Feature engineering for atmospheric optics rules."""

from __future__ import annotations

import math
from typing import Mapping

from feature_engineering.cirrus import compute_ice_presence, compute_thin_cirrus
from feature_engineering.dynamics import (
    compute_cloud_variability,
    compute_plate_alignment,
    compute_wind_stability,
)

OPTICAL_THICKNESS_CONDENSATE_WEIGHT = 0.6
OPTICAL_THICKNESS_HUMIDITY_WEIGHT = 0.4
OPTICAL_THICKNESS_CIRRUS_FALLBACK_SCALE = 0.5
TWILIGHT_VISIBILITY_SIGMOID_K = 1.5
TWILIGHT_VISIBILITY_SIGMOID_X0 = -2.0


def compute_features(
    weather: Mapping[str, object],
    source: Mapping[str, object],
    *,
    illumination: str = "solar",
    solar: Mapping[str, object] | None = None,
) -> dict[str, float]:
    """Compute rule-friendly features from weather and source-position inputs."""

    source_elevation = _coerce_float(source.get("elevation"))
    source_azimuth = _coerce_float(source.get("azimuth"))
    cirrus_coverage = _normalize_fraction(
        weather.get("cirrus_coverage", weather.get("cloud_cover_high"))
    )
    cloud_optical_thickness = _normalize_fraction(weather.get("cloud_optical_thickness"))
    condensate_proxy = _normalize_fraction(weather.get("condensate_proxy"))
    humidity_250 = _normalize_fraction(weather.get("humidity_250"))
    if not math.isfinite(cloud_optical_thickness):
        cloud_optical_thickness = _hybrid_optical_thickness(
            condensate_proxy,
            humidity_250,
            cirrus_coverage=cirrus_coverage,
        )

    thin_cirrus = compute_thin_cirrus(cirrus_coverage, cloud_optical_thickness)
    ice_cloud_fraction = _normalize_fraction(weather.get("ice_cloud_fraction"))
    source_visibility = _sigmoid(
        source_elevation,
        k=TWILIGHT_VISIBILITY_SIGMOID_K,
        x0=TWILIGHT_VISIBILITY_SIGMOID_X0,
    )
    vertical_velocity_variance = _normalize_fraction(weather.get("vertical_velocity_variance"))
    wind_shear_250 = _nonnegative_or_nan(weather.get("wind_shear_250"))
    ice_300mb = _normalize_fraction(weather.get("ice_300mb"))
    ice_250mb = _normalize_fraction(weather.get("ice_250mb"))
    ice_200mb = _normalize_fraction(weather.get("ice_200mb"))
    wind_stability = compute_wind_stability(
        wind_shear_250,
        vertical_velocity_variance=vertical_velocity_variance,
    )
    is_lunar = illumination.strip().lower() == "lunar"
    solar_elevation = source_elevation
    solar_azimuth = source_azimuth
    moon_elevation = math.nan
    moon_azimuth = math.nan
    moon_phase = math.nan
    moon_visible = math.nan
    moon_illuminance = math.nan
    sky_darkness = _compute_sky_darkness(solar_elevation)
    brightness_factor = 1.0

    if is_lunar:
        solar_elevation = _coerce_float((solar or {}).get("elevation"))
        solar_azimuth = _coerce_float((solar or {}).get("azimuth"))
        moon_elevation = source_elevation
        moon_azimuth = source_azimuth
        moon_phase = _normalize_fraction(source.get("phase"))
        moon_visible = source_visibility * _sigmoid(moon_phase, k=10.0, x0=0.5)
        moon_illuminance = _normalize_fraction(source.get("illuminance"))
        if not math.isfinite(moon_illuminance):
            moon_illuminance = _clamp_unit_interval(moon_phase**1.5) if math.isfinite(moon_phase) else 0.0
        sky_darkness = _compute_sky_darkness(solar_elevation)
        brightness_factor = (
            _clamp_unit_interval(moon_phase**1.5) * sky_darkness if math.isfinite(moon_phase) else 0.0
        )
        source_visibility = moon_visible

    features = {
        "cirrus_coverage": cirrus_coverage,
        "cloud_optical_thickness": cloud_optical_thickness,
        "condensate_proxy": condensate_proxy,
        "temp_250": _coerce_float(weather.get("temp_250")),
        "humidity_250": humidity_250,
        "ice_cloud_fraction": ice_cloud_fraction,
        "ice_300mb": ice_300mb,
        "ice_250mb": ice_250mb,
        "ice_200mb": ice_200mb,
        "wind_shear_250": wind_shear_250,
        "vertical_velocity_variance": vertical_velocity_variance,
        "source_elevation": source_elevation,
        "source_azimuth": source_azimuth,
        "source_visible": source_visibility,
        "brightness_factor": brightness_factor,
        "is_lunar": 1.0 if is_lunar else 0.0,
        "solar_elevation": solar_elevation,
        "solar_azimuth": solar_azimuth,
        "sun_visible": source_visibility if not is_lunar else _sigmoid(solar_elevation, k=TWILIGHT_VISIBILITY_SIGMOID_K, x0=TWILIGHT_VISIBILITY_SIGMOID_X0),
        "sun_visibility": source_visibility if not is_lunar else _sigmoid(solar_elevation, k=TWILIGHT_VISIBILITY_SIGMOID_K, x0=TWILIGHT_VISIBILITY_SIGMOID_X0),
        "moon_elevation": moon_elevation,
        "moon_azimuth": moon_azimuth,
        "moon_phase": moon_phase,
        "moon_visible": moon_visible,
        "moon_illuminance": moon_illuminance,
        "sky_darkness": sky_darkness,
        "precipitation": _nonnegative_or_nan(weather.get("precipitation")),
        "surface_visibility": _nonnegative_or_nan(weather.get("surface_visibility")),
        "fog_presence": _normalize_fraction(weather.get("fog_presence")),
        "thin_cirrus": thin_cirrus,
        "ice_presence": compute_ice_presence(
            thin_cirrus,
            ice_cloud_fraction,
            ice_300mb=ice_300mb,
            ice_250mb=ice_250mb,
            ice_200mb=ice_200mb,
        ),
        "plate_alignment": compute_plate_alignment(
            wind_shear_250,
            vertical_velocity_variance=vertical_velocity_variance,
        ),
        "wind_stability": wind_stability,
        "cloud_variability": compute_cloud_variability(
            weather.get("cloud_cover_grid", weather.get("cloud_variability"))
        ),
    }
    return features


def _coerce_float(value: object) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _normalize_fraction(value: object) -> float:
    coerced_value = _coerce_float(value)
    if not math.isfinite(coerced_value):
        return math.nan
    if coerced_value > 1.0:
        coerced_value /= 100.0
    return _clamp_unit_interval(coerced_value)


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, value))


def _nonnegative_or_nan(value: object) -> float:
    coerced_value = _coerce_float(value)
    if not math.isfinite(coerced_value):
        return math.nan
    return max(0.0, coerced_value)


def _sigmoid(x: float, *, k: float, x0: float) -> float:
    if not math.isfinite(x) or not math.isfinite(k) or not math.isfinite(x0):
        return 0.0
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


def _compute_sky_darkness(solar_elevation: float) -> float:
    if not math.isfinite(solar_elevation):
        return 0.0
    if solar_elevation <= -18.0:
        return 1.0
    if solar_elevation >= -6.0:
        return 0.0
    return _clamp_unit_interval((-6.0 - solar_elevation) / 12.0)


def _hybrid_optical_thickness(
    condensate_proxy: float,
    humidity_250: float,
    *,
    cirrus_coverage: float,
) -> float:
    if math.isfinite(condensate_proxy) or math.isfinite(humidity_250):
        weighted_sum = 0.0
        total_weight = 0.0
        if math.isfinite(condensate_proxy):
            weighted_sum += OPTICAL_THICKNESS_CONDENSATE_WEIGHT * _clamp_unit_interval(condensate_proxy)
            total_weight += OPTICAL_THICKNESS_CONDENSATE_WEIGHT
        if math.isfinite(humidity_250):
            weighted_sum += OPTICAL_THICKNESS_HUMIDITY_WEIGHT * _clamp_unit_interval(humidity_250)
            total_weight += OPTICAL_THICKNESS_HUMIDITY_WEIGHT
        if total_weight > 0.0:
            return _clamp_unit_interval(weighted_sum / total_weight)

    if math.isfinite(cirrus_coverage):
        return _clamp_unit_interval(cirrus_coverage * OPTICAL_THICKNESS_CIRRUS_FALLBACK_SCALE)
    return math.nan
