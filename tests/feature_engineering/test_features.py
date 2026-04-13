from __future__ import annotations

import math

import pytest

from feature_engineering.features import compute_features


def test_compute_features_maps_weather_and_solar_inputs() -> None:
    weather = {
        "temp_250": -35.0,
        "humidity_250": 62.0,
        "cloud_cover_high": 0.75,
        "precipitation": 1.5,
        "surface_visibility": 1.25,
        "fog_presence": 0.7,
    }
    solar = {
        "elevation": 22.0,
        "azimuth": 180.0,
    }

    result = compute_features(weather, solar)

    assert result["cirrus_coverage"] == pytest.approx(0.75)
    assert result["cloud_optical_thickness"] == pytest.approx(0.62)
    assert math.isnan(result["condensate_proxy"])
    assert result["temp_250"] == pytest.approx(-35.0)
    assert result["humidity_250"] == pytest.approx(0.62)
    assert math.isnan(result["ice_cloud_fraction"])
    assert math.isnan(result["ice_300mb"])
    assert math.isnan(result["ice_250mb"])
    assert math.isnan(result["ice_200mb"])
    assert math.isnan(result["wind_shear_250"])
    assert math.isnan(result["vertical_velocity_variance"])
    assert result["thin_cirrus"] == pytest.approx(0.21703816345428795)
    assert result["ice_presence"] == pytest.approx(0.21703816345428795)
    assert result["plate_alignment"] == pytest.approx(0.5)
    assert result["wind_stability"] == pytest.approx(0.5)
    assert result["cloud_variability"] == pytest.approx(0.3)
    assert result["solar_elevation"] == pytest.approx(22.0)
    assert result["solar_azimuth"] == pytest.approx(180.0)
    assert result["sun_visible"] == pytest.approx(1.0)
    assert result["sun_visibility"] == pytest.approx(1.0)
    assert result["precipitation"] == pytest.approx(1.5)
    assert result["surface_visibility"] == pytest.approx(1.25)
    assert result["fog_presence"] == pytest.approx(0.7)


def test_compute_features_handles_missing_values_gracefully() -> None:
    result = compute_features({}, {"elevation": -1.0})

    assert math.isnan(result["cirrus_coverage"])
    assert math.isnan(result["cloud_optical_thickness"])
    assert math.isnan(result["condensate_proxy"])
    assert math.isnan(result["temp_250"])
    assert math.isnan(result["humidity_250"])
    assert math.isnan(result["ice_cloud_fraction"])
    assert math.isnan(result["ice_300mb"])
    assert math.isnan(result["ice_250mb"])
    assert math.isnan(result["ice_200mb"])
    assert math.isnan(result["wind_shear_250"])
    assert math.isnan(result["vertical_velocity_variance"])
    assert math.isnan(result["thin_cirrus"])
    assert math.isnan(result["ice_presence"])
    assert math.isnan(result["precipitation"])
    assert math.isnan(result["surface_visibility"])
    assert math.isnan(result["fog_presence"])
    assert math.isnan(result["solar_azimuth"])
    assert result["plate_alignment"] == pytest.approx(0.5)
    assert result["wind_stability"] == pytest.approx(0.5)
    assert result["cloud_variability"] == pytest.approx(0.3)
    assert result["solar_elevation"] == pytest.approx(-1.0)
    assert result["sun_visible"] == pytest.approx(0.8175744761936437)
    assert result["sun_visibility"] == pytest.approx(0.8175744761936437)


def test_compute_features_uses_explicit_optical_thickness_and_normalizes_percentages() -> None:
    weather = {
        "cloud_cover_high": 75.0,
        "cloud_optical_thickness": 40.0,
        "humidity_250": 80.0,
        "precipitation": -2.0,
        "fog_presence": 70.0,
        "wind_shear_250": 0.2,
        "cloud_cover_grid": [10.0, 50.0, 90.0],
    }

    result = compute_features(weather, {"elevation": 5.0, "azimuth": 120.0})

    assert result["cirrus_coverage"] == pytest.approx(0.75)
    assert result["cloud_optical_thickness"] == pytest.approx(0.4)
    assert result["humidity_250"] == pytest.approx(0.8)
    assert math.isnan(result["ice_cloud_fraction"])
    assert result["wind_shear_250"] == pytest.approx(0.2)
    assert math.isnan(result["vertical_velocity_variance"])
    assert result["thin_cirrus"] == pytest.approx(0.33699672200000006)
    assert result["ice_presence"] == pytest.approx(0.33699672200000006)
    assert result["plate_alignment"] == pytest.approx(0.8187307530779818)
    assert result["wind_stability"] == pytest.approx(0.8187307530779818)
    assert result["cloud_variability"] == pytest.approx(0.32659863237109044)
    assert result["precipitation"] == pytest.approx(0.0)
    assert result["fog_presence"] == pytest.approx(0.7)
    assert result["solar_azimuth"] == pytest.approx(120.0)


def test_compute_features_uses_ice_fraction_and_twilight_visibility() -> None:
    weather = {
        "cloud_cover_high": 0.6,
        "cloud_optical_thickness": 0.2,
        "ice_cloud_fraction": 0.8,
        "wind_shear_250": 0.25,
        "vertical_velocity_variance": 0.3,
    }

    result = compute_features(weather, {"elevation": -3.0})

    assert result["thin_cirrus"] == pytest.approx(0.4021920276213836)
    assert result["ice_presence"] == pytest.approx(0.3217536220971069)
    assert result["sun_visible"] == pytest.approx(0.18242552380635635)
    assert result["sun_visibility"] == pytest.approx(0.18242552380635635)
    assert result["plate_alignment"] == pytest.approx(0.5769498103804866)
    assert result["wind_stability"] == pytest.approx(0.5769498103804866)


def test_compute_features_uses_multi_layer_ice_and_condensate_proxy() -> None:
    weather = {
        "cloud_cover_high": 0.8,
        "condensate_proxy": 0.4,
        "humidity_250": 60.0,
        "ice_300mb": 0.2,
        "ice_250mb": 0.5,
        "ice_200mb": 0.7,
    }

    result = compute_features(weather, {"elevation": 18.0})

    assert result["cloud_optical_thickness"] == pytest.approx(0.48)
    assert result["thin_cirrus"] == pytest.approx(0.3063143087800897)
    assert result["ice_presence"] == pytest.approx(0.47)
