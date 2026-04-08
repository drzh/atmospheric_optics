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
    }
    solar = {
        "elevation": 22.0,
        "azimuth": 180.0,
    }

    result = compute_features(weather, solar)

    assert result == {
        "cirrus_coverage": pytest.approx(0.75),
        "cloud_optical_thickness": pytest.approx(0.375),
        "temp_250": pytest.approx(-35.0),
        "humidity_250": pytest.approx(0.62),
        "solar_elevation": pytest.approx(22.0),
        "sun_visible": pytest.approx(1.0),
        "precipitation": pytest.approx(1.5),
    }


def test_compute_features_handles_missing_values_gracefully() -> None:
    result = compute_features({}, {"elevation": -1.0})

    assert math.isnan(result["cirrus_coverage"])
    assert math.isnan(result["cloud_optical_thickness"])
    assert math.isnan(result["temp_250"])
    assert math.isnan(result["humidity_250"])
    assert math.isnan(result["precipitation"])
    assert result["solar_elevation"] == pytest.approx(-1.0)
    assert result["sun_visible"] == pytest.approx(0.0)


def test_compute_features_uses_explicit_optical_thickness_and_normalizes_percentages() -> None:
    weather = {
        "cloud_cover_high": 75.0,
        "cloud_optical_thickness": 40.0,
        "humidity_250": 80.0,
        "precipitation": -2.0,
    }

    result = compute_features(weather, {"elevation": 5.0})

    assert result["cirrus_coverage"] == pytest.approx(0.75)
    assert result["cloud_optical_thickness"] == pytest.approx(0.4)
    assert result["humidity_250"] == pytest.approx(0.8)
    assert result["precipitation"] == pytest.approx(0.0)
