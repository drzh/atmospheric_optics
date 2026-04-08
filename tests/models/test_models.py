from __future__ import annotations

import pytest

from models.cza_model import predict_cza
from models.halo_model import predict_halo, predict_parhelia
from models.rainbow_model import predict_rainbow


def test_predict_halo_matches_quantitative_ice_visibility_geometry_model() -> None:
    features = {
        "cirrus_coverage": 0.7,
        "cloud_optical_thickness": 0.3,
        "temp_250": -35.0,
        "solar_elevation": 18.0,
        "sun_visible": 0.8,
    }

    probability = predict_halo(features)

    assert probability == pytest.approx(0.34, abs=0.03)
    assert predict_parhelia(features) == pytest.approx(probability)
    assert predict_halo({**features, "cloud_optical_thickness": 0.9}) < probability
    assert predict_halo({**features, "temp_250": -5.0}) < probability


def test_predict_cza_prefers_optimal_solar_elevation_with_strict_geometry() -> None:
    optimal_features = {
        "cirrus_coverage": 0.7,
        "cloud_optical_thickness": 0.2,
        "temp_250": -35.0,
        "solar_elevation": 22.0,
        "sun_visible": 1.0,
    }
    off_angle_features = {**optimal_features, "solar_elevation": 40.0}

    assert predict_cza(optimal_features) > 0.45
    assert predict_cza(off_angle_features) < 0.01


def test_predict_rainbow_uses_smooth_precipitation_and_geometry_response() -> None:
    optimal_features = {
        "sun_visible": 1.0,
        "precipitation": 1.0,
        "solar_elevation": 20.0,
    }

    assert predict_rainbow(optimal_features) > 0.95
    assert predict_rainbow({**optimal_features, "sun_visible": 0.0}) == pytest.approx(0.0)
    assert predict_rainbow({**optimal_features, "solar_elevation": 60.0}) < 0.05
    assert predict_rainbow({**optimal_features, "precipitation": 0.0}) < predict_rainbow(optimal_features)
