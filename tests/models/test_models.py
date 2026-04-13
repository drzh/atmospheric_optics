from __future__ import annotations

import pytest

from models.cza_model import predict_cza
from models.halo_model import predict_halo, predict_parhelia
from models.ice_crystal_model import (
    predict_circumhorizontal_arc,
    predict_sun_pillar,
    predict_upper_tangent_arc,
)
from models.rainbow_model import predict_rainbow
from models.scattering_model import predict_crepuscular_rays, predict_fogbow


def test_predict_halo_matches_quantitative_ice_visibility_geometry_model() -> None:
    features = {
        "ice_presence": 0.55,
        "thin_cirrus": 0.6,
        "cloud_optical_thickness": 0.3,
        "solar_elevation": 20.0,
        "sun_visible": 1.0,
        "plate_alignment": 0.8,
    }

    probability = predict_halo(features)
    parhelia_probability = predict_parhelia(features)

    assert probability == pytest.approx(0.16, abs=0.03)
    assert parhelia_probability < probability
    assert predict_parhelia({**features, "plate_alignment": 0.3}) < parhelia_probability
    assert predict_halo({**features, "cloud_optical_thickness": 0.9}) < probability
    assert predict_halo({**features, "thin_cirrus": 0.15}) < probability


def test_predict_cza_prefers_optimal_solar_elevation_with_strict_geometry() -> None:
    optimal_features = {
        "ice_presence": 0.55,
        "thin_cirrus": 0.6,
        "cloud_optical_thickness": 0.2,
        "solar_elevation": 22.0,
        "sun_visible": 1.0,
    }
    off_angle_features = {**optimal_features, "solar_elevation": 40.0}

    assert predict_cza(optimal_features) > 0.15
    assert predict_cza(off_angle_features) < 0.01


def test_predict_rainbow_uses_smooth_precipitation_and_geometry_response() -> None:
    optimal_features = {
        "sun_visible": 1.0,
        "precipitation": 1.0,
        "solar_elevation": 20.0,
    }

    assert predict_rainbow(optimal_features) > 0.45
    assert predict_rainbow({**optimal_features, "sun_visible": 0.0}) < 0.001
    assert predict_rainbow({**optimal_features, "solar_elevation": 60.0}) < 0.05
    assert predict_rainbow({**optimal_features, "precipitation": 0.0}) < predict_rainbow(optimal_features)


def test_predict_circumhorizontal_arc_requires_high_solar_elevation() -> None:
    optimal_features = {
        "ice_presence": 0.55,
        "thin_cirrus": 0.6,
        "cloud_optical_thickness": 0.2,
        "solar_elevation": 66.0,
        "sun_visible": 1.0,
    }

    assert predict_circumhorizontal_arc(optimal_features) > 0.15
    assert predict_circumhorizontal_arc({**optimal_features, "solar_elevation": 35.0}) < 0.01


def test_predict_upper_tangent_arc_prefers_low_solar_elevation() -> None:
    optimal_features = {
        "ice_presence": 0.55,
        "thin_cirrus": 0.6,
        "cloud_optical_thickness": 0.2,
        "solar_elevation": 12.0,
        "sun_visible": 1.0,
    }

    assert predict_upper_tangent_arc(optimal_features) > 0.15
    assert predict_upper_tangent_arc({**optimal_features, "solar_elevation": 42.0}) < 0.01


def test_predict_sun_pillar_prefers_low_sun_with_ice_support() -> None:
    optimal_features = {
        "ice_presence": 0.55,
        "thin_cirrus": 0.45,
        "cloud_optical_thickness": 0.2,
        "humidity_250": 0.9,
        "solar_elevation": 2.0,
        "sun_visible": 1.0,
    }

    assert predict_sun_pillar(optimal_features) > 0.2
    assert predict_sun_pillar({**optimal_features, "solar_elevation": 22.0}) < 0.01
    assert predict_sun_pillar({**optimal_features, "humidity_250": 0.05, "thin_cirrus": 0.05}) < 0.1


def test_predict_crepuscular_rays_prefers_broken_cloud_and_low_sun() -> None:
    optimal_features = {
        "cirrus_coverage": 0.5,
        "cloud_variability": 0.5,
        "cloud_optical_thickness": 0.2,
        "solar_elevation": 6.0,
        "sun_visible": 1.0,
    }

    assert predict_crepuscular_rays(optimal_features) > 0.45
    assert predict_crepuscular_rays({**optimal_features, "cirrus_coverage": 0.05, "cloud_variability": 0.1}) < 0.1
    assert predict_crepuscular_rays({**optimal_features, "solar_elevation": 30.0}) < 0.05


def test_predict_fogbow_prefers_foggy_low_visibility_with_limited_rain() -> None:
    optimal_features = {
        "fog_presence": 1.0,
        "surface_visibility": 0.8,
        "precipitation": 0.0,
        "cloud_optical_thickness": 0.15,
        "solar_elevation": 15.0,
        "sun_visible": 1.0,
    }

    assert predict_fogbow(optimal_features) > 0.45
    assert predict_fogbow({**optimal_features, "precipitation": 1.0}) < 0.05
    assert predict_fogbow({**optimal_features, "solar_elevation": 45.0}) < 0.05
