from __future__ import annotations

import math

import pytest

from optics_models.circumzenithal_arc import predict_cza
from optics_models.ice_crystals import predict_circumhorizontal_arc
from optics_models.lunar import predict_lunar_corona
from optics_models.rainbow import predict_rainbow
from optics_models.scattering import predict_fogbow

FAVORABLE = {
    "ice_presence": 1.0, "thin_cirrus": 1.0, "sun_visible": 1.0,
    "source_visible": 1.0, "brightness_factor": 1.0, "cloud_optical_thickness": 0.05,
    "solar_elevation": 20.0, "source_elevation": 20.0, "precipitation": 2.0,
    "fog_presence": 1.0, "surface_visibility": 0.8,
}


@pytest.mark.parametrize("rain", [0.0, -1.0, math.nan, None])
def test_dry_or_missing_rain_cannot_support_a_bow(rain) -> None:
    components = predict_rainbow({**FAVORABLE, "precipitation": rain}, return_components=True)
    assert components.physical == 0.0
    assert components.probability == 0.0


def test_rain_support_is_monotonic_and_cloud_attenuation_matters() -> None:
    scores = [predict_rainbow({**FAVORABLE, "precipitation": value}) for value in (0.0, 0.01, 0.1, 1.0, 2.0)]
    assert scores == sorted(scores)
    assert scores[-1] >= 0.8
    assert predict_rainbow({**FAVORABLE, "cloud_optical_thickness": 1.0}) < 0.8


@pytest.mark.parametrize(
    "model,elevation",
    [(predict_cza, -1.0), (predict_cza, 32.0), (predict_cza, 32.1),
     (predict_circumhorizontal_arc, 57.9), (predict_circumhorizontal_arc, 58.0),
     (predict_rainbow, -1.0), (predict_rainbow, 42.0),
     (predict_fogbow, -1.0), (predict_fogbow, 42.0)],
)
def test_geometry_excludes_impossible_ground_level_elevations(model, elevation) -> None:
    components = model({**FAVORABLE, "solar_elevation": elevation, "source_elevation": elevation}, return_components=True)
    assert components.geometry == 0.0
    assert components.probability == 0.0


def test_arc_models_remain_possible_inside_their_elevation_ranges() -> None:
    assert predict_cza({**FAVORABLE, "solar_elevation": 22.0}) > 0.8
    assert predict_cza({**FAVORABLE, "solar_elevation": 31.9}) > 0.0
    assert predict_circumhorizontal_arc({**FAVORABLE, "source_elevation": 66.0}) > 0.8
    assert predict_circumhorizontal_arc({**FAVORABLE, "source_elevation": 58.1}) > 0.0


@pytest.mark.parametrize("visibility", [0.8, None, math.nan, -1.0])
def test_low_or_unknown_visibility_without_fog_is_not_droplet_evidence(visibility) -> None:
    assert predict_fogbow({**FAVORABLE, "precipitation": 0.0, "fog_presence": 0.0, "surface_visibility": visibility}) == 0.0


def test_reported_fog_can_support_a_bow_when_visibility_is_missing() -> None:
    assert predict_fogbow({**FAVORABLE, "solar_elevation": 15.0, "precipitation": 0.0, "surface_visibility": math.nan}) > 0.8


@pytest.mark.parametrize("optical_depth", [0.0, -1.0, None, math.nan])
def test_clear_or_unknown_cloud_cannot_support_a_lunar_corona(optical_depth) -> None:
    assert predict_lunar_corona({**FAVORABLE, "cloud_optical_thickness": optical_depth}) == 0.0
    assert predict_lunar_corona({**FAVORABLE, "cloud_optical_thickness": 0.3}) > 0.8
