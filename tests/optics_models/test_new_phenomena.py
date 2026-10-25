from __future__ import annotations

import math

import pytest

from optics_models.corona import predict_solar_corona
from optics_models.lunar import predict_lunar_corona
from optics_models.rainbow import predict_rainbow, predict_secondary_rainbow
from optics_models.scattering import predict_anticrepuscular_rays

DAY = {"solar_elevation": 20.0, "source_elevation": 20.0, "source_visible": 1.0, "sun_visible": 1.0,
       "cloud_optical_thickness": 0.3, "cloud_variability": 0.3, "precipitation": 2.0}


def test_solar_corona_reuses_diffraction_support_without_moon_requirements() -> None:
    assert predict_solar_corona(DAY) > 0.8
    assert predict_solar_corona({**DAY, "cloud_optical_thickness": 0.0}) == 0.0
    assert predict_solar_corona({**DAY, "cloud_optical_thickness": math.nan}) == 0.0
    assert predict_solar_corona({**DAY, "source_elevation": -1.0}) == 0.0
    assert predict_solar_corona({**DAY, "source_visible": 0.0}) == 0.0
    assert predict_solar_corona({**DAY, "cloud_optical_thickness": 0.9}) < 0.1


def test_lunar_compatibility_does_not_treat_sun_visibility_as_moonlight() -> None:
    features = {key: value for key, value in DAY.items() if key != "source_visible"}
    assert predict_lunar_corona(features) == 0.0
    assert predict_lunar_corona({**features, "moon_visible": 1.0}) == pytest.approx(predict_solar_corona(DAY))
    assert predict_lunar_corona({**features, "moon_visible": math.nan}) == 0.0
    assert predict_solar_corona({**features, "sun_visible": 0.0, "moon_visible": 1.0}) == 0.0


@pytest.mark.parametrize("elevation", [42.0, 45.0, 51.0, 52.0, 53.9])
def test_secondary_geometry_is_independent_of_primary_cutoff(elevation) -> None:
    features = {**DAY, "source_elevation": elevation}
    assert predict_rainbow(features) == 0.0
    assert predict_secondary_rainbow(features) > 0.0


@pytest.mark.parametrize("overrides", [{"source_elevation": 54.0}, {"source_elevation": -1.0}, {"precipitation": 0.0}, {"source_visible": 0.0}])
def test_secondary_bow_preserves_required_ingredients(overrides) -> None:
    assert predict_secondary_rainbow({**DAY, **overrides}) == 0.0


def test_secondary_visibility_is_more_sensitive_without_an_unreachable_score_ceiling() -> None:
    ideal = {**DAY, "cloud_optical_thickness": 0.01}
    dim = {**ideal, "source_visible": 0.4}
    primary_loss = predict_rainbow(dim) / predict_rainbow(ideal)
    secondary_loss = predict_secondary_rainbow(dim) / predict_secondary_rainbow(ideal)
    assert predict_secondary_rainbow(ideal) > 0.8
    assert secondary_loss < primary_loss
    assert predict_secondary_rainbow({**ideal, "source_elevation": 53.0}) < predict_secondary_rainbow({**ideal, "source_elevation": 51.0})


def test_anticrepuscular_rays_require_source_obstruction_and_opposite_clearance() -> None:
    features = {**DAY, "solar_elevation": 6.0, "cloud_optical_thickness": 0.05, "source_cloud_fraction": 0.5, "antisolar_cloud_fraction": 0.0}
    assert predict_anticrepuscular_rays(features) > 0.8
    for overrides in [
        {"source_cloud_fraction": 0.0}, {"source_cloud_fraction": 1.0},
        {"antisolar_cloud_fraction": 1.0}, {"source_cloud_fraction": math.nan},
        {"antisolar_cloud_fraction": math.nan}, {"solar_elevation": -20.0},
    ]:
        assert predict_anticrepuscular_rays({**features, **overrides}) == 0.0
