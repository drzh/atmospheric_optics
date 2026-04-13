from __future__ import annotations

import pytest

from core.spatial import (
    SpatialSample,
    adaptive_radius,
    adjust_confidence,
    aggregate_probabilities,
    apply_spatial_weights,
    generate_grid,
    spatial_context,
)


def test_generate_grid_returns_centered_3x3_layout() -> None:
    grid = generate_grid(32.8, -96.8, 20.0)

    assert len(grid) == 9
    assert grid[4] == pytest.approx((32.8, -96.8))
    assert grid[0][0] < 32.8
    assert grid[8][0] > 32.8
    assert grid[0][1] < -96.8
    assert grid[8][1] > -96.8


def test_adaptive_radius_expands_with_variability_and_is_bounded() -> None:
    assert adaptive_radius(20.0, cloud_variability=0.0, wind_shear=0.0) == pytest.approx(20.0)
    assert adaptive_radius(20.0, cloud_variability=1.0, wind_shear=1.0) == pytest.approx(36.0)
    assert adaptive_radius(1.0, cloud_variability=0.0, wind_shear=0.0) == pytest.approx(5.0)
    assert adaptive_radius(100.0, cloud_variability=1.0, wind_shear=1.0) == pytest.approx(80.0)


def test_spatial_context_and_confidence_reflect_consistency() -> None:
    probabilities = [0.2, 0.22, 0.21, 0.19, 0.2, 0.23, 0.18, 0.21, 0.2]

    context = spatial_context("halo", probabilities)
    confidence = adjust_confidence(0.9, probabilities)

    assert context["aggregation"] == "weighted_blend"
    assert context["radius_km"] == 40.0
    assert context["center_probability"] == pytest.approx(0.2)
    assert context["mean_probability"] == pytest.approx(sum(probabilities) / len(probabilities))
    assert context["max_probability"] == pytest.approx(0.23)
    assert context["min_probability"] == pytest.approx(0.18)
    assert context["spatial_gradient"] == pytest.approx(0.05)
    assert context["edge_signal"] == pytest.approx(0.03)
    assert 0.0 <= context["spatial_variance"] <= 1.0
    assert 0.0 <= context["spatial_consistency"] <= 1.0
    assert 0.0 <= confidence <= 0.9


def test_weighted_spatial_sampling_biases_toward_center_and_solar_direction() -> None:
    samples = [
        SpatialSample(lat=0.0, lon=0.0, dx_km=-10.0, dy_km=0.0, distance_km=10.0),
        SpatialSample(lat=0.0, lon=0.0, dx_km=0.0, dy_km=0.0, distance_km=0.0),
        SpatialSample(lat=0.0, lon=0.0, dx_km=10.0, dy_km=0.0, distance_km=10.0),
    ]
    probabilities = [0.9, 0.4, 0.9]

    halo_weighted = apply_spatial_weights("halo", probabilities, samples, radius_km=20.0)
    directional_weighted = apply_spatial_weights(
        "rainbow",
        probabilities,
        samples,
        radius_km=20.0,
        solar_azimuth=90.0,
    )

    assert halo_weighted[1] == pytest.approx(0.4)
    assert halo_weighted[0] < probabilities[0]
    assert halo_weighted[2] < probabilities[2]
    assert directional_weighted[2] > directional_weighted[0]


def test_weighted_blend_aggregation_retains_local_peaks_without_becoming_max_only() -> None:
    weighted_probabilities = [0.05, 0.12, 0.06, 0.48, 0.54, 0.5, 0.04, 0.11, 0.05]

    aggregated = aggregate_probabilities(weighted_probabilities, "weighted_blend")

    assert aggregated > (sum(weighted_probabilities) / len(weighted_probabilities))
    assert aggregated < max(weighted_probabilities)
