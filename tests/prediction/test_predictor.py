from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from prediction.pipeline import PHENOMENA, predict_all
from weather_sources.snapshots import SourceAttribution, WeatherSnapshot
from optics_models.combination import ModelComponents


def _phenomena_by_id(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    phenomena = payload.get("phenomena")
    if not isinstance(phenomena, list):
        return {}
    result: dict[str, dict[str, object]] = {}
    for entry in phenomena:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            result[entry["id"]] = entry
    return result


@pytest.mark.parametrize(
    "phenomenon,weather,elevation,should_alert",
    [
        ("halo", {"cloud_cover_high": 1.0, "cloud_optical_thickness": 0.02, "ice_cloud_fraction": 1.0}, 20.0, True),
        ("rainbow", {"cloud_cover_high": 0.2, "cloud_optical_thickness": 0.05, "precipitation": 2.0}, 20.0, True),
        ("halo", {"cloud_cover_high": 1.0, "cloud_optical_thickness": 1.0, "ice_cloud_fraction": 1.0}, 20.0, False),
        ("rainbow", {"cloud_cover_high": 0.0, "precipitation": 0.0}, 20.0, False),
        ("halo", {"cloud_cover_high": 1.0, "cloud_optical_thickness": 0.02, "ice_cloud_fraction": 1.0}, -25.0, False),
        ("rainbow", {"cloud_cover_high": 0.2, "precipitation": 2.0}, -25.0, False),
    ],
)
def test_spatial_pipeline_can_trigger_alerts_only_in_favorable_conditions(phenomenon, weather, elevation, should_alert) -> None:
    snapshot = WeatherSnapshot(weather=weather, sources=(SourceAttribution(name="goes-east", timestamp="20261024 180000z"),))
    with patch("prediction.pipeline.get_weather_snapshot", return_value=snapshot), patch(
        "prediction.pipeline.get_solar_position", return_value={"elevation": elevation, "azimuth": 180.0}
    ):
        common = dict(
            lat=32.847, lon=-96.806, mode="observed", illumination="solar", phenomena=(phenomenon,),
            at_time=datetime(2026, 10, 24, 18, tzinfo=timezone.utc),
        )
        spatial = _phenomena_by_id(predict_all(**common))[phenomenon]
        center = _phenomena_by_id(predict_all(**common, lightweight=True))[phenomenon]

    assert spatial["peak"]["probability"] == pytest.approx(center["peak"]["probability"])
    assert (spatial["peak"]["probability"] >= 0.8) is should_alert


@pytest.mark.parametrize("phenomenon", ["rainbow", "cza"])
def test_temporal_smoothing_does_not_create_support_in_an_impossible_slot(phenomenon) -> None:
    start = datetime(2026, 10, 24, 18, tzinfo=timezone.utc)

    def weather_at_time(lat, lon, **kwargs):
        future = kwargs["at_time"] > start
        return WeatherSnapshot(
            weather={
                "cloud_cover_high": 1.0, "cloud_optical_thickness": 0.02,
                "ice_cloud_fraction": 1.0, "precipitation": 2.0 if future else 0.0,
            },
            sources=(SourceAttribution(name="gfs", timestamp="20261024 12z f006"),),
        )

    def sun_at_time(lat, lon, time):
        elevation = 35.0 if phenomenon == "cza" and time == start else 22.0
        return {"elevation": elevation, "azimuth": 180.0}

    with patch("prediction.pipeline.get_weather_snapshot", side_effect=weather_at_time), patch(
        "prediction.pipeline.get_solar_position", side_effect=sun_at_time
    ):
        result = predict_all(
            32.847, -96.806, mode="forecast", illumination="solar", phenomena=(phenomenon,),
            at_time=start, time_window_hours=(0, 1), lightweight=True,
        )
    prediction = _phenomena_by_id(result)[phenomenon]
    assert prediction["current"]["probability"] == 0.0
    assert prediction["timeline"][0]["probability"] == 0.0
    assert prediction["peak"]["probability"] > 0.8
    assert prediction["peak"]["time"] == "2026-10-24T19:00:00Z"


def test_predict_all_returns_structured_payload() -> None:
    with patch(
        "prediction.pipeline.get_weather_snapshot",
        return_value=WeatherSnapshot(
            weather={
                "temp_250": -30.0,
                "humidity_250": 70.0,
                "cloud_cover_high": 0.8,
                "precipitation": 1.2,
            },
            sources=(
                SourceAttribution(name="goes-east", timestamp="20260407 124617z"),
                SourceAttribution(name="metar", timestamp="20260407 1653z"),
            ),
        ),
    ) as get_weather_mock:
        with patch(
            "prediction.pipeline.get_solar_position",
            return_value={
                "elevation": 20.0,
                "azimuth": 180.0,
            },
        ):
            with patch(
                "prediction.pipeline.get_lunar_position",
                return_value={"elevation": 12.0, "azimuth": 140.0, "phase": 0.9, "illuminance": 0.8},
            ):
                result = predict_all(
                    32.8,
                    -96.8,
                    at_time=datetime(2026, 4, 7, 18, 0, tzinfo=timezone.utc),
                    mode="observed",
                    keep_downloaded_files=True,
                    download_dir="/tmp/noaa-gfs-cache",
                )

    assert any(
        call.args == (32.8, -96.8)
        and call.kwargs == {
            "mode": "observed",
            "keep_downloaded_files": True,
            "download_dir": "/tmp/noaa-gfs-cache",
            "at_time": datetime(2026, 4, 7, 18, 0, tzinfo=timezone.utc),
        }
        for call in get_weather_mock.call_args_list
    )
    assert set(result) == {
        "generated_at",
        "request",
        "sources",
        "clouds",
        "celestial",
        "phenomena",
    }
    assert result["generated_at"].endswith("Z")
    assert result["request"] == {
        "location": {"lat": 32.8, "lon": -96.8},
        "mode": "observed",
        "prediction_time": "2026-04-07T18:00:00Z",
        "time_window_hours": [0, 1, 2, 3],
        "options": {
            "lightweight": False,
            "debug": False,
            "illumination": "solar,lunar",
        },
    }
    assert result["sources"] == [
        {"id": "goes-east", "label": "GOES East", "kind": "satellite", "timestamp": "20260407 124617z"},
        {"id": "metar", "label": "METAR", "kind": "surface_observation", "timestamp": "20260407 1653z"},
    ]
    assert result["clouds"]["schema_version"] == "1.0"
    assert result["clouds"]["sources"] == {"goes": True, "metar": True}
    assert result["clouds"]["timestamp"] == "2026-04-07T18:00:00Z"
    assert result["clouds"]["cloud_layers"][0]["wmo_code"] == "Cs"
    assert result["celestial"] == {
        "sun": {"altitude": 20.0},
        "moon": {"altitude": 12.0},
    }

    phenomena = _phenomena_by_id(result)
    assert tuple(phenomena) == PHENOMENA
    for phenomenon_id, payload in phenomena.items():
        assert set(payload) == {"id", "label", "category", "current", "peak", "timeline"}
        assert payload["id"] == phenomenon_id
        assert isinstance(payload["label"], str)
        assert isinstance(payload["category"], str)
        current = payload["current"]
        peak = payload["peak"]
        timeline = payload["timeline"]
        assert isinstance(current, dict)
        assert isinstance(peak, dict)
        assert isinstance(timeline, list)
        assert 0.0 <= current["probability"] <= 1.0
        assert 0.0 <= current["confidence"] <= 1.0
        assert set(current["confidence_components"]) == {"data", "spatial", "temporal", "feature"}
        assert set(current["spatial_context"]) == {
            "radius_km",
            "aggregation",
            "center_probability",
            "mean_probability",
            "max_probability",
            "min_probability",
            "spatial_variance",
            "spatial_consistency",
            "spatial_gradient",
            "edge_signal",
        }
        assert isinstance(current["reason"], str)
        assert peak["time"].endswith("Z")
        assert len(timeline) == 4
        assert timeline[0]["label"] == "now"
        assert timeline[0]["offset_hours"] == 0
        assert timeline[0]["probability"] == current["probability"]


def test_predict_all_passes_mode_to_weather_layer() -> None:
    with patch(
        "prediction.pipeline.get_weather_snapshot",
        return_value=WeatherSnapshot(
            weather={
                "temp_250": -30.0,
                "humidity_250": 70.0,
                "cloud_cover_high": 0.8,
                "precipitation": 1.2,
            },
            sources=(SourceAttribution(name="gfs", timestamp="20260407 12z f006"),),
        ),
    ) as get_weather_mock:
        with patch(
            "prediction.pipeline.get_solar_position",
            return_value={
                "elevation": 20.0,
                "azimuth": 180.0,
            },
        ):
            with patch(
                "prediction.pipeline.get_lunar_position",
                return_value={"elevation": 12.0, "azimuth": 140.0, "phase": 0.9, "illuminance": 0.8},
            ):
                predict_all(32.8, -96.8, mode="observed")

    assert get_weather_mock.call_count >= 1
    assert all(call.kwargs["mode"] == "observed" for call in get_weather_mock.call_args_list)
    assert all("at_time" in call.kwargs for call in get_weather_mock.call_args_list)


def test_predict_all_supports_selected_phenomena_and_debug_payloads() -> None:
    scalar_components = ModelComponents(probability=0.12349, physical=0.6, visibility=0.7, geometry=0.8)

    def fake_debug_predictor(features: dict[str, float], return_components: bool = False):
        del features
        if return_components:
            return scalar_components
        return scalar_components.probability

    with patch(
        "prediction.pipeline.get_weather_snapshot",
        return_value=WeatherSnapshot(
            weather={},
            sources=(SourceAttribution(name="gfs", timestamp="20260407 12z f006"),),
        ),
    ):
        with patch(
            "prediction.pipeline.get_solar_position",
            return_value={"elevation": 20.0, "azimuth": 180.0},
        ):
            with patch(
                "prediction.pipeline.compute_features",
                return_value={
                    "solar_elevation": 20.0,
                    "solar_azimuth": 180.0,
                    "sun_visible": 1.0,
                    "cloud_variability": 0.2,
                    "wind_shear_250": 0.1,
                },
            ):
                with patch("prediction.pipeline.predict_halo", side_effect=fake_debug_predictor):
                    result = predict_all(
                        32.8,
                        -96.8,
                        time_window_hours=(0, 1),
                        lightweight=True,
                        debug=True,
                        phenomena=("halo", "rainbow"),
                    )

    assert result["request"]["time_window_hours"] == [0, 1]
    assert result["request"]["options"] == {
        "lightweight": True,
        "debug": True,
        "illumination": "solar,lunar",
        "phenomena": ["halo", "rainbow"],
    }
    phenomena = _phenomena_by_id(result)
    assert tuple(phenomena) == ("halo", "rainbow")
    assert phenomena["halo"]["current"]["probability"] == 0.123
    assert phenomena["halo"]["timeline"] == [
        {"label": "now", "offset_hours": 0, "probability": 0.123},
        {"label": "1h", "offset_hours": 1, "probability": 0.123},
    ]
    assert phenomena["halo"]["current"]["debug"] == {"P": 0.6, "V": 0.7, "G": 0.8}
    assert phenomena["halo"]["current"]["spatial_context"]["aggregation"] == "weighted_blend"


def test_predict_all_supports_lightweight_mode() -> None:
    with patch(
        "prediction.pipeline.get_weather_snapshot",
        return_value=WeatherSnapshot(
            weather={},
            sources=(SourceAttribution(name="gfs", timestamp="20260407 12z f006"),),
        ),
    ) as get_weather_mock:
        with patch(
            "prediction.pipeline.get_solar_position",
            return_value={"elevation": 20.0, "azimuth": 180.0},
        ):
            with patch(
                "prediction.pipeline.get_lunar_position",
                return_value={"elevation": 12.0, "azimuth": 140.0, "phase": 0.9, "illuminance": 0.8},
            ):
                result = predict_all(32.8, -96.8, lightweight=True, mode="observed")

    halo = _phenomena_by_id(result)["halo"]
    assert halo["current"]["spatial_context"]["center_probability"] == halo["current"]["spatial_context"]["max_probability"]
    assert get_weather_mock.call_count == 1


def test_predict_all_ignores_time_slots_beyond_three_hours() -> None:
    with patch(
        "prediction.pipeline.get_weather_snapshot",
        return_value=WeatherSnapshot(
            weather={},
            sources=(SourceAttribution(name="gfs", timestamp="20260407 12z f006"),),
        ),
    ):
        with patch(
            "prediction.pipeline.get_solar_position",
            return_value={"elevation": 20.0, "azimuth": 180.0},
        ):
            with patch(
                "prediction.pipeline.get_lunar_position",
                return_value={"elevation": 12.0, "azimuth": 140.0, "phase": 0.9, "illuminance": 0.8},
            ):
                result = predict_all(32.8, -96.8, time_window_hours=(0, 1, 3, 6, 12))

    assert result["request"]["time_window_hours"] == [0, 1, 3]
    halo_timeline = _phenomena_by_id(result)["halo"]["timeline"]
    assert [entry["label"] for entry in halo_timeline] == ["now", "1h", "3h"]


def test_predict_all_supports_lunar_mode() -> None:
    with patch(
        "prediction.pipeline.get_weather_snapshot",
        return_value=WeatherSnapshot(
            weather={
                "cloud_cover_high": 0.6,
                "cloud_optical_thickness": 0.2,
                "humidity_250": 0.8,
                "precipitation": 0.6,
            },
            sources=(SourceAttribution(name="gfs", timestamp="20260407 12z f006"),),
        ),
    ):
        with patch(
            "prediction.pipeline.get_solar_position",
            return_value={"elevation": -18.0, "azimuth": 180.0},
        ):
            with patch(
                "prediction.pipeline.get_lunar_position",
                return_value={"elevation": 12.0, "azimuth": 140.0, "phase": 0.95, "illuminance": 0.93},
            ):
                result = predict_all(32.8, -96.8, illumination="lunar", lightweight=True)

    assert result["request"]["options"]["illumination"] == "lunar"
    assert result["celestial"] == {
        "sun": {"altitude": -18.0},
        "moon": {"altitude": 12.0},
    }
    phenomena = _phenomena_by_id(result)
    assert tuple(phenomena) == (
        "lunar_halo",
        "paraselenae",
        "lunar_pillar",
        "lunar_corona",
        "moonbow",
    )
