from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

from core.predictor import PHENOMENA, predict_all
from data_ingestion.weather import SourceAttribution, WeatherSnapshot
from models.combine import ModelComponents


def _phenomena_by_id(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    phenomena = payload.get("phenomena")
    if not isinstance(phenomena, list):
        return {}
    result: dict[str, dict[str, object]] = {}
    for entry in phenomena:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            result[entry["id"]] = entry
    return result


def test_predict_all_returns_structured_payload() -> None:
    with patch(
        "core.predictor.get_weather_snapshot",
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
            "core.predictor.get_solar_position",
            return_value={
                "elevation": 20.0,
                "azimuth": 180.0,
            },
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
    assert set(result) == {"generated_at", "request", "sources", "celestial", "phenomena"}
    assert result["generated_at"].endswith("Z")
    assert result["request"] == {
        "location": {"lat": 32.8, "lon": -96.8},
        "mode": "observed",
        "prediction_time": "2026-04-07T18:00:00Z",
        "time_window_hours": [0, 1, 2, 3],
        "options": {
            "lightweight": False,
            "debug": False,
            "illumination": "solar",
        },
    }
    assert result["sources"] == [
        {"id": "goes-east", "label": "GOES East", "kind": "satellite", "timestamp": "20260407 124617z"},
        {"id": "metar", "label": "METAR", "kind": "surface_observation", "timestamp": "20260407 1653z"},
    ]
    assert result["celestial"] == {
        "sun": {"altitude": 20.0},
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
        "core.predictor.get_weather_snapshot",
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
            "core.predictor.get_solar_position",
            return_value={
                "elevation": 20.0,
                "azimuth": 180.0,
            },
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
        "core.predictor.get_weather_snapshot",
        return_value=WeatherSnapshot(
            weather={},
            sources=(SourceAttribution(name="gfs", timestamp="20260407 12z f006"),),
        ),
    ):
        with patch(
            "core.predictor.get_solar_position",
            return_value={"elevation": 20.0, "azimuth": 180.0},
        ):
            with patch(
                "core.predictor.compute_features",
                return_value={
                    "solar_elevation": 20.0,
                    "solar_azimuth": 180.0,
                    "sun_visible": 1.0,
                    "cloud_variability": 0.2,
                    "wind_shear_250": 0.1,
                },
            ):
                with patch("core.predictor.predict_halo", side_effect=fake_debug_predictor):
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
        "illumination": "solar",
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
        "core.predictor.get_weather_snapshot",
        return_value=WeatherSnapshot(
            weather={},
            sources=(SourceAttribution(name="gfs", timestamp="20260407 12z f006"),),
        ),
    ) as get_weather_mock:
        with patch(
            "core.predictor.get_solar_position",
            return_value={"elevation": 20.0, "azimuth": 180.0},
        ):
            result = predict_all(32.8, -96.8, lightweight=True, mode="observed")

    halo = _phenomena_by_id(result)["halo"]
    assert halo["current"]["spatial_context"]["center_probability"] == halo["current"]["spatial_context"]["max_probability"]
    assert get_weather_mock.call_count == 1


def test_predict_all_ignores_time_slots_beyond_three_hours() -> None:
    with patch(
        "core.predictor.get_weather_snapshot",
        return_value=WeatherSnapshot(
            weather={},
            sources=(SourceAttribution(name="gfs", timestamp="20260407 12z f006"),),
        ),
    ):
        with patch(
            "core.predictor.get_solar_position",
            return_value={"elevation": 20.0, "azimuth": 180.0},
        ):
            result = predict_all(32.8, -96.8, time_window_hours=(0, 1, 3, 6, 12))

    assert result["request"]["time_window_hours"] == [0, 1, 3]
    halo_timeline = _phenomena_by_id(result)["halo"]["timeline"]
    assert [entry["label"] for entry in halo_timeline] == ["now", "1h", "3h"]


def test_predict_all_supports_lunar_mode() -> None:
    with patch(
        "core.predictor.get_weather_snapshot",
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
            "core.predictor.get_solar_position",
            return_value={"elevation": -18.0, "azimuth": 180.0},
        ):
            with patch(
                "core.predictor.get_lunar_position",
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
