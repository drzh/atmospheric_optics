from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from api.server import build_prediction_response
from cli.command import main
from prediction.pipeline import predict_all
from weather_sources.snapshots import SourceAttribution, WeatherSnapshot

NEW = ("solar_corona", "anticrepuscular_rays", "secondary_rainbow")
LAT = 32.847


def snapshot(lat, lon, **kwargs):
    total = 0.5 if lat < LAT else 0.0 if lat > LAT else 0.25
    return WeatherSnapshot(
        weather={"cloud_cover_high": 0.5, "cloud_cover_total": total,
                 "cloud_optical_thickness": 0.3, "cloud_variability": 0.3, "precipitation": 2.0},
        sources=(SourceAttribution(name="goes-east", timestamp="20261024 180000z"),),
    )


@pytest.fixture
def model_inputs(monkeypatch):
    monkeypatch.setattr("prediction.pipeline.get_weather_snapshot", snapshot)
    monkeypatch.setattr("prediction.pipeline.get_solar_position", lambda *args: {"elevation": 6.0, "azimuth": 180.0})
    monkeypatch.setattr("prediction.pipeline.get_lunar_position", lambda *args: {"elevation": 20.0, "azimuth": 90.0, "phase": 1.0, "illuminance": 1.0})


def test_new_entries_are_in_default_output_and_selected_forecasts(model_inputs) -> None:
    payload = predict_all(LAT, -96.806, at_time=datetime(2026, 10, 24, 18, tzinfo=timezone.utc), debug=True)
    items = {entry["id"]: entry for entry in payload["phenomena"]}
    assert len(items) == 17
    assert set(NEW).issubset(items)
    rays = items["anticrepuscular_rays"]
    assert rays["current"]["probability"] > 0.8
    assert rays["current"]["viewing_azimuth"] == 0.0
    assert items["solar_corona"]["current"]["viewing_azimuth"] == 180.0
    assert rays["current"]["spatial_context"]["source_cloud_fraction"] == pytest.approx(0.5)
    assert rays["current"]["spatial_context"]["antisolar_cloud_fraction"] == 0.0
    assert rays["current"]["debug"]["P"] == 1.0
    json.dumps(payload, allow_nan=False)


def test_same_average_clouds_with_wrong_direction_do_not_predict_rays(model_inputs, monkeypatch) -> None:
    def reversed_clouds(lat, lon, **kwargs):
        result = snapshot(lat, lon, **kwargs)
        result.weather["cloud_cover_total"] = 0.5 if lat > LAT else 0.0 if lat < LAT else 0.25
        return result
    monkeypatch.setattr("prediction.pipeline.get_weather_snapshot", reversed_clouds)
    payload = predict_all(LAT, -96.806, phenomena=("anticrepuscular_rays",), illumination="solar")
    assert payload["phenomena"][0]["peak"]["probability"] == 0.0


def test_missing_directional_evidence_and_lightweight_mode_are_explicit(model_inputs, monkeypatch) -> None:
    payload = predict_all(LAT, -96.806, phenomena=("anticrepuscular_rays",), illumination="solar", lightweight=True)
    assert payload["phenomena"][0]["peak"]["probability"] == 0.0
    assert "center-only" in payload["phenomena"][0]["current"]["reason"]
    def missing_total(lat, lon, **kwargs):
        result = snapshot(lat, lon, **kwargs)
        del result.weather["cloud_cover_total"]
        return result
    monkeypatch.setattr("prediction.pipeline.get_weather_snapshot", missing_total)
    payload = predict_all(LAT, -96.806, phenomena=("anticrepuscular_rays",), illumination="solar")
    assert payload["phenomena"][0]["peak"]["probability"] == 0.0
    json.dumps(payload, allow_nan=False)


def test_new_solar_corona_does_not_use_moonlight_at_night(model_inputs, monkeypatch) -> None:
    monkeypatch.setattr("prediction.pipeline.get_solar_position", lambda *args: {"elevation": -20.0, "azimuth": 180.0})
    payload = predict_all(LAT, -96.806, phenomena=("solar_corona", "lunar_corona"))
    items = {entry["id"]: entry for entry in payload["phenomena"]}
    assert items["solar_corona"]["peak"]["probability"] == 0.0
    assert items["lunar_corona"]["peak"]["probability"] > 0.8


def test_cli_and_api_accept_new_selection(model_inputs, capsys) -> None:
    selected = ",".join(NEW)
    assert main(["--lat", str(LAT), "--lon", "-96.806", "--mode", "observed", "--illumination", "solar", "--phenomena", selected]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert tuple(entry["id"] for entry in payload["phenomena"]) == NEW
    payload = build_prediction_response(lat=LAT, lon=-96.806, mode="observed", illumination="solar", phenomena=selected)
    assert tuple(entry["id"] for entry in payload["phenomena"]) == NEW
