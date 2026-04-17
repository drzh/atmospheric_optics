from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

import api.main as api_main


def _sample_payload() -> dict[str, object]:
    return {
        "generated_at": "2026-04-13T17:00:00Z",
        "request": {
            "location": {"lat": 32.0, "lon": -96.0},
            "mode": "observed",
            "prediction_time": "2026-04-13T18:00:00Z",
            "time_window_hours": [0, 1, 3],
            "options": {
                "lightweight": False,
                "debug": False,
                "illumination": "solar",
                "phenomena": ["halo", "fogbow"],
            },
        },
        "phenomena": [
            {
                "id": "halo",
                "label": "Halo",
                "category": "ice_crystal",
                "current": {
                    "probability": 0.2,
                    "confidence": 0.7,
                    "reason": "halo reason",
                    "spatial_context": {
                        "radius_km": 40.0,
                        "aggregation": "weighted_blend",
                    },
                },
                "peak": {
                    "probability": 0.25,
                    "time": "2026-04-13T19:00:00Z",
                },
                "timeline": [
                    {"label": "now", "offset_hours": 0, "probability": 0.2},
                    {"label": "1h", "offset_hours": 1, "probability": 0.25},
                    {"label": "3h", "offset_hours": 3, "probability": 0.18},
                ],
            },
            {
                "id": "fogbow",
                "label": "Fogbow",
                "category": "water_droplet",
                "current": {
                    "probability": 0.5,
                    "confidence": 0.6,
                    "reason": "fogbow reason",
                    "spatial_context": {
                        "radius_km": 10.0,
                        "aggregation": "weighted_blend",
                    },
                },
                "peak": {
                    "probability": 0.52,
                    "time": "2026-04-13T19:00:00Z",
                },
                "timeline": [
                    {"label": "now", "offset_hours": 0, "probability": 0.5},
                    {"label": "1h", "offset_hours": 1, "probability": 0.52},
                    {"label": "3h", "offset_hours": 3, "probability": 0.48},
                ],
            },
        ],
        "sources": [
            {"id": "goes-east", "label": "GOES East", "kind": "satellite", "timestamp": "20260407 124617z"},
            {"id": "metar", "label": "METAR", "kind": "surface_observation", "timestamp": "20260407 1653z"},
        ],
    }


def test_build_prediction_response_delegates_to_predictor(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_predict_all(
        lat: float,
        lon: float,
        mode: str = "forecast",
        illumination: str = "solar",
        at_time: datetime | None = None,
        time_window_hours: tuple[int, ...] | None = None,
        phenomena: tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        captured["lat"] = lat
        captured["lon"] = lon
        captured["mode"] = mode
        captured["illumination"] = illumination
        captured["at_time"] = at_time
        captured["time_window_hours"] = time_window_hours
        captured["phenomena"] = phenomena
        return _sample_payload()

    monkeypatch.setattr(api_main, "predict_all", fake_predict_all)

    result = api_main.build_prediction_response(
        32.0,
        -96.0,
        mode="observed",
        at_time="2026-04-13T18:00:00Z",
        time_window_hours="0,1,3",
        phenomena="halo,fogbow",
    )

    assert captured == {
        "lat": 32.0,
        "lon": -96.0,
        "mode": "observed",
        "illumination": "solar",
        "at_time": datetime(2026, 4, 13, 18, 0, tzinfo=timezone.utc),
        "time_window_hours": (0, 1, 3),
        "phenomena": ("halo", "fogbow"),
    }
    assert result["phenomena"][0]["peak"]["probability"] == 0.25
    assert result["phenomena"][0]["current"]["spatial_context"]["aggregation"] == "weighted_blend"
    assert result["phenomena"][1]["current"]["reason"] == "fogbow reason"
    assert result["sources"] == [
        {"id": "goes-east", "label": "GOES East", "kind": "satellite", "timestamp": "20260407 124617z"},
        {"id": "metar", "label": "METAR", "kind": "surface_observation", "timestamp": "20260407 1653z"},
    ]


@pytest.mark.skipif(api_main.FASTAPI_AVAILABLE, reason="WSGI fallback is only used when FastAPI is unavailable.")
def test_wsgi_predict_route_returns_json(monkeypatch) -> None:
    monkeypatch.setattr(
        api_main,
        "build_prediction_response",
        lambda lat, lon, mode="forecast", illumination="solar", at_time=None, time_window_hours=None, phenomena=None, spatial_resolution_km=None, lightweight=False, debug=False: _sample_payload(),
    )

    captured: dict[str, object] = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        captured["status"] = status
        captured["headers"] = headers

    body = b"".join(
        api_main.app(
            {
                "REQUEST_METHOD": "GET",
                "PATH_INFO": "/predict",
                "QUERY_STRING": "lat=32&lon=-96&mode=observed&at_time=2026-04-13T18:00:00Z&time_window_hours=0,1,3&phenomena=halo,fogbow",
            },
            start_response,
        )
    )

    assert captured["status"] == "200 OK"
    assert json.loads(body) == _sample_payload()
