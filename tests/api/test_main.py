from __future__ import annotations

import json

import pytest

import api.main as api_main


def test_build_prediction_response_delegates_to_predictor(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_predict_all(lat: float, lon: float, mode: str = "forecast") -> dict[str, object]:
        captured["lat"] = lat
        captured["lon"] = lon
        captured["mode"] = mode
        return {
            "halo": 0.2,
            "parhelia": 0.2,
            "cza": 0.1,
            "rainbow": 0.7,
            "sources": [
                {"name": "goes-east", "timestamp": "20260407 124617z"},
                {"name": "metar", "timestamp": "20260407 1653z"},
            ],
        }

    monkeypatch.setattr(
        api_main,
        "predict_all",
        fake_predict_all,
    )

    result = api_main.build_prediction_response(32.0, -96.0, mode="observed")

    assert captured == {"lat": 32.0, "lon": -96.0, "mode": "observed"}
    assert result["rainbow"] == 0.7
    assert result["sources"] == [
        {"name": "goes-east", "timestamp": "20260407 124617z"},
        {"name": "metar", "timestamp": "20260407 1653z"},
    ]


@pytest.mark.skipif(api_main.FASTAPI_AVAILABLE, reason="WSGI fallback is only used when FastAPI is unavailable.")
def test_wsgi_predict_route_returns_json(monkeypatch) -> None:
    monkeypatch.setattr(
        api_main,
        "build_prediction_response",
        lambda lat, lon, mode="forecast": {
            "halo": 0.2,
            "parhelia": 0.2,
            "cza": 0.1,
            "rainbow": 0.7,
            "sources": [{"name": "gfs", "timestamp": "20260407 12z f006"}],
        },
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
                "QUERY_STRING": "lat=32&lon=-96&mode=observed",
            },
            start_response,
        )
    )

    assert captured["status"] == "200 OK"
    assert json.loads(body) == {
        "halo": 0.2,
        "parhelia": 0.2,
        "cza": 0.1,
        "rainbow": 0.7,
        "sources": [{"name": "gfs", "timestamp": "20260407 12z f006"}],
    }
