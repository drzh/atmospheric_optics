from __future__ import annotations

import json
from datetime import datetime, timezone

from cli.main import main


def test_cli_main_prints_prediction_json(capsys, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_predict_all(lat: float, lon: float, **kwargs: object) -> dict[str, object]:
        captured["lat"] = lat
        captured["lon"] = lon
        captured.update(kwargs)
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
                        "probability": 0.8,
                        "confidence": 0.7,
                        "reason": "halo reason",
                        "spatial_context": {
                            "radius_km": 40.0,
                            "aggregation": "weighted_blend",
                        },
                    },
                    "peak": {
                        "probability": 0.9,
                        "time": "2026-04-13T19:00:00Z",
                    },
                    "timeline": [
                        {"label": "now", "offset_hours": 0, "probability": 0.8},
                        {"label": "1h", "offset_hours": 1, "probability": 0.9},
                        {"label": "3h", "offset_hours": 3, "probability": 0.7},
                    ],
                },
                {
                    "id": "fogbow",
                    "label": "Fogbow",
                    "category": "water_droplet",
                    "current": {
                        "probability": 0.2,
                        "confidence": 0.2,
                        "reason": "fogbow reason",
                        "spatial_context": {
                            "radius_km": 10.0,
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
            ],
            "sources": [
                {"id": "goes-east", "label": "GOES East", "kind": "satellite", "timestamp": "20260407 124617z"},
                {"id": "metar", "label": "METAR", "kind": "surface_observation", "timestamp": "20260407 1653z"},
            ],
        }

    monkeypatch.setattr("cli.main.predict_all", fake_predict_all)

    exit_code = main(
        [
            "--lat",
            "32",
            "--lon",
            "-96",
            "--mode",
            "observed",
            "--at-time",
            "2026-04-13T18:00:00Z",
            "--time-window-hours",
            "0,1,3",
            "--phenomena",
            "halo,fogbow",
            "--keep-downloaded-files",
            "--download-dir",
            "/tmp/noaa-gfs-cache",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert captured == {
        "lat": 32.0,
        "lon": -96.0,
        "at_time": datetime(2026, 4, 13, 18, 0, tzinfo=timezone.utc),
        "mode": "observed",
        "illumination": "solar",
        "keep_downloaded_files": True,
        "download_dir": "/tmp/noaa-gfs-cache",
        "time_window_hours": (0, 1, 3),
        "phenomena": ("halo", "fogbow"),
    }
    assert output["request"]["prediction_time"] == "2026-04-13T18:00:00Z"
    assert output["phenomena"][0]["peak"]["probability"] == 0.9
    assert output["phenomena"][0]["current"]["spatial_context"]["aggregation"] == "weighted_blend"
    assert output["phenomena"][1]["current"]["reason"] == "fogbow reason"
    assert output["sources"] == [
        {"id": "goes-east", "kind": "satellite", "label": "GOES East", "timestamp": "20260407 124617z"},
        {"id": "metar", "kind": "surface_observation", "label": "METAR", "timestamp": "20260407 1653z"},
    ]
