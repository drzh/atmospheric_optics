from __future__ import annotations

import json

from cli.main import main


def test_cli_main_prints_prediction_json(capsys, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_predict_all(lat: float, lon: float, **kwargs: object) -> dict[str, object]:
        captured["lat"] = lat
        captured["lon"] = lon
        captured.update(kwargs)
        return {
            "halo": 0.8,
            "parhelia": 0.8,
            "cza": 0.7,
            "rainbow": 0.1,
            "sources": [
                {"name": "goes-east", "timestamp": "20260407 124617z"},
                {"name": "metar", "timestamp": "20260407 1653z"},
            ],
        }

    monkeypatch.setattr(
        "cli.main.predict_all",
        fake_predict_all,
    )

    exit_code = main(
        [
            "--lat",
            "32",
            "--lon",
            "-96",
            "--mode",
            "observed",
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
        "mode": "observed",
        "keep_downloaded_files": True,
        "download_dir": "/tmp/noaa-gfs-cache",
    }
    assert output["halo"] == 0.8
    assert output["cza"] == 0.7
    assert output["sources"] == [
        {"name": "goes-east", "timestamp": "20260407 124617z"},
        {"name": "metar", "timestamp": "20260407 1653z"},
    ]
