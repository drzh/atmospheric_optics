from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

from core.predictor import predict_all
from data_ingestion.weather import SourceAttribution, WeatherSnapshot


def test_predict_all_returns_bounded_probabilities() -> None:
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

    get_weather_mock.assert_called_once_with(
        32.8,
        -96.8,
        mode="observed",
        keep_downloaded_files=True,
        download_dir="/tmp/noaa-gfs-cache",
    )
    assert set(result) == {"halo", "parhelia", "cza", "rainbow", "sources"}
    assert all(0.0 <= value <= 1.0 for key, value in result.items() if key != "sources")
    assert result["sources"] == [
        {"name": "goes-east", "timestamp": "20260407 124617z"},
        {"name": "metar", "timestamp": "20260407 1653z"},
    ]


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

    assert get_weather_mock.call_args.kwargs["mode"] == "observed"


def test_predict_all_rounds_probabilities_to_three_decimal_places() -> None:
    with patch(
        "core.predictor.get_weather_snapshot",
        return_value=WeatherSnapshot(
            weather={},
            sources=(SourceAttribution(name="gfs", timestamp="20260407 12z f006"),),
        ),
    ):
        with patch("core.predictor.get_solar_position", return_value={"elevation": 20.0, "azimuth": 180.0}):
            with patch("core.predictor.compute_features", return_value={"dummy": 1.0}):
                with patch("core.predictor.predict_halo", return_value=0.12349):
                    with patch("core.predictor.predict_parhelia", return_value=0.98764):
                        with patch("core.predictor.predict_cza", return_value=2.649168282275027e-10):
                            with patch("core.predictor.predict_rainbow", return_value=0.021146560833793692):
                                result = predict_all(32.8, -96.8)

    assert result["halo"] == 0.123
    assert result["parhelia"] == 0.988
    assert result["cza"] == 0.0
    assert result["rainbow"] == 0.021
