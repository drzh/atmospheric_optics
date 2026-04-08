"""End-to-end predictor orchestration."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

from data_ingestion.weather import get_weather_snapshot
from feature_engineering.features import compute_features
from models.cza_model import predict_cza
from models.halo_model import predict_halo, predict_parhelia
from models.rainbow_model import predict_rainbow
from solar.solar_position import get_solar_position

OUTPUT_DECIMALS = 3


def predict_all(
    lat: float,
    lon: float,
    at_time: datetime | None = None,
    mode: str = "forecast",
    keep_downloaded_files: bool = False,
    download_dir: str | Path | None = None,
) -> dict[str, object]:
    """Predict all supported atmospheric optics probabilities."""

    prediction_time = _resolve_prediction_time(at_time)
    weather_snapshot = get_weather_snapshot(
        lat,
        lon,
        mode=mode,
        keep_downloaded_files=keep_downloaded_files,
        download_dir=download_dir,
    )
    weather = weather_snapshot.weather
    solar = get_solar_position(lat, lon, prediction_time)
    features = compute_features(weather, solar)

    return {
        "halo": _round_output_float(predict_halo(features)),
        "parhelia": _round_output_float(predict_parhelia(features)),
        "cza": _round_output_float(predict_cza(features)),
        "rainbow": _round_output_float(predict_rainbow(features)),
        "sources": [
            {
                "name": source.name,
                "timestamp": source.timestamp,
            }
            for source in weather_snapshot.sources
        ],
    }


def _resolve_prediction_time(at_time: datetime | None) -> datetime:
    if at_time is None:
        return datetime.now(timezone.utc)
    if at_time.tzinfo is None:
        return at_time.replace(tzinfo=timezone.utc)
    return at_time.astimezone(timezone.utc)


def _round_output_float(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(value, OUTPUT_DECIMALS)
