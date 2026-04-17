"""Command-line interface for atmospheric optics prediction."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.predictor import predict_all

WEATHER_MODES = ("forecast", "observed")
ILLUMINATION_MODES = ("solar", "lunar")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict atmospheric optical phenomena.")
    parser.add_argument("--lat", type=float, required=True, help="Latitude in decimal degrees.")
    parser.add_argument("--lon", type=float, required=True, help="Longitude in decimal degrees.")
    parser.add_argument(
        "--mode",
        choices=WEATHER_MODES,
        default="forecast",
        help="Weather input mode: forecast uses NOAA GFS, observed uses GOES cloud layers plus nearby METAR observations.",
    )
    parser.add_argument(
        "--illumination",
        choices=ILLUMINATION_MODES,
        default="solar",
        help="Illumination mode: solar predicts sunlit optics, lunar predicts moonlit optics.",
    )
    parser.add_argument(
        "--keep-downloaded-files",
        action="store_true",
        help="Keep the downloaded weather-source artifacts on disk.",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        help="Directory where downloaded weather-source artifacts should be saved.",
    )
    parser.add_argument(
        "--at-time",
        type=str,
        help="Prediction time in ISO 8601 format. Naive values are interpreted as UTC.",
    )
    parser.add_argument(
        "--time-window-hours",
        type=str,
        help="Comma-separated forecast offsets in hours, such as 0,1,2,3.",
    )
    parser.add_argument(
        "--phenomena",
        type=str,
        help="Optional comma-separated phenomenon ids such as halo,rainbow.",
    )
    parser.add_argument(
        "--spatial-resolution-km",
        type=float,
        help="Optional 3x3 sample spacing in kilometers within the adaptive radius.",
    )
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Skip spatial sampling and evaluate only the center point for faster output.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include per-phenomenon physical, visibility, and geometry components.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parsed_phenomena = _parse_csv_values(args.phenomena)
    predictor_kwargs: dict[str, object] = {
        "at_time": _parse_at_time(args.at_time),
        "mode": args.mode,
        "illumination": args.illumination,
        "keep_downloaded_files": args.keep_downloaded_files or bool(args.download_dir),
        "download_dir": args.download_dir,
        "time_window_hours": _parse_time_window_hours(args.time_window_hours),
    }
    if parsed_phenomena is not None:
        predictor_kwargs["phenomena"] = parsed_phenomena
    if args.spatial_resolution_km is not None:
        predictor_kwargs["spatial_resolution_km"] = args.spatial_resolution_km
    if args.lightweight:
        predictor_kwargs["lightweight"] = True
    if args.debug:
        predictor_kwargs["debug"] = True
    payload = predict_all(
        args.lat,
        args.lon,
        **predictor_kwargs,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parse_at_time(value: str | None) -> datetime | None:
    if value is None or not value.strip():
        return None

    parsed_value = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed_value.tzinfo is None:
        return parsed_value.replace(tzinfo=timezone.utc)
    return parsed_value.astimezone(timezone.utc)


def _parse_time_window_hours(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None

    hours: list[int] = []
    for part in value.split(","):
        normalized_part = part.strip()
        if not normalized_part:
            continue
        hours.append(int(normalized_part))
    return tuple(hours)


def _parse_csv_values(value: str | None) -> tuple[str, ...] | None:
    if value is None or not value.strip():
        return None

    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        return None
    return tuple(items)


if __name__ == "__main__":
    raise SystemExit(main())
