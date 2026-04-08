"""Command-line interface for atmospheric optics prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.predictor import predict_all

WEATHER_MODES = ("forecast", "observed")


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
        "--keep-downloaded-files",
        action="store_true",
        help="Keep the downloaded weather-source artifacts on disk.",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        help="Directory where downloaded weather-source artifacts should be saved.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = predict_all(
        args.lat,
        args.lon,
        mode=args.mode,
        keep_downloaded_files=args.keep_downloaded_files or bool(args.download_dir),
        download_dir=args.download_dir,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
