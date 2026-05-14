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

from interfaces.parameters import (
    DEFAULT_ILLUMINATION,
    ILLUMINATION_MODES,
    WEATHER_MODES,
    normalize_illumination,
    parse_at_time,
    parse_csv_values,
    parse_time_window_hours,
)
from prediction.pipeline import predict_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict atmospheric optical phenomena.")
    parser.add_argument(
        "--lat",
        action="append",
        required=True,
        help="Latitude in decimal degrees. Repeat the option or use comma-separated values for multiple locations.",
    )
    parser.add_argument(
        "--lon",
        action="append",
        required=True,
        help=(
            "Longitude in decimal degrees. Repeat the option or use comma-separated values for multiple locations. "
            "Use --lon=-96.8,-97.8 for comma-separated negative longitudes."
        ),
    )
    parser.add_argument(
        "--site",
        action="append",
        help="Optional site names matching --lat/--lon. Repeat the option or use comma-separated values. Defaults to NA.",
    )
    parser.add_argument(
        "--mode",
        choices=WEATHER_MODES,
        default="forecast",
        help="Weather input mode: forecast uses NOAA GFS, observed uses GOES cloud layers plus nearby METAR observations.",
    )
    parser.add_argument(
        "--illumination",
        default=DEFAULT_ILLUMINATION,
        help=(
            "Comma-separated illumination modes. "
            f"Use {', '.join(ILLUMINATION_MODES)} or {DEFAULT_ILLUMINATION}."
        ),
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


def _flatten_csv_values(values: list[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()

    flattened: list[str] = []
    for value in values:
        parsed_values = parse_csv_values(value)
        if parsed_values is None:
            continue
        flattened.extend(parsed_values)
    return tuple(flattened)


def _parse_float_values(values: list[str] | None, option_name: str) -> tuple[float, ...]:
    parsed: list[float] = []
    for value in _flatten_csv_values(values):
        try:
            parsed.append(float(value))
        except ValueError as exc:
            raise ValueError(f"{option_name} must contain numeric values: {value}") from exc
    if not parsed:
        raise ValueError(f"{option_name} requires at least one value.")
    return tuple(parsed)


def _parse_site_values(values: list[str] | None, location_count: int) -> tuple[str, ...]:
    parsed = _flatten_csv_values(values)
    if not parsed:
        return tuple("NA" for _ in range(location_count))
    if len(parsed) != location_count:
        raise ValueError(
            f"--site must contain {location_count} value(s) to match --lat/--lon; got {len(parsed)}."
        )
    return parsed


def _location_requests(args: argparse.Namespace) -> tuple[dict[str, object], ...]:
    latitudes = _parse_float_values(args.lat, "--lat")
    longitudes = _parse_float_values(args.lon, "--lon")
    if len(latitudes) != len(longitudes):
        raise ValueError(
            f"--lat and --lon must contain the same number of values; got {len(latitudes)} and {len(longitudes)}."
        )

    sites = _parse_site_values(args.site, len(latitudes))
    return tuple(
        {
            "site": sites[index],
            "lat": latitudes[index],
            "lon": longitudes[index],
        }
        for index in range(len(latitudes))
    )


def _location_payload(location: dict[str, object], include_site: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "lat": location["lat"],
        "lon": location["lon"],
    }
    if include_site:
        payload["site"] = location["site"]
    return payload


def _enrich_prediction_location(
    payload: dict[str, object],
    location: dict[str, object],
    include_site: bool,
) -> dict[str, object]:
    request = payload.get("request")
    if not isinstance(request, dict):
        return payload

    request_location = request.get("location")
    if not isinstance(request_location, dict):
        request_location = {}
        request["location"] = request_location
    request_location["lat"] = location["lat"]
    request_location["lon"] = location["lon"]
    if include_site:
        request_location["site"] = location["site"]
        payload["site"] = location["site"]
    return payload


def _combined_generated_at(payloads: list[dict[str, object]]) -> str:
    for payload in payloads:
        generated_at = payload.get("generated_at")
        if isinstance(generated_at, str) and generated_at.strip():
            return generated_at
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _combined_request(
    payloads: list[dict[str, object]],
    locations: tuple[dict[str, object], ...],
) -> dict[str, object]:
    first_request = payloads[0].get("request") if payloads else {}
    request: dict[str, object] = {}
    if isinstance(first_request, dict):
        request.update(
            {
                str(key): value
                for key, value in first_request.items()
                if key != "location"
            }
        )
    request["locations"] = [
        _location_payload(location, include_site=True)
        for location in locations
    ]
    return request


def _build_multi_location_payload(
    payloads: list[dict[str, object]],
    locations: tuple[dict[str, object], ...],
) -> dict[str, object]:
    return {
        "generated_at": _combined_generated_at(payloads),
        "request": _combined_request(payloads, locations),
        "locations": [
            {
                "site": location["site"],
                "location": _location_payload(location, include_site=False),
                "prediction": payloads[index],
            }
            for index, location in enumerate(locations)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    locations = _location_requests(args)
    parsed_phenomena = parse_csv_values(args.phenomena)
    predictor_kwargs: dict[str, object] = {
        "at_time": parse_at_time(args.at_time),
        "mode": args.mode,
        "illumination": normalize_illumination(args.illumination),
        "keep_downloaded_files": args.keep_downloaded_files or bool(args.download_dir),
        "download_dir": args.download_dir,
        "time_window_hours": parse_time_window_hours(args.time_window_hours),
    }
    if parsed_phenomena is not None:
        predictor_kwargs["phenomena"] = parsed_phenomena
    if args.spatial_resolution_km is not None:
        predictor_kwargs["spatial_resolution_km"] = args.spatial_resolution_km
    if args.lightweight:
        predictor_kwargs["lightweight"] = True
    if args.debug:
        predictor_kwargs["debug"] = True

    include_site = args.site is not None or len(locations) > 1
    payloads = [
        _enrich_prediction_location(
            predict_all(
                float(location["lat"]),
                float(location["lon"]),
                **predictor_kwargs,
            ),
            location,
            include_site=include_site,
        )
        for location in locations
    ]
    payload = (
        payloads[0]
        if len(payloads) == 1
        else _build_multi_location_payload(payloads, locations)
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
