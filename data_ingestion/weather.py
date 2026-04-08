"""Weather ingestion for atmospheric optics prediction."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import requests

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

NOMADS_GFS_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
NWS_API_BASE_URL = "https://api.weather.gov"
AWC_METAR_URL = "https://aviationweather.gov/api/data/metar"
NWS_USER_AGENT = "atmospheric-optics-predictor/1.0 (local development)"
REQUEST_TIMEOUT_SECONDS = 20
NETWORK_RETRIES = 2
GRID_BUFFER_DEGREES = 0.125
MAX_CYCLE_LOOKBACKS = 4
MAX_OBS_STATIONS = 5
METAR_HIGH_CLOUD_BASE_FEET = 18000.0
GOES_PRODUCT_PREFIX = "ABI-L2-CCLC"
GOES_LOOKBACK_HOURS = 24
GOES_EAST_BUCKETS = ("noaa-goes19", "noaa-goes16")
GOES_WEST_BUCKETS = ("noaa-goes18",)
GOES_HIGH_CLOUD_VARIABLES = ("CF4", "CF5")
MISSING_VALUE = math.nan
DEFAULT_DOWNLOAD_DIR = PROJECT_ROOT / "data_cache" / "noaa_gfs"
DEFAULT_OBS_DOWNLOAD_DIR = PROJECT_ROOT / "data_cache" / "nws_observations"
WEATHER_MODES = ("forecast", "observed")
WEATHER_KEYS = (
    "temp_250",
    "humidity_250",
    "cloud_cover_high",
    "precipitation",
)
WGRIB2_FALLBACKS = (
    "/home/celaeno/usr/bin/wgrib2",
    "/usr/local/bin/wgrib2",
    "/usr/bin/wgrib2",
)
LIQUID_PRECIP_CODES = (
    "DZ",
    "RA",
    "SHRA",
    "SHDZ",
    "TSRA",
    "TS",
    "VCSH",
    "FZRA",
    "FZDZ",
)
GOES_START_TIME_PATTERN = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})\d_")
METAR_REPORT_TIME_PATTERN = re.compile(r"\b(\d{2})(\d{2})(\d{2})Z\b")


@dataclass(frozen=True)
class GfsRequest:
    cycle_date: str
    cycle_hour: int
    forecast_hour: int


@dataclass(frozen=True)
class GribRecord:
    init_time: str
    valid_time: str
    variable: str
    level: str
    longitude: float
    latitude: float
    value: float


@dataclass(frozen=True)
class MetarObservation:
    station_identifier: str
    raw_record: dict[str, object]
    precipitation: float
    high_cloud_cover: float
    timestamp: str


@dataclass(frozen=True)
class GoesObservation:
    bucket: str
    key: str
    high_cloud_cover: float
    downloaded_path: Path | None
    timestamp: str


@dataclass(frozen=True)
class SourceAttribution:
    name: str
    timestamp: str


@dataclass(frozen=True)
class WeatherSnapshot:
    weather: dict[str, float]
    sources: tuple[SourceAttribution, ...]


class WeatherDataUnavailable(RuntimeError):
    """Raised when the required weather inputs cannot be loaded or parsed."""


def get_weather(
    lat: float,
    lon: float,
    mode: str = "forecast",
    keep_downloaded_files: bool = False,
    download_dir: str | Path | None = None,
) -> dict[str, float]:
    """Return a Phase 1 weather snapshot for the supplied coordinates.

    Returned values are normalized for the downstream prediction steps:
    - ``temp_250`` is converted to Celsius
    - ``humidity_250`` remains relative humidity percent
    - ``cloud_cover_high`` is normalized to a 0-1 fraction
    - ``precipitation`` remains millimeters over the selected forecast step
    """

    return get_weather_snapshot(
        lat,
        lon,
        mode=mode,
        keep_downloaded_files=keep_downloaded_files,
        download_dir=download_dir,
    ).weather


def get_weather_snapshot(
    lat: float,
    lon: float,
    mode: str = "forecast",
    keep_downloaded_files: bool = False,
    download_dir: str | Path | None = None,
) -> WeatherSnapshot:
    """Return the weather payload along with the source feeds that supplied it."""

    _validate_coordinates(lat, lon)
    normalized_mode = _normalize_mode(mode)

    if normalized_mode == "forecast":
        return _get_forecast_snapshot(
            lat,
            lon,
            keep_downloaded_files=keep_downloaded_files,
            download_dir=download_dir,
        )
    return _get_observed_snapshot(
        lat,
        lon,
        keep_downloaded_files=keep_downloaded_files,
        download_dir=download_dir,
    )


def _get_forecast_weather(
    lat: float,
    lon: float,
    keep_downloaded_files: bool = False,
    download_dir: str | Path | None = None,
) -> dict[str, float]:
    return _get_forecast_snapshot(
        lat,
        lon,
        keep_downloaded_files=keep_downloaded_files,
        download_dir=download_dir,
    ).weather


def _get_forecast_snapshot(
    lat: float,
    lon: float,
    keep_downloaded_files: bool = False,
    download_dir: str | Path | None = None,
) -> WeatherSnapshot:
    lat = round(lat, 4)
    lon = round(lon, 4)
    resolved_download_dir = _resolve_download_dir(
        keep_downloaded_files=keep_downloaded_files,
        download_dir=download_dir,
        default_dir=DEFAULT_DOWNLOAD_DIR,
    )
    failures: list[tuple[GfsRequest, str]] = []

    for request_info in _build_request_candidates(datetime.now(timezone.utc)):
        try:
            values = _fetch_weather_for_request(
                lat,
                lon,
                request_info.cycle_date,
                request_info.cycle_hour,
                request_info.forecast_hour,
                keep_downloaded_files=resolved_download_dir is not None,
                download_dir=resolved_download_dir,
            )
            weather = dict(zip(WEATHER_KEYS, values, strict=True))
            LOGGER.info(
                "Loaded NOAA GFS weather for lat=%s lon=%s using cycle %s %02dz f%03d",
                lat,
                lon,
                request_info.cycle_date,
                request_info.cycle_hour,
                request_info.forecast_hour,
            )
            return WeatherSnapshot(
                weather=weather,
                sources=(
                    SourceAttribution(
                        name="gfs",
                        timestamp=_format_gfs_source_timestamp(request_info),
                    ),
                ),
            )
        except WeatherDataUnavailable as exc:
            failures.append((request_info, str(exc)))
            LOGGER.debug(
                "Skipping NOAA GFS cycle for lat=%s lon=%s from cycle %s %02dz f%03d: %s",
                lat,
                lon,
                request_info.cycle_date,
                request_info.cycle_hour,
                request_info.forecast_hour,
                exc,
            )

    if failures:
        last_request, last_error = failures[-1]
        LOGGER.warning(
            "Unable to load NOAA GFS weather for lat=%s lon=%s after %d attempts; last tried cycle %s %02dz f%03d: %s",
            lat,
            lon,
            len(failures),
            last_request.cycle_date,
            last_request.cycle_hour,
            last_request.forecast_hour,
            last_error,
        )

    return WeatherSnapshot(weather=_missing_weather_payload(), sources=())


def _get_observed_weather(
    lat: float,
    lon: float,
    keep_downloaded_files: bool = False,
    download_dir: str | Path | None = None,
) -> dict[str, float]:
    return _get_observed_snapshot(
        lat,
        lon,
        keep_downloaded_files=keep_downloaded_files,
        download_dir=download_dir,
    ).weather


def _get_observed_snapshot(
    lat: float,
    lon: float,
    keep_downloaded_files: bool = False,
    download_dir: str | Path | None = None,
) -> WeatherSnapshot:
    lat = round(lat, 4)
    lon = round(lon, 4)
    resolved_download_dir = _resolve_download_dir(
        keep_downloaded_files=keep_downloaded_files,
        download_dir=download_dir,
        default_dir=DEFAULT_OBS_DOWNLOAD_DIR,
    )

    try:
        points_data = _request_nws_json(f"{NWS_API_BASE_URL}/points/{lat},{lon}")
        stations_url = points_data["properties"]["observationStations"]
        stations_data = _request_nws_json(stations_url)
    except (KeyError, TypeError, WeatherDataUnavailable, requests.RequestException, ValueError) as exc:
        LOGGER.warning(
            "Unable to resolve nearby observation stations for lat=%s lon=%s: %s",
            lat,
            lon,
            exc,
        )
        points_data = {}
        stations_data = {}

    station_feature: dict[str, object] = {"properties": {"stationIdentifier": "unknown"}}
    metar_observation = MetarObservation(
        station_identifier="unknown",
        raw_record={},
        precipitation=0.0,
        high_cloud_cover=MISSING_VALUE,
        timestamp="",
    )
    try:
        if stations_data:
            station_feature, metar_observation = _select_best_metar_station(stations_data)
    except (WeatherDataUnavailable, requests.RequestException, ValueError) as exc:
        LOGGER.warning(
            "Unable to load METAR observations for lat=%s lon=%s: %s",
            lat,
            lon,
            exc,
        )

    try:
        goes_observation = _get_goes_observation(
            lat,
            lon,
            download_dir=resolved_download_dir,
        )
    except (WeatherDataUnavailable, requests.RequestException, ValueError) as exc:
        LOGGER.warning(
            "Unable to load GOES observations for lat=%s lon=%s: %s",
            lat,
            lon,
            exc,
        )
        goes_observation = GoesObservation(
            bucket="",
            key="",
            high_cloud_cover=MISSING_VALUE,
            downloaded_path=None,
            timestamp="",
        )

    try:
        forecast_fallback = (
            _get_forecast_snapshot(lat, lon)
            if not math.isfinite(goes_observation.high_cloud_cover)
            else None
        )
        weather = _compose_observed_weather(
            metar_observation=metar_observation,
            goes_observation=goes_observation,
            forecast_fallback=forecast_fallback,
        )

        if resolved_download_dir is not None:
            _persist_observed_files(
                points_data=points_data,
                stations_data=stations_data,
                station_feature=station_feature,
                metar_observation=metar_observation,
                goes_observation=goes_observation,
                lat=lat,
                lon=lon,
                download_dir=resolved_download_dir,
            )

        station_identifier = metar_observation.station_identifier
        LOGGER.info(
            "Loaded observed weather for lat=%s lon=%s using METAR station %s and GOES bucket %s",
            lat,
            lon,
            station_identifier,
            goes_observation.bucket or "unavailable",
        )
        return WeatherSnapshot(
            weather=weather,
            sources=_collect_observed_sources(
                metar_observation=metar_observation,
                goes_observation=goes_observation,
                forecast_fallback=forecast_fallback,
            ),
        )
    except (KeyError, TypeError, WeatherDataUnavailable, requests.RequestException, ValueError) as exc:
        LOGGER.warning(
            "Unable to load observed weather for lat=%s lon=%s: %s",
            lat,
            lon,
            exc,
        )
        return WeatherSnapshot(weather=_missing_weather_payload(), sources=())


def _validate_coordinates(lat: float, lon: float) -> None:
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"Latitude must be between -90 and 90. Received {lat}.")
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"Longitude must be between -180 and 180. Received {lon}.")


def _build_request_candidates(now: datetime) -> list[GfsRequest]:
    utc_now = now.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    latest_cycle_hour = (utc_now.hour // 6) * 6
    latest_cycle = utc_now.replace(hour=latest_cycle_hour)

    requests_to_try: list[GfsRequest] = []
    for lookback in range(MAX_CYCLE_LOOKBACKS + 1):
        cycle_time = latest_cycle - timedelta(hours=6 * lookback)
        forecast_hour = int((utc_now - cycle_time).total_seconds() // 3600)
        if forecast_hour < 0:
            continue

        requests_to_try.append(
            GfsRequest(
                cycle_date=cycle_time.strftime("%Y%m%d"),
                cycle_hour=cycle_time.hour,
                forecast_hour=forecast_hour,
            )
        )

    return requests_to_try


def _normalize_mode(mode: str) -> str:
    normalized_mode = mode.strip().lower()
    if normalized_mode not in WEATHER_MODES:
        raise ValueError(
            f"Unsupported weather mode: {mode}. Expected one of {', '.join(WEATHER_MODES)}."
        )
    return normalized_mode


def _compose_observed_weather(
    metar_observation: MetarObservation,
    goes_observation: GoesObservation,
    forecast_fallback: WeatherSnapshot | None = None,
) -> dict[str, float]:
    weather = {
        "temp_250": MISSING_VALUE,
        "humidity_250": MISSING_VALUE,
        "cloud_cover_high": goes_observation.high_cloud_cover,
        "precipitation": metar_observation.precipitation,
    }

    if not math.isfinite(weather["cloud_cover_high"]):
        weather["cloud_cover_high"] = metar_observation.high_cloud_cover

    if forecast_fallback is not None:
        weather["temp_250"] = forecast_fallback.weather["temp_250"]
        weather["humidity_250"] = forecast_fallback.weather["humidity_250"]
        if not math.isfinite(weather["cloud_cover_high"]):
            weather["cloud_cover_high"] = forecast_fallback.weather["cloud_cover_high"]

    return weather


def _collect_observed_sources(
    metar_observation: MetarObservation,
    goes_observation: GoesObservation,
    forecast_fallback: WeatherSnapshot | None,
) -> tuple[SourceAttribution, ...]:
    sources: list[SourceAttribution] = []

    if math.isfinite(goes_observation.high_cloud_cover):
        sources.append(
            SourceAttribution(
                name=_goes_source_label(goes_observation.bucket),
                timestamp=goes_observation.timestamp,
            )
        )
    elif forecast_fallback is not None and _has_any_weather_data(forecast_fallback.weather):
        sources.extend(forecast_fallback.sources)

    if metar_observation.raw_record:
        sources.append(
            SourceAttribution(
                name="metar",
                timestamp=metar_observation.timestamp,
            )
        )

    return tuple(source for source in sources if source.name)


def _goes_source_label(bucket: str) -> str:
    if bucket in GOES_EAST_BUCKETS:
        return "goes-east"
    if bucket in GOES_WEST_BUCKETS:
        return "goes-west"
    return "goes"


def _has_any_weather_data(weather: dict[str, float]) -> bool:
    return any(math.isfinite(weather[key]) for key in WEATHER_KEYS)


def _format_gfs_source_timestamp(request_info: GfsRequest) -> str:
    return (
        f"{request_info.cycle_date} "
        f"{request_info.cycle_hour:02d}z "
        f"f{request_info.forecast_hour:03d}"
    )


def _format_compact_utc_timestamp(at_time: datetime, include_seconds: bool) -> str:
    normalized_time = at_time.astimezone(timezone.utc)
    if include_seconds:
        return normalized_time.strftime("%Y%m%d %H%M%Sz")
    return normalized_time.strftime("%Y%m%d %H%Mz")


def _extract_goes_timestamp(key: str) -> str:
    match = GOES_START_TIME_PATTERN.search(key)
    if not match:
        return ""

    year, julian_day, hour, minute, second = match.groups()
    goes_time = datetime.strptime(
        f"{year}{julian_day}{hour}{minute}{second}",
        "%Y%j%H%M%S",
    ).replace(tzinfo=timezone.utc)
    return _format_compact_utc_timestamp(goes_time, include_seconds=True)


def _extract_metar_timestamp(record: dict[str, object]) -> str:
    for field_name in ("obsTime", "observationTime", "reportTime", "receiptTime"):
        formatted = _normalize_timestamp_value(record.get(field_name), include_seconds=False)
        if formatted:
            return formatted

    raw_ob = str(record.get("rawOb", "")).upper()
    match = METAR_REPORT_TIME_PATTERN.search(raw_ob)
    if not match:
        return ""

    day, hour, minute = (int(part) for part in match.groups())
    return _format_compact_utc_timestamp(
        _resolve_metar_report_datetime(datetime.now(timezone.utc), day, hour, minute),
        include_seconds=False,
    )


def _normalize_timestamp_value(value: object, include_seconds: bool) -> str:
    if value in (None, ""):
        return ""

    if isinstance(value, (int, float)):
        epoch_value = float(value)
        if epoch_value > 1_000_000_000_000:
            epoch_value /= 1000.0
        return _format_compact_utc_timestamp(
            datetime.fromtimestamp(epoch_value, tz=timezone.utc),
            include_seconds=include_seconds,
        )

    if not isinstance(value, str):
        return ""

    stripped_value = value.strip()
    if not stripped_value:
        return ""

    if stripped_value.isdigit():
        return _normalize_timestamp_value(int(stripped_value), include_seconds=include_seconds)

    try:
        parsed_time = datetime.fromisoformat(stripped_value.replace("Z", "+00:00"))
    except ValueError:
        return ""

    if parsed_time.tzinfo is None:
        parsed_time = parsed_time.replace(tzinfo=timezone.utc)
    return _format_compact_utc_timestamp(parsed_time, include_seconds=include_seconds)


def _resolve_metar_report_datetime(now: datetime, day: int, hour: int, minute: int) -> datetime:
    candidates: list[datetime] = []
    for month_delta in (-1, 0, 1):
        year, month = _shift_year_month(now.year, now.month, month_delta)
        try:
            candidates.append(datetime(year, month, day, hour, minute, tzinfo=timezone.utc))
        except ValueError:
            continue

    if not candidates:
        raise ValueError(f"Invalid METAR report time: day={day} hour={hour} minute={minute}")

    return min(candidates, key=lambda candidate: abs(candidate - now))


def _shift_year_month(year: int, month: int, delta: int) -> tuple[int, int]:
    shifted_month = month + delta
    shifted_year = year

    while shifted_month < 1:
        shifted_month += 12
        shifted_year -= 1
    while shifted_month > 12:
        shifted_month -= 12
        shifted_year += 1

    return shifted_year, shifted_month


def _fetch_weather_for_request(
    lat: float,
    lon: float,
    cycle_date: str,
    cycle_hour: int,
    forecast_hour: int,
    keep_downloaded_files: bool = False,
    download_dir: Path | None = None,
) -> tuple[float, float, float, float]:
    request_info = GfsRequest(
        cycle_date=cycle_date,
        cycle_hour=cycle_hour,
        forecast_hour=forecast_hour,
    )
    records = _download_and_parse_records(
        request_info,
        lat,
        lon,
        keep_downloaded_files=keep_downloaded_files,
        download_dir=download_dir,
    )
    weather = _extract_weather_payload(records, lat, lon)

    if all(math.isnan(value) for value in weather.values()):
        raise WeatherDataUnavailable("Downloaded data did not contain the required fields.")

    return tuple(weather[key] for key in WEATHER_KEYS)


def _download_and_parse_records(
    request_info: GfsRequest,
    lat: float,
    lon: float,
    keep_downloaded_files: bool = False,
    download_dir: Path | None = None,
) -> list[GribRecord]:
    params = _build_nomads_params(request_info, lat, lon)
    content = _download_gfs_subset(params)
    return _convert_grib_to_records(
        content,
        request_info,
        lat,
        lon,
        keep_downloaded_files=keep_downloaded_files,
        download_dir=download_dir,
    )


def _build_nomads_params(request_info: GfsRequest, lat: float, lon: float) -> dict[str, str]:
    nomads_lon = lon % 360.0
    left_lon = max(0.0, nomads_lon - GRID_BUFFER_DEGREES)
    right_lon = min(359.75, nomads_lon + GRID_BUFFER_DEGREES)
    top_lat = min(90.0, lat + GRID_BUFFER_DEGREES)
    bottom_lat = max(-90.0, lat - GRID_BUFFER_DEGREES)

    return {
        "file": f"gfs.t{request_info.cycle_hour:02d}z.pgrb2.0p25.f{request_info.forecast_hour:03d}",
        "dir": f"/gfs.{request_info.cycle_date}/{request_info.cycle_hour:02d}/atmos",
        "subregion": "",
        "leftlon": f"{left_lon:.3f}",
        "rightlon": f"{right_lon:.3f}",
        "toplat": f"{top_lat:.3f}",
        "bottomlat": f"{bottom_lat:.3f}",
        "lev_250_mb": "on",
        "lev_high_cloud_layer": "on",
        "lev_surface": "on",
        "var_TMP": "on",
        "var_RH": "on",
        "var_HCDC": "on",
        "var_APCP": "on",
    }


def _download_gfs_subset(params: dict[str, str]) -> bytes:
    return _download_gfs_subset_cached(tuple(sorted(params.items())))


@lru_cache(maxsize=256)
def _download_gfs_subset_cached(params_key: tuple[tuple[str, str], ...]) -> bytes:
    params = dict(params_key)
    last_error: Exception | None = None

    for attempt in range(1, NETWORK_RETRIES + 1):
        try:
            response = requests.get(NOMADS_GFS_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()

            content = response.content
            if not content:
                raise WeatherDataUnavailable("NOAA returned an empty response.")
            if not content.startswith(b"GRIB"):
                preview = response.text[:120].strip().replace("\n", " ")
                raise WeatherDataUnavailable(f"NOAA returned a non-GRIB payload: {preview}")

            return content
        except (requests.RequestException, WeatherDataUnavailable) as exc:
            last_error = exc
            if attempt < NETWORK_RETRIES:
                time.sleep(0.5 * attempt)

    raise WeatherDataUnavailable(str(last_error) if last_error else "NOAA download failed.")


def _convert_grib_to_records(
    content: bytes,
    request_info: GfsRequest,
    lat: float,
    lon: float,
    keep_downloaded_files: bool = False,
    download_dir: Path | None = None,
) -> list[GribRecord]:
    wgrib2_path = _resolve_wgrib2()

    with tempfile.TemporaryDirectory() as temp_dir:
        grib_path = Path(temp_dir) / "subset.grib2"
        csv_path = Path(temp_dir) / "subset.csv"
        grib_path.write_bytes(content)

        completed = subprocess.run(
            [wgrib2_path, str(grib_path), "-csv", str(csv_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise WeatherDataUnavailable(
                completed.stderr.strip() or "wgrib2 failed to parse the GRIB file."
            )

        if keep_downloaded_files and download_dir is not None:
            _persist_downloaded_files(
                grib_path,
                csv_path,
                request_info,
                lat,
                lon,
                download_dir,
            )

        return _read_csv_records(csv_path)


def _resolve_wgrib2() -> str:
    candidates = (
        os.getenv("WGRIB2_BIN"),
        shutil.which("wgrib2"),
        *WGRIB2_FALLBACKS,
    )
    for candidate in candidates:
        if not candidate:
            continue

        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)

    raise WeatherDataUnavailable("wgrib2 is required to parse NOAA GFS GRIB2 responses.")


def _read_csv_records(csv_path: Path) -> list[GribRecord]:
    records: list[GribRecord] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) == 7:
                init_time, valid_time, variable, level, longitude, latitude, value = row
            elif len(row) == 6:
                init_time = ""
                valid_time, variable, level, longitude, latitude, value = row
            else:
                continue

            records.append(
                GribRecord(
                    init_time=init_time,
                    valid_time=valid_time,
                    variable=variable,
                    level=level,
                    longitude=float(longitude),
                    latitude=float(latitude),
                    value=float(value),
                )
            )

    if not records:
        raise WeatherDataUnavailable("wgrib2 did not emit any rows for the requested subset.")

    return records


def _resolve_download_dir(
    keep_downloaded_files: bool,
    download_dir: str | Path | None,
    default_dir: Path,
) -> Path | None:
    if download_dir is not None:
        return Path(download_dir)
    if keep_downloaded_files:
        return default_dir
    return None


def _persist_downloaded_files(
    grib_path: Path,
    csv_path: Path,
    request_info: GfsRequest,
    lat: float,
    lon: float,
    download_dir: Path,
) -> tuple[Path, Path]:
    download_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = _artifact_stem(request_info, lat, lon)

    persisted_grib_path = download_dir / f"{artifact_stem}.grib2"
    persisted_csv_path = download_dir / f"{artifact_stem}.csv"

    shutil.copy2(grib_path, persisted_grib_path)
    shutil.copy2(csv_path, persisted_csv_path)

    LOGGER.info(
        "Saved NOAA GFS artifacts to %s and %s",
        persisted_grib_path,
        persisted_csv_path,
    )

    return persisted_grib_path, persisted_csv_path


def _artifact_stem(request_info: GfsRequest, lat: float, lon: float) -> str:
    return (
        f"gfs_{request_info.cycle_date}_t{request_info.cycle_hour:02d}z"
        f"_f{request_info.forecast_hour:03d}_lat{_coord_token(lat)}_lon{_coord_token(lon)}"
    )


def _coord_token(value: float) -> str:
    return f"{value:+0.4f}".replace("+", "p").replace("-", "m").replace(".", "p")


def _request_nws_json(url: str) -> dict[str, object]:
    headers = {
        "Accept": "application/geo+json",
        "User-Agent": NWS_USER_AGENT,
    }
    last_error: Exception | None = None

    for attempt in range(1, NETWORK_RETRIES + 1):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < NETWORK_RETRIES:
                time.sleep(0.5 * attempt)

    raise WeatherDataUnavailable(str(last_error) if last_error else "NWS observation request failed.")


def _select_best_metar_station(
    stations_data: dict[str, object],
) -> tuple[dict[str, object], MetarObservation]:
    features = stations_data.get("features")
    if not isinstance(features, list) or not features:
        raise WeatherDataUnavailable("NWS did not return any observation stations.")

    best_choice: tuple[int, dict[str, object], MetarObservation] | None = None

    for feature in features[:MAX_OBS_STATIONS]:
        if not isinstance(feature, dict):
            continue

        properties = feature.get("properties")
        if not isinstance(properties, dict):
            continue

        station_identifier = str(properties.get("stationIdentifier", "")).strip().upper()
        if not station_identifier:
            continue

        metar_records = _request_awc_metar_records(station_identifier)
        if not metar_records:
            continue

        metar_observation = _build_metar_observation(station_identifier, metar_records[0])
        score = _score_metar_record(metar_observation.raw_record)
        if best_choice is None or score > best_choice[0]:
            best_choice = (score, feature, metar_observation)
        if score >= 3:
            break

    if best_choice is None:
        raise WeatherDataUnavailable("Unable to retrieve any nearby METAR observations.")

    return best_choice[1], best_choice[2]


def _request_awc_metar_records(station_identifier: str) -> list[dict[str, object]]:
    headers = {
        "Accept": "application/json",
        "User-Agent": NWS_USER_AGENT,
    }
    last_error: Exception | None = None

    for attempt in range(1, NETWORK_RETRIES + 1):
        try:
            response = requests.get(
                AWC_METAR_URL,
                params={"ids": station_identifier, "format": "json"},
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list):
                raise WeatherDataUnavailable("Aviation Weather Center returned a non-list METAR payload.")
            return [record for record in data if isinstance(record, dict)]
        except (requests.RequestException, ValueError, WeatherDataUnavailable) as exc:
            last_error = exc
            if attempt < NETWORK_RETRIES:
                time.sleep(0.5 * attempt)

    raise WeatherDataUnavailable(str(last_error) if last_error else "METAR request failed.")


def _score_metar_record(record: dict[str, object]) -> int:
    if not isinstance(record, dict):
        return 0

    score = 0
    if math.isfinite(_float_or_nan(record.get("temp"))):
        score += 1
    if record.get("rawOb"):
        score += 1
    if isinstance(record.get("clouds"), list):
        score += 1
    if _extract_metar_precipitation(record) > 0.0:
        score += 1

    return score


def _build_metar_observation(
    station_identifier: str,
    record: dict[str, object],
) -> MetarObservation:
    return MetarObservation(
        station_identifier=station_identifier,
        raw_record=record,
        precipitation=_extract_metar_precipitation(record),
        high_cloud_cover=_extract_metar_high_cloud_cover(record),
        timestamp=_extract_metar_timestamp(record),
    )


def _extract_metar_high_cloud_cover(record: dict[str, object]) -> float:
    clouds = record.get("clouds")
    if not isinstance(clouds, list):
        return MISSING_VALUE

    high_cloud_fractions: list[float] = []
    for layer in clouds:
        if not isinstance(layer, dict):
            continue
        base_feet = _float_or_nan(layer.get("base"))
        cover = str(layer.get("cover", "")).upper()
        if math.isfinite(base_feet) and base_feet >= METAR_HIGH_CLOUD_BASE_FEET:
            fraction = _cloud_amount_fraction(cover)
            if math.isfinite(fraction):
                high_cloud_fractions.append(fraction)

    if high_cloud_fractions:
        return max(high_cloud_fractions)
    if clouds:
        return 0.0
    return MISSING_VALUE


def _extract_metar_precipitation(record: dict[str, object]) -> float:
    raw_ob = str(record.get("rawOb", "")).upper()
    if not raw_ob:
        return 0.0

    tokens = raw_ob.split()
    for token in tokens:
        if any(code in token for code in LIQUID_PRECIP_CODES):
            return 1.0
    return 0.0


def _get_goes_observation(
    lat: float,
    lon: float,
    download_dir: Path | None = None,
) -> GoesObservation:
    bucket, key = _find_latest_goes_object(datetime.now(timezone.utc), lon)
    downloaded_path = _download_goes_object(bucket, key, download_dir)
    high_cloud_cover = _extract_goes_high_cloud_cover(downloaded_path, lat, lon)
    return GoesObservation(
        bucket=bucket,
        key=key,
        high_cloud_cover=high_cloud_cover,
        downloaded_path=downloaded_path,
        timestamp=_extract_goes_timestamp(key),
    )


def _find_latest_goes_object(now: datetime, lon: float) -> tuple[str, str]:
    for bucket in _preferred_goes_buckets(lon):
        for lookback in range(GOES_LOOKBACK_HOURS + 1):
            target_time = now - timedelta(hours=lookback)
            prefix = f"{GOES_PRODUCT_PREFIX}/{target_time:%Y/%j/%H}/"
            keys = _list_s3_keys(bucket, prefix)
            if keys:
                return bucket, keys[-1]

    raise WeatherDataUnavailable("Unable to find a recent GOES cloud-layer product.")


def _preferred_goes_buckets(lon: float) -> tuple[str, ...]:
    if lon >= -105.0:
        return GOES_EAST_BUCKETS + GOES_WEST_BUCKETS
    return GOES_WEST_BUCKETS + GOES_EAST_BUCKETS


@lru_cache(maxsize=512)
def _list_s3_keys(bucket: str, prefix: str) -> tuple[str, ...]:
    url = f"https://{bucket}.s3.amazonaws.com/"
    params = {
        "list-type": "2",
        "prefix": prefix,
        "max-keys": "1000",
    }
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    root = ET.fromstring(response.text)
    keys = tuple(
        element.text
        for element in root.findall("s3:Contents/s3:Key", namespace)
        if element.text
    )
    return keys


def _download_goes_object(bucket: str, key: str, download_dir: Path | None) -> Path:
    target_dir = download_dir or Path(tempfile.mkdtemp(prefix="goes-ccl-"))
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / Path(key).name

    if target_path.exists():
        return target_path

    url = f"https://{bucket}.s3.amazonaws.com/{key}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    target_path.write_bytes(response.content)
    return target_path


def _extract_goes_high_cloud_cover(goes_path: Path, lat: float, lon: float) -> float:
    try:
        from netCDF4 import Dataset
        import numpy as np
        from pyproj import Proj
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise WeatherDataUnavailable("GOES dependencies are not installed.") from exc

    with Dataset(goes_path) as dataset:
        projection = dataset.variables["goes_imager_projection"]
        perspective_height = float(projection.perspective_point_height)
        projector = Proj(
            proj="geos",
            h=perspective_height,
            lon_0=float(projection.longitude_of_projection_origin),
            sweep=projection.sweep_angle_axis,
            a=float(projection.semi_major_axis),
            b=float(projection.semi_minor_axis),
        )

        projected_x, projected_y = projector(lon, lat)
        scan_x = projected_x / perspective_height
        scan_y = projected_y / perspective_height

        x_values = dataset.variables["x"][:]
        y_values = dataset.variables["y"][:]
        if not (float(x_values.min()) <= scan_x <= float(x_values.max())):
            raise WeatherDataUnavailable("Location is outside the available GOES scan for the selected file.")
        if not (float(y_values.min()) <= scan_y <= float(y_values.max())):
            raise WeatherDataUnavailable("Location is outside the available GOES scan for the selected file.")

        x_index = int(np.argmin(np.abs(x_values - scan_x)))
        y_index = int(np.argmin(np.abs(y_values - scan_y)))

        layer_fractions: list[float] = []
        for variable_name in GOES_HIGH_CLOUD_VARIABLES:
            if variable_name not in dataset.variables:
                continue
            layer_value = dataset.variables[variable_name][
                max(0, y_index - 1): y_index + 2,
                max(0, x_index - 1): x_index + 2,
            ]
            if hasattr(layer_value, "filled"):
                layer_value = layer_value.filled(np.nan)
            layer_array = np.array(layer_value, dtype=float)
            finite_values = layer_array[np.isfinite(layer_array)]
            if finite_values.size:
                layer_fractions.append(float(finite_values.max()) / 100.0)

        if layer_fractions:
            return max(layer_fractions)

        total_cloud_fraction = dataset.variables.get("TCF")
        if total_cloud_fraction is not None:
            fallback_value = total_cloud_fraction[y_index, x_index]
            if hasattr(fallback_value, "filled"):
                fallback_value = fallback_value.filled(np.nan)
            fallback_float = float(fallback_value)
            if math.isfinite(fallback_float):
                return fallback_float

    return MISSING_VALUE


def _float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return MISSING_VALUE


def _cloud_amount_fraction(amount: str) -> float:
    return {
        "CLR": 0.0,
        "SKC": 0.0,
        "NCD": 0.0,
        "NSC": 0.0,
        "FEW": 0.125,
        "SCT": 0.375,
        "BKN": 0.75,
        "OVC": 1.0,
        "VV": 1.0,
    }.get(amount, MISSING_VALUE)


def _persist_observed_files(
    points_data: dict[str, object],
    stations_data: dict[str, object],
    station_feature: dict[str, object],
    metar_observation: MetarObservation,
    goes_observation: GoesObservation,
    lat: float,
    lon: float,
    download_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    download_dir.mkdir(parents=True, exist_ok=True)
    station_identifier = str(
        station_feature.get("properties", {}).get("stationIdentifier", "unknown")
    ).lower()
    artifact_stem = f"observed_{station_identifier}_lat{_coord_token(lat)}_lon{_coord_token(lon)}"

    points_path = download_dir / f"{artifact_stem}.points.json"
    stations_path = download_dir / f"{artifact_stem}.stations.json"
    metar_path = download_dir / f"{artifact_stem}.metar.json"
    goes_path = download_dir / f"{artifact_stem}.goes.nc"

    points_path.write_text(json.dumps(points_data, indent=2, sort_keys=True), encoding="utf-8")
    stations_path.write_text(json.dumps(stations_data, indent=2, sort_keys=True), encoding="utf-8")
    metar_path.write_text(
        json.dumps(metar_observation.raw_record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if goes_observation.downloaded_path is not None and goes_observation.downloaded_path.exists():
        shutil.copy2(goes_observation.downloaded_path, goes_path)

    LOGGER.info(
        "Saved observed weather artifacts to %s, %s, %s, and %s",
        points_path,
        stations_path,
        metar_path,
        goes_path,
    )

    return points_path, stations_path, metar_path, goes_path


def _extract_weather_payload(records: Iterable[GribRecord], lat: float, lon: float) -> dict[str, float]:
    records = list(records)
    selectors = {
        "temp_250": ("TMP", {"250 mb"}, _kelvin_to_celsius),
        "humidity_250": ("RH", {"250 mb"}, float),
        "cloud_cover_high": ("HCDC", {"high cloud layer"}, _percent_to_fraction),
        "precipitation": ("APCP", {"surface"}, float),
    }

    weather: dict[str, float] = {}
    for key, (variable, levels, converter) in selectors.items():
        record = _find_nearest_record(records, lat, lon, variable, levels)
        weather[key] = converter(record.value) if record else MISSING_VALUE

    return weather


def _find_nearest_record(
    records: Iterable[GribRecord],
    lat: float,
    lon: float,
    variable: str,
    levels: set[str],
) -> GribRecord | None:
    normalized_levels = {level.casefold() for level in levels}
    candidates = [
        record
        for record in records
        if record.variable == variable and record.level.casefold() in normalized_levels
    ]
    if not candidates:
        return None

    return min(
        candidates,
        key=lambda record: _distance(
            record.latitude,
            _normalize_longitude(record.longitude),
            lat,
            lon,
        ),
    )


def _distance(record_lat: float, record_lon: float, target_lat: float, target_lon: float) -> float:
    lon_delta = abs(record_lon - target_lon)
    lon_delta = min(lon_delta, 360.0 - lon_delta)
    lat_delta = record_lat - target_lat
    return (lat_delta * lat_delta) + (lon_delta * lon_delta)


def _normalize_longitude(longitude: float) -> float:
    return ((longitude + 180.0) % 360.0) - 180.0


def _kelvin_to_celsius(value: float) -> float:
    return value - 273.15


def _percent_to_fraction(value: float) -> float:
    return value / 100.0


def _missing_weather_payload() -> dict[str, float]:
    return {key: MISSING_VALUE for key in WEATHER_KEYS}
