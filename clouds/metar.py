"""METAR parsing helpers for WMO cloud classification."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

from clouds.constants import (
    CONVECTIVE_WEATHER_CODES,
    COVERAGE_BY_METAR_CODE,
    FEET_TO_METERS,
    PRECIPITATION_WEATHER_CODES,
)
from clouds.types import MetarCloudLayer, ParsedMetar

METAR_CLOUD_TOKEN = re.compile(r"\b(FEW|SCT|BKN|OVC|VV)(\d{3}|///)?(CB|TCU)?\b")
WX_TOKEN = re.compile(r"^[+-]?(VC)?[A-Z]{2,}$")


def parse_metar(metar_data: Mapping[str, object] | None) -> ParsedMetar:
    if not isinstance(metar_data, Mapping):
        return ParsedMetar("", "", (), (), False, False, False)

    raw_record = _raw_record(metar_data)
    raw_observation = str(raw_record.get("rawOb") or metar_data.get("rawOb") or "").strip().upper()
    weather_codes = _extract_weather_codes(raw_record, raw_observation)
    layers = _extract_structured_layers(raw_record)
    if not layers:
        layers = _extract_raw_layers(raw_observation)

    precipitation = _has_any_code(weather_codes, PRECIPITATION_WEATHER_CODES) or _truthy_fraction(
        metar_data.get("precipitation")
    )
    thunderstorm = _has_any_code(weather_codes, CONVECTIVE_WEATHER_CODES) or any(code.startswith("TS") for code in weather_codes)
    fog = "FG" in weather_codes or _truthy_fraction(metar_data.get("fog_presence"))
    return ParsedMetar(
        station_identifier=str(
            raw_record.get("stationId")
            or raw_record.get("station_id")
            or raw_record.get("icaoId")
            or metar_data.get("metar_station")
            or ""
        ).upper(),
        raw_observation=raw_observation,
        layers=tuple(layers),
        weather_codes=tuple(weather_codes),
        precipitation=precipitation,
        thunderstorm=thunderstorm,
        fog=fog,
        timestamp=str(metar_data.get("metar_timestamp") or raw_record.get("obsTime") or ""),
    )


def metar_layer_evidence(layer: MetarCloudLayer) -> dict[str, object]:
    return {"metar_layers": [layer.raw]}


def _raw_record(metar_data: Mapping[str, object]) -> Mapping[str, object]:
    candidate = metar_data.get("metar_raw") or metar_data.get("raw_record")
    if isinstance(candidate, Mapping):
        return candidate
    if "rawOb" in metar_data or "clouds" in metar_data:
        return metar_data
    return {}


def _extract_structured_layers(record: Mapping[str, object]) -> list[MetarCloudLayer]:
    clouds = record.get("clouds")
    if not isinstance(clouds, list):
        return []

    layers: list[MetarCloudLayer] = []
    for cloud in clouds:
        if not isinstance(cloud, Mapping):
            continue
        cover_code = str(cloud.get("cover", "")).strip().upper()
        if cover_code not in COVERAGE_BY_METAR_CODE:
            continue
        base_ft = _finite_or_none(cloud.get("base"))
        base_m = base_ft * FEET_TO_METERS if base_ft is not None else None
        convective_type = _normalize_convective_type(cloud.get("type") or cloud.get("cloudType"))
        raw = _raw_layer_token(cover_code, base_ft, convective_type)
        layers.append(
            MetarCloudLayer(
                cover_code=cover_code,
                coverage=COVERAGE_BY_METAR_CODE[cover_code],
                base_m=base_m,
                raw=raw,
                convective_type=convective_type,
            )
        )
    return layers


def _extract_raw_layers(raw_observation: str) -> list[MetarCloudLayer]:
    layers: list[MetarCloudLayer] = []
    for match in METAR_CLOUD_TOKEN.finditer(raw_observation):
        cover_code, base_code, convective_type = match.groups()
        base_m = None
        if base_code and base_code != "///":
            base_m = int(base_code) * 100.0 * FEET_TO_METERS
        layers.append(
            MetarCloudLayer(
                cover_code=cover_code,
                coverage=COVERAGE_BY_METAR_CODE[cover_code],
                base_m=base_m,
                raw=match.group(0),
                convective_type=_normalize_convective_type(convective_type),
            )
        )
    return layers


def _extract_weather_codes(record: Mapping[str, object], raw_observation: str) -> list[str]:
    weather_text = str(record.get("wxString") or record.get("weather") or "").strip().upper()
    tokens = [token.strip() for token in f"{weather_text} {raw_observation}".split() if token.strip()]
    weather_codes: list[str] = []
    for token in tokens:
        normalized_token = token.strip().upper()
        if not WX_TOKEN.match(normalized_token):
            continue
        if normalized_token in {"METAR", "SPECI", "AUTO", "COR", "KT", "RMK"}:
            continue
        normalized_token = normalized_token.lstrip("+-")
        for code in (*PRECIPITATION_WEATHER_CODES, *CONVECTIVE_WEATHER_CODES, "FG", "BR"):
            if code in normalized_token and code not in weather_codes:
                weather_codes.append(code)
    return weather_codes


def _has_any_code(weather_codes: tuple[str, ...] | list[str], candidates: tuple[str, ...]) -> bool:
    return any(candidate in weather_codes for candidate in candidates)


def _truthy_fraction(value: object) -> bool:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric_value) and numeric_value > 0.0


def _finite_or_none(value: object) -> float | None:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _normalize_convective_type(value: object) -> str | None:
    if value is None:
        return None
    normalized_value = str(value).strip().upper()
    if normalized_value in {"CB", "TCU"}:
        return normalized_value
    if "CUMULONIMBUS" in normalized_value:
        return "CB"
    if "TOWERING" in normalized_value:
        return "TCU"
    return None


def _raw_layer_token(cover_code: str, base_ft: float | None, convective_type: str | None) -> str:
    if base_ft is None:
        base_token = "///"
    else:
        base_token = f"{int(round(base_ft / 100.0)):03d}"
    return f"{cover_code}{base_token}{convective_type or ''}"
