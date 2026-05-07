"""Data structures for WMO cloud classification."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping


@dataclass(frozen=True)
class MetarCloudLayer:
    cover_code: str
    coverage: str
    base_m: float | None
    raw: str
    convective_type: str | None = None


@dataclass(frozen=True)
class ParsedMetar:
    station_identifier: str
    raw_observation: str
    layers: tuple[MetarCloudLayer, ...]
    weather_codes: tuple[str, ...]
    precipitation: bool
    thunderstorm: bool
    fog: bool
    timestamp: str = ""


@dataclass(frozen=True)
class GoesFeatures:
    available: bool
    altitude_hint: str | None = None
    cloud_phase: str | None = None
    texture: str | None = None
    cloud_top_temp_k: float | None = None
    cloud_top_m: float | None = None
    cloud_cover: float | None = None
    cloud_optical_thickness: float | None = None


@dataclass
class CloudLayer:
    layer_id: int
    genus: str
    code: str
    altitude_category: str
    coverage: str
    base_m: float | None
    top_m: float | None
    confidence: float
    metar_evidence: dict[str, object] = field(default_factory=dict)
    goes_evidence: dict[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "layer_id": self.layer_id,
            "wmo_genus": self.genus,
            "wmo_code": self.code,
            "altitude_category": self.altitude_category,
            "coverage": self.coverage,
            "cloud_base_m": _round_optional(self.base_m),
            "cloud_top_m": _round_optional(self.top_m),
            "confidence": _round_unit(self.confidence),
            "evidence": {},
        }
        evidence: dict[str, object] = {}
        if self.metar_evidence:
            evidence.update(self.metar_evidence)
        if self.goes_evidence:
            evidence.update(self.goes_evidence)
        payload["evidence"] = evidence
        return payload


def _round_optional(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, 1)


def _round_unit(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(max(0.0, min(1.0, value)), 3)


def payloads_for_layers(layers: tuple[CloudLayer, ...] | list[CloudLayer]) -> list[dict[str, object]]:
    return [layer.to_payload() for layer in layers]


def normalize_timestamp(timestamp: datetime | str | None) -> str:
    if isinstance(timestamp, datetime):
        return timestamp.isoformat().replace("+00:00", "Z")
    if timestamp is None:
        return ""
    return str(timestamp)
