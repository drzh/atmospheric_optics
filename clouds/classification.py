"""Public cloud classification entry points."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime

from clouds.fusion import fuse_cloud_layers
from clouds.goes import extract_goes_features
from clouds.metar import parse_metar
from clouds.types import CloudLayer, normalize_timestamp, payloads_for_layers

SCHEMA_VERSION = "1.0"


def classify_clouds(
    goes_data: Mapping[str, object] | None,
    metar_data: Mapping[str, object] | None,
    location: Mapping[str, object] | None = None,
    timestamp: datetime | str | None = None,
) -> list[CloudLayer]:
    """Return all WMO cloud layers detected from GOES and METAR observations."""

    del location, timestamp
    return fuse_cloud_layers(parse_metar(metar_data), extract_goes_features(goes_data))


def build_cloud_classification_payload(
    goes_data: Mapping[str, object] | None,
    metar_data: Mapping[str, object] | None,
    *,
    sources: Iterable[object] = (),
    location: Mapping[str, object] | None = None,
    timestamp: datetime | str | None = None,
) -> dict[str, object]:
    source_names = {_source_name(source) for source in sources}
    has_goes = any(name.startswith("goes") for name in source_names)
    has_metar = "metar" in source_names
    layers = classify_clouds(
        goes_data if has_goes else {},
        metar_data if has_metar else {},
        location=location,
        timestamp=timestamp,
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "sources": {
            "goes": has_goes,
            "metar": has_metar,
        },
        "cloud_layers": payloads_for_layers(layers),
    }
    timestamp_value = normalize_timestamp(timestamp)
    if timestamp_value:
        payload["timestamp"] = timestamp_value
    return payload


def _source_name(source: object) -> str:
    name = getattr(source, "name", source)
    return str(name).strip().lower()
