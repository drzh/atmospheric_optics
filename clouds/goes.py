"""GOES feature extraction for cloud classification."""

from __future__ import annotations

import math
from collections.abc import Mapping

from clouds.types import GoesFeatures
from clouds.texture import classify_texture


def extract_goes_features(goes_data: Mapping[str, object] | None) -> GoesFeatures:
    if not isinstance(goes_data, Mapping):
        return GoesFeatures(available=False)

    has_explicit_goes = any(str(key).startswith("goes_") for key in goes_data)
    cloud_cover = _first_finite_fraction(goes_data.get("goes_cloud_cover"))
    if cloud_cover is None and not has_explicit_goes:
        cloud_cover = _first_finite_fraction(
            goes_data.get("cloud_cover_high"),
            goes_data.get("high_cloud_cover"),
        )
    cloud_optical_thickness = _first_finite_fraction(goes_data.get("goes_cloud_optical_thickness"))
    if cloud_optical_thickness is None and not has_explicit_goes:
        cloud_optical_thickness = _first_finite_fraction(goes_data.get("cloud_optical_thickness"))
    altitude_hint = _normalize_altitude_hint(
        goes_data.get("goes_altitude_category") or goes_data.get("cloud_altitude_category")
    )
    if altitude_hint is None and (
        "cloud_cover_high" in goes_data
        or "high_cloud_cover" in goes_data
        or goes_data.get("goes_cloud_cover_kind") == "high"
    ):
        altitude_hint = "high"
    texture = _normalize_text(goes_data.get("goes_texture") or goes_data.get("cloud_texture"))
    if texture is None:
        texture = classify_texture(
            _iterable_values(
                goes_data.get(
                    "goes_cloud_cover_grid",
                    () if has_explicit_goes else goes_data.get("cloud_cover_grid"),
                )
            ),
            optical_thickness=cloud_optical_thickness,
        )

    cloud_phase = _normalize_phase(goes_data.get("goes_cloud_phase") or goes_data.get("cloud_phase"))
    cloud_top_temp_k = _first_finite(
        goes_data.get("goes_cloud_top_temp_k"),
        goes_data.get("cloud_top_temp_k"),
    )
    cloud_top_m = _first_finite(
        goes_data.get("goes_cloud_top_m"),
        goes_data.get("cloud_top_m"),
        goes_data.get("cloud_top_height_m"),
    )
    available = any(
        value is not None
        for value in (
            cloud_phase,
            texture,
            cloud_top_temp_k,
            cloud_top_m,
            altitude_hint,
            cloud_cover,
            cloud_optical_thickness,
        )
    )
    return GoesFeatures(
        available=available,
        altitude_hint=altitude_hint,
        cloud_phase=cloud_phase,
        texture=texture,
        cloud_top_temp_k=cloud_top_temp_k,
        cloud_top_m=cloud_top_m,
        cloud_cover=cloud_cover,
        cloud_optical_thickness=cloud_optical_thickness,
    )


def goes_evidence_payload(features: GoesFeatures) -> dict[str, object]:
    evidence: dict[str, object] = {}
    if features.cloud_phase is not None:
        evidence["goes_cloud_phase"] = features.cloud_phase
    if features.altitude_hint is not None:
        evidence["goes_altitude_category"] = features.altitude_hint
    if features.texture is not None:
        evidence["goes_texture"] = features.texture
    if features.cloud_top_temp_k is not None:
        evidence["goes_cloud_top_temp_k"] = round(features.cloud_top_temp_k, 1)
    if features.cloud_top_m is not None:
        evidence["goes_cloud_top_m"] = round(features.cloud_top_m, 1)
    if features.cloud_cover is not None:
        evidence["goes_cloud_cover"] = round(features.cloud_cover, 3)
    if features.cloud_optical_thickness is not None:
        evidence["goes_cloud_optical_thickness"] = round(features.cloud_optical_thickness, 3)
    return evidence


def _first_finite(*values: object) -> float | None:
    for value in values:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_value):
            return numeric_value
    return None


def _first_finite_fraction(*values: object) -> float | None:
    value = _first_finite(*values)
    if value is None:
        return None
    if value > 1.0:
        value /= 100.0
    return max(0.0, min(1.0, value))


def _normalize_text(value: object) -> str | None:
    if value is None:
        return None
    normalized_value = str(value).strip().lower().replace("_", " ")
    if not normalized_value:
        return None
    if normalized_value in {"smooth layered", "layered"}:
        return "smooth"
    if normalized_value in {"towering", "deep convection", "deep convective"}:
        return "towering"
    if normalized_value in {"smooth", "cellular", "fibrous", "granular", "towering"}:
        return normalized_value
    return normalized_value


def _normalize_phase(value: object) -> str | None:
    normalized_value = _normalize_text(value)
    if normalized_value is None:
        return None
    if normalized_value in {"ice", "liquid", "mixed"}:
        return normalized_value
    if "ice" in normalized_value:
        return "ice"
    if "water" in normalized_value or "liquid" in normalized_value:
        return "liquid"
    if "mixed" in normalized_value:
        return "mixed"
    return normalized_value


def _normalize_altitude_hint(value: object) -> str | None:
    normalized_value = _normalize_text(value)
    if normalized_value in {"low", "middle", "high"}:
        return normalized_value
    if normalized_value == "mid":
        return "middle"
    return None


def _iterable_values(value: object) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or value is None:
        return ()
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return ()
