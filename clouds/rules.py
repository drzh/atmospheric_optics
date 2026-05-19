"""Deterministic WMO cloud genus rules."""

from __future__ import annotations

from clouds.constants import HIGH_CLOUD_BASE_M, LOW_CLOUD_TOP_M, MID_CLOUD_TOP_M
from clouds.types import GoesFeatures, MetarCloudLayer, ParsedMetar


def altitude_category_for_base(base_m: float | None, top_m: float | None = None) -> str:
    reference_height = base_m if base_m is not None else top_m
    if reference_height is None:
        return "unknown"
    if reference_height < LOW_CLOUD_TOP_M:
        return "low"
    if reference_height < HIGH_CLOUD_BASE_M:
        return "middle"
    return "high"


def altitude_category_for_goes(goes: GoesFeatures) -> str:
    if goes.altitude_hint in {"low", "middle", "high"}:
        return goes.altitude_hint
    if goes.cloud_top_m is not None:
        if goes.cloud_top_m < LOW_CLOUD_TOP_M:
            return "low"
        if goes.cloud_top_m < HIGH_CLOUD_BASE_M:
            return "middle"
        return "high"
    if goes.cloud_top_temp_k is not None:
        if goes.cloud_top_temp_k <= 235.0:
            return "high"
        if goes.cloud_top_temp_k <= 265.0:
            return "middle"
        return "low"
    if goes.cloud_phase == "ice":
        return "high"
    return "unknown"


def classify_metar_layer(layer: MetarCloudLayer, metar: ParsedMetar, goes: GoesFeatures) -> str:
    category = altitude_category_for_base(layer.base_m, goes.cloud_top_m)
    texture = goes.texture
    phase = goes.cloud_phase
    if _is_cumulonimbus(layer, metar, goes):
        return "Cb"
    if metar.precipitation and layer.coverage in {"broken", "overcast"} and category in {"low", "middle"}:
        return "Ns"
    if layer.convective_type == "TCU" or texture == "towering":
        return "Cu"
    if category == "low" and _is_detached_cellular_cloud(layer.coverage, texture):
        return "Cu"
    if category == "high" or (category == "unknown" and phase == "ice"):
        return classify_high_cloud(goes, coverage=layer.coverage)
    if category == "middle":
        return "Ac" if texture in {"cellular", "granular"} else "As"
    if texture in {"cellular", "granular"}:
        return "Sc"
    return "St"


def classify_goes_layer(goes: GoesFeatures, metar: ParsedMetar) -> str:
    if _goes_deep_convection(goes) and metar.thunderstorm:
        return "Cb"
    if metar.precipitation and goes.texture == "smooth":
        return "Ns"
    category = altitude_category_for_goes(goes)
    coverage = _coverage_name(goes.cloud_cover)
    if category == "high" or goes.cloud_phase == "ice":
        return classify_high_cloud(goes, coverage=coverage)
    if category == "middle":
        return "Ac" if goes.texture in {"cellular", "granular"} else "As"
    if goes.texture == "towering":
        return "Cu"
    if category == "low" and _is_detached_cellular_cloud(coverage, goes.texture):
        return "Cu"
    if goes.texture in {"cellular", "granular"}:
        return "Sc"
    return "St"


def classify_high_cloud(goes: GoesFeatures, *, coverage: str) -> str:
    if goes.texture == "granular":
        return "Cc"
    if goes.texture == "fibrous":
        return "Ci"
    if coverage in {"broken", "overcast"} or goes.texture == "smooth":
        return "Cs"
    return "Ci"


def estimated_top_for_layer(layer: MetarCloudLayer, code: str, goes: GoesFeatures) -> float | None:
    layer_category = altitude_category_for_base(layer.base_m)
    goes_category = altitude_category_for_goes(goes)
    if goes.cloud_top_m is not None and layer_category in {goes_category, "unknown"}:
        return goes.cloud_top_m
    if layer.base_m is None:
        return None
    if code == "Cb":
        return max(layer.base_m + 7_000.0, 10_000.0)
    if code == "Ns":
        return layer.base_m + 3_500.0
    if code in {"Ci", "Cs", "Cc"}:
        return layer.base_m + 2_000.0
    if code in {"As", "Ac"}:
        return min(layer.base_m + 2_500.0, MID_CLOUD_TOP_M)
    if code == "Cu":
        return layer.base_m + 1_800.0
    return layer.base_m + 900.0


def _is_cumulonimbus(layer: MetarCloudLayer, metar: ParsedMetar, goes: GoesFeatures) -> bool:
    if layer.convective_type == "CB":
        return True
    return metar.thunderstorm and _goes_deep_convection(goes)


def _goes_deep_convection(goes: GoesFeatures) -> bool:
    if goes.texture == "towering":
        return True
    if goes.cloud_top_m is not None and goes.cloud_top_m >= 9_000.0:
        return True
    return goes.cloud_top_temp_k is not None and goes.cloud_top_temp_k <= 225.0


def _is_detached_cellular_cloud(coverage: str, texture: str | None) -> bool:
    return coverage in {"few", "scattered"} and texture in {"cellular", "granular"}


def _coverage_name(cloud_cover: float | None) -> str:
    if cloud_cover is None:
        return "unknown"
    if cloud_cover >= 0.875:
        return "overcast"
    if cloud_cover >= 0.625:
        return "broken"
    if cloud_cover >= 0.25:
        return "scattered"
    if cloud_cover > 0.0:
        return "few"
    return "clear"
