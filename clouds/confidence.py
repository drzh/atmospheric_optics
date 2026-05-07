"""Confidence scoring for deterministic cloud classifications."""

from __future__ import annotations

import math

from clouds.types import GoesFeatures, MetarCloudLayer, ParsedMetar


def confidence_for_layer(
    *,
    code: str,
    metar_layer: MetarCloudLayer | None,
    goes: GoesFeatures,
    metar: ParsedMetar,
) -> float:
    score = 0.35
    if metar_layer is not None:
        score += 0.22
        if metar_layer.base_m is not None:
            score += 0.1
        if metar_layer.coverage in {"broken", "overcast"}:
            score += 0.04
    if goes.available:
        score += 0.14
        if goes.cloud_phase is not None:
            score += 0.08
        if goes.cloud_top_temp_k is not None or goes.cloud_top_m is not None:
            score += 0.08
        if goes.texture is not None:
            score += 0.07
    if code == "Cb":
        if metar.thunderstorm or (metar_layer is not None and metar_layer.convective_type == "CB"):
            score += 0.12
        if goes.texture == "towering":
            score += 0.09
    elif code == "Ns" and metar.precipitation:
        score += 0.1
    elif code in {"Ci", "Cs", "Cc"} and goes.cloud_phase == "ice":
        score += 0.08
    elif code in {"St", "Sc", "Cu"} and goes.cloud_phase in {"liquid", "mixed"}:
        score += 0.05

    return _clamp(score)


def _clamp(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
