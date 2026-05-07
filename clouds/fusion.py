"""Fusion of METAR layers and GOES morphology into WMO cloud layers."""

from __future__ import annotations

from clouds.confidence import confidence_for_layer
from clouds.constants import WMO_NAME_BY_CODE
from clouds.goes import goes_evidence_payload
from clouds.metar import metar_layer_evidence
from clouds.rules import (
    altitude_category_for_base,
    altitude_category_for_goes,
    classify_goes_layer,
    classify_metar_layer,
    estimated_top_for_layer,
)
from clouds.types import CloudLayer, GoesFeatures, ParsedMetar


def fuse_cloud_layers(metar: ParsedMetar, goes: GoesFeatures) -> list[CloudLayer]:
    layers: list[CloudLayer] = []
    goes_evidence = goes_evidence_payload(goes)

    for metar_layer in metar.layers:
        code = classify_metar_layer(metar_layer, metar, goes)
        top_m = estimated_top_for_layer(metar_layer, code, goes)
        layers.append(
            CloudLayer(
                layer_id=len(layers) + 1,
                genus=WMO_NAME_BY_CODE[code],
                code=code,
                altitude_category=altitude_category_for_base(metar_layer.base_m, top_m),
                coverage=metar_layer.coverage,
                base_m=metar_layer.base_m,
                top_m=top_m,
                confidence=confidence_for_layer(
                    code=code,
                    metar_layer=metar_layer,
                    goes=goes,
                    metar=metar,
                ),
                metar_evidence=metar_layer_evidence(metar_layer),
                goes_evidence=goes_evidence,
            )
        )

    if _should_add_goes_layer(layers, goes):
        code = classify_goes_layer(goes, metar)
        coverage = _coverage_name(goes.cloud_cover)
        layers.append(
            CloudLayer(
                layer_id=len(layers) + 1,
                genus=WMO_NAME_BY_CODE[code],
                code=code,
                altitude_category=altitude_category_for_goes(goes),
                coverage=coverage,
                base_m=_estimated_goes_base_m(code, goes),
                top_m=goes.cloud_top_m,
                confidence=confidence_for_layer(
                    code=code,
                    metar_layer=None,
                    goes=goes,
                    metar=metar,
                ),
                metar_evidence={},
                goes_evidence=goes_evidence,
            )
        )

    return layers


def _should_add_goes_layer(layers: list[CloudLayer], goes: GoesFeatures) -> bool:
    if not goes.available:
        return False
    if goes.cloud_cover is not None and goes.cloud_cover <= 0.02:
        return False
    if any(layer.code == "Cb" for layer in layers):
        return False
    if not layers:
        return True

    goes_category = altitude_category_for_goes(goes)
    if goes_category == "unknown":
        return False
    return all(layer.altitude_category != goes_category for layer in layers)


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


def _estimated_goes_base_m(code: str, goes: GoesFeatures) -> float | None:
    if goes.cloud_top_m is None:
        return None
    depth = {
        "Cb": 8_000.0,
        "Ns": 4_000.0,
        "Ci": 2_000.0,
        "Cs": 2_500.0,
        "Cc": 1_500.0,
        "As": 2_500.0,
        "Ac": 1_500.0,
        "Cu": 1_800.0,
        "Sc": 1_000.0,
        "St": 700.0,
    }[code]
    return max(0.0, goes.cloud_top_m - depth)
