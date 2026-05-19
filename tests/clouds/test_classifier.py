from __future__ import annotations

from datetime import datetime, timezone

from clouds.classification import build_cloud_classification_payload, classify_clouds


def test_classify_clouds_preserves_multiple_metar_layers() -> None:
    layers = classify_clouds(
        {
            "goes_cloud_phase": "ice",
            "goes_texture": "smooth",
            "goes_cloud_top_m": 10_000.0,
            "goes_cloud_top_temp_k": 230.0,
            "goes_cloud_cover": 0.9,
            "goes_altitude_category": "high",
        },
        {
            "metar_raw": {
                "rawOb": "METAR KDAL 071653Z 15009KT 10SM SCT010 BKN080 OVC250 19/02 A3019",
            },
        },
        {"lat": 32.8, "lon": -96.8},
        datetime(2026, 4, 7, 18, 0, tzinfo=timezone.utc),
    )

    assert [layer.code for layer in layers] == ["St", "As", "Cs"]
    assert [layer.altitude_category for layer in layers] == ["low", "middle", "high"]
    assert [layer.coverage for layer in layers] == ["scattered", "broken", "overcast"]


def test_classify_clouds_identifies_cumulonimbus_from_metar_and_goes() -> None:
    layers = classify_clouds(
        {
            "goes_cloud_phase": "mixed",
            "goes_texture": "towering",
            "goes_cloud_top_m": 11_500.0,
            "goes_cloud_top_temp_k": 218.0,
            "goes_cloud_cover": 0.75,
        },
        {
            "metar_raw": {
                "rawOb": "METAR KDAL 071653Z 15009KT 3SM TSRA BKN030CB 19/18 A3019",
            },
        },
    )

    assert len(layers) == 1
    assert layers[0].code == "Cb"
    assert layers[0].genus == "Cumulonimbus"
    assert layers[0].confidence >= 0.9


def test_classify_clouds_identifies_cumulus_from_low_scattered_cellular_layer() -> None:
    layers = classify_clouds(
        {
            "goes_cloud_phase": "liquid",
            "goes_texture": "cellular",
            "goes_cloud_cover": 0.35,
            "goes_altitude_category": "low",
        },
        {
            "metar_raw": {
                "rawOb": "METAR KDAL 071653Z 15009KT 10SM SCT025 24/14 A3019",
            },
        },
    )

    assert len(layers) == 1
    assert layers[0].code == "Cu"
    assert layers[0].genus == "Cumulus"
    assert layers[0].altitude_category == "low"


def test_classify_clouds_keeps_broken_low_cellular_deck_as_stratocumulus() -> None:
    layers = classify_clouds(
        {
            "goes_cloud_phase": "liquid",
            "goes_texture": "cellular",
            "goes_cloud_cover": 0.75,
            "goes_altitude_category": "low",
            "goes_cloud_top_m": 1_500.0,
        },
        {},
    )

    assert len(layers) == 1
    assert layers[0].code == "Sc"
    assert layers[0].genus == "Stratocumulus"


def test_build_cloud_classification_payload_uses_source_gates() -> None:
    payload = build_cloud_classification_payload(
        {"cloud_cover_high": 0.8},
        {
            "metar_raw": {
                "rawOb": "METAR KDAL 071653Z 15009KT 10SM BKN020 19/02 A3019",
            }
        },
        sources=("gfs",),
        timestamp=datetime(2026, 4, 7, 18, 0, tzinfo=timezone.utc),
    )

    assert payload == {
        "schema_version": "1.0",
        "sources": {"goes": False, "metar": False},
        "cloud_layers": [],
        "timestamp": "2026-04-07T18:00:00Z",
    }
