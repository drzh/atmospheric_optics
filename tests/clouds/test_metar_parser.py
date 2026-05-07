from __future__ import annotations

import pytest

from clouds.metar import parse_metar


def test_parse_metar_uses_structured_cloud_layers() -> None:
    parsed = parse_metar(
        {
            "metar_station": "KDAL",
            "metar_raw": {
                "rawOb": "METAR KDAL 071653Z 15009KT 10SM SCT025 BKN250 19/02 A3019",
                "clouds": [
                    {"cover": "SCT", "base": 2500},
                    {"cover": "BKN", "base": 25000},
                ],
            },
        }
    )

    assert parsed.station_identifier == "KDAL"
    assert [layer.raw for layer in parsed.layers] == ["SCT025", "BKN250"]
    assert parsed.layers[0].coverage == "scattered"
    assert parsed.layers[0].base_m == pytest.approx(762.0)
    assert parsed.layers[1].base_m == pytest.approx(7620.0)


def test_parse_metar_falls_back_to_raw_cloud_and_weather_tokens() -> None:
    parsed = parse_metar(
        {
            "metar_raw": {
                "rawOb": "METAR KDAL 071653Z 15009KT 3SM TSRA BKN030CB OVC080 19/18 A3019",
            },
        }
    )

    assert [layer.raw for layer in parsed.layers] == ["BKN030CB", "OVC080"]
    assert parsed.layers[0].convective_type == "CB"
    assert parsed.precipitation is True
    assert parsed.thunderstorm is True
