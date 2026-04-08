from __future__ import annotations

from datetime import datetime, timezone

import pytest

from solar.solar_position import get_solar_position


def test_get_solar_position_returns_valid_ranges() -> None:
    result = get_solar_position(32.0, -96.0, datetime(2026, 6, 21, 18, 0, tzinfo=timezone.utc))

    assert -90.0 <= result["elevation"] <= 90.0
    assert 0.0 <= result["azimuth"] <= 360.0


def test_get_solar_position_accepts_naive_datetimes_as_utc() -> None:
    aware = get_solar_position(32.0, -96.0, datetime(2026, 6, 21, 18, 0, tzinfo=timezone.utc))
    naive = get_solar_position(32.0, -96.0, datetime(2026, 6, 21, 18, 0))

    assert naive == pytest.approx(aware)


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        (-91.0, 0.0),
        (91.0, 0.0),
        (0.0, -181.0),
        (0.0, 181.0),
    ],
)
def test_get_solar_position_rejects_invalid_coordinates(lat: float, lon: float) -> None:
    with pytest.raises(ValueError):
        get_solar_position(lat, lon, datetime(2026, 6, 21, 18, 0, tzinfo=timezone.utc))
