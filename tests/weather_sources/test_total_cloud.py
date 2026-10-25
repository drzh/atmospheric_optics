from __future__ import annotations

import math

import pytest

from weather_sources import snapshots as weather


def test_gfs_total_cloud_request_and_percent_conversion() -> None:
    params = weather._build_nomads_params(weather.GfsRequest("20261024", 18, 3), 32.847, -96.806)
    assert params["lev_entire_atmosphere"] == "on"
    assert params["var_TCDC"] == "on"
    records = [
        weather.GribRecord("2026102418", "2026102421", "TCDC", "entire atmosphere", 263.2, 32.8, 80.0),
        weather.GribRecord("2026102418", "2026102421", "TCDC", "entire atmosphere", 263.0, 32.8, 40.0),
    ]
    payload = weather._extract_weather_payload(records, 32.8, -96.8)
    assert payload["cloud_cover_total"] == pytest.approx(0.8)
    assert payload["total_cloud_cover_grid"] == pytest.approx([0.8, 0.4])
    assert math.isnan(payload["cloud_cover_high"])


def test_goes_total_cloud_fraction_uses_scaled_fraction_and_masks_missing(tmp_path, monkeypatch) -> None:
    netcdf = pytest.importorskip("netCDF4")
    np = pytest.importorskip("numpy")
    path = tmp_path / "clouds.nc"
    with netcdf.Dataset(path, "w") as dataset:
        dataset.createDimension("y", 2)
        dataset.createDimension("x", 2)
        variable = dataset.createVariable("TCF", "u2", ("y", "x"), fill_value=65535)
        variable.scale_factor = 0.0001
        variable.units = "1"
        variable[:] = np.ma.array([[0.0, 0.2], [1.0, 0.0]], mask=[[False, False], [False, True]])
    monkeypatch.setattr(weather, "_resolve_goes_window", lambda *args, **kwargs: (slice(None), slice(None)))
    cover, grid = weather._extract_goes_total_cloud_cover_sample(path, 32.847, -96.806)
    assert cover == pytest.approx(0.4)
    assert grid == pytest.approx((0.0, 0.2, 1.0))


def test_observed_total_cloud_priority_and_missing_fallback() -> None:
    metar = weather.MetarObservation("KDAL", {"rawOb": "METAR KDAL 250353Z 00000KT 10SM OVC010 20/10 A3000"}, 0.0, 0.0, "now")
    gfs = weather.WeatherSnapshot({"cloud_cover_total": 0.2, "total_cloud_cover_grid": [0.1, 0.2]}, ())
    goes = weather.GoesObservation("bucket", "key", 0.0, total_cloud_cover=0.0, total_cloud_cover_grid=(0.0, 0.1))
    assert weather._compose_observed_weather(metar, goes, gfs)["cloud_cover_total"] == 0.0
    missing_goes = weather.GoesObservation("bucket", "key", math.nan)
    assert weather._compose_observed_weather(metar, missing_goes, gfs)["cloud_cover_total"] == 1.0
    missing_metar = weather.MetarObservation("", {}, math.nan, math.nan, "")
    assert weather._compose_observed_weather(missing_metar, missing_goes, gfs)["cloud_cover_total"] == 0.2
    assert math.isnan(weather._compose_observed_weather(missing_metar, missing_goes)["cloud_cover_total"])
    assert math.isnan(weather._extract_metar_total_cloud_cover({}))


def test_total_cloud_data_is_attributed_to_goes_even_without_high_cloud() -> None:
    metar = weather.MetarObservation("", {}, math.nan, math.nan, "")
    goes = weather.GoesObservation("noaa-goes19", "key", math.nan, timestamp="now", total_cloud_cover=0.5)
    sources = weather._collect_observed_sources(metar, goes, None)
    assert any(source.name == "goes-east" for source in sources)
