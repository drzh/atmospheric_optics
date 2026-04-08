from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from data_ingestion import weather


def test_build_request_candidates_tracks_recent_cycles() -> None:
    candidates = weather._build_request_candidates(datetime(2026, 4, 7, 11, 42, tzinfo=timezone.utc))

    assert candidates[:3] == [
        weather.GfsRequest(cycle_date="20260407", cycle_hour=6, forecast_hour=5),
        weather.GfsRequest(cycle_date="20260407", cycle_hour=0, forecast_hour=11),
        weather.GfsRequest(cycle_date="20260406", cycle_hour=18, forecast_hour=17),
    ]


def test_build_nomads_params_requests_phase_one_fields() -> None:
    params = weather._build_nomads_params(
        weather.GfsRequest(cycle_date="20260407", cycle_hour=6, forecast_hour=5),
        lat=32.8,
        lon=-96.8,
    )

    assert params["file"] == "gfs.t06z.pgrb2.0p25.f005"
    assert params["dir"] == "/gfs.20260407/06/atmos"
    assert params["lev_250_mb"] == "on"
    assert params["lev_high_cloud_layer"] == "on"
    assert params["lev_surface"] == "on"
    assert params["var_TMP"] == "on"
    assert params["var_RH"] == "on"
    assert params["var_HCDC"] == "on"
    assert params["var_APCP"] == "on"


def test_extract_weather_payload_applies_unit_conversions() -> None:
    records = [
        weather.GribRecord("2026040700", "2026040706", "TMP", "250 mb", 263.25, 32.75, 243.15),
        weather.GribRecord("2026040700", "2026040706", "RH", "250 mb", 263.25, 32.75, 72.0),
        weather.GribRecord("2026040700", "2026040706", "HCDC", "high cloud layer", 263.25, 32.75, 85.0),
        weather.GribRecord("2026040700", "2026040706", "APCP", "surface", 263.25, 32.75, 1.8),
    ]

    payload = weather._extract_weather_payload(records, lat=32.8, lon=-96.8)

    assert payload == {
        "temp_250": pytest.approx(-30.0),
        "humidity_250": pytest.approx(72.0),
        "cloud_cover_high": pytest.approx(0.85),
        "precipitation": pytest.approx(1.8),
    }


def test_get_weather_returns_missing_payload_when_no_cycle_succeeds() -> None:
    candidates = [weather.GfsRequest(cycle_date="20260407", cycle_hour=6, forecast_hour=5)]

    with patch.object(weather, "_build_request_candidates", return_value=candidates):
        with patch.object(weather, "_fetch_weather_for_request", side_effect=weather.WeatherDataUnavailable("offline")):
            with patch.object(weather.LOGGER, "warning") as warning_mock:
                payload = weather.get_weather(32.8, -96.8)

    assert all(math.isnan(payload[key]) for key in weather.WEATHER_KEYS)
    warning_mock.assert_called_once()


def test_get_weather_silently_falls_back_to_older_gfs_cycle() -> None:
    candidates = [
        weather.GfsRequest(cycle_date="20260407", cycle_hour=18, forecast_hour=0),
        weather.GfsRequest(cycle_date="20260407", cycle_hour=12, forecast_hour=6),
    ]

    with patch.object(weather, "_build_request_candidates", return_value=candidates):
        with patch.object(
            weather,
            "_fetch_weather_for_request",
            side_effect=[weather.WeatherDataUnavailable("404 cycle not ready"), (-30.0, 72.0, 0.85, 1.8)],
        ):
            with patch.object(weather.LOGGER, "warning") as warning_mock:
                snapshot = weather.get_weather_snapshot(32.8, -96.8, mode="forecast")

    warning_mock.assert_not_called()
    assert snapshot.sources == (
        weather.SourceAttribution(name="gfs", timestamp="20260407 12z f006"),
    )
    assert snapshot.weather["temp_250"] == pytest.approx(-30.0)


def test_get_weather_snapshot_reports_gfs_source_when_forecast_succeeds() -> None:
    candidates = [weather.GfsRequest(cycle_date="20260407", cycle_hour=6, forecast_hour=5)]

    with patch.object(weather, "_build_request_candidates", return_value=candidates):
        with patch.object(
            weather,
            "_fetch_weather_for_request",
            return_value=(-30.0, 72.0, 0.85, 1.8),
        ):
            snapshot = weather.get_weather_snapshot(32.8, -96.8, mode="forecast")

    assert snapshot.sources == (
        weather.SourceAttribution(name="gfs", timestamp="20260407 06z f005"),
    )
    assert snapshot.weather["cloud_cover_high"] == pytest.approx(0.85)


def test_read_csv_records_supports_wgrib2_seven_column_output(tmp_path: Path) -> None:
    csv_path = tmp_path / "subset.csv"
    csv_path.write_text(
        '"2026-04-07 12:00:00","2026-04-07 16:00:00","TMP","250 mb",-96.75,32.75,221.522\n',
        encoding="utf-8",
    )

    records = weather._read_csv_records(csv_path)

    assert records == [
        weather.GribRecord(
            init_time="2026-04-07 12:00:00",
            valid_time="2026-04-07 16:00:00",
            variable="TMP",
            level="250 mb",
            longitude=-96.75,
            latitude=32.75,
            value=221.522,
        )
    ]


def test_persist_downloaded_files_copies_grib_and_csv(tmp_path: Path) -> None:
    grib_path = tmp_path / "subset.grib2"
    csv_path = tmp_path / "subset.csv"
    grib_path.write_bytes(b"GRIBtest")
    csv_path.write_text("value\n", encoding="utf-8")

    download_dir = tmp_path / "cache"
    persisted_grib_path, persisted_csv_path = weather._persist_downloaded_files(
        grib_path,
        csv_path,
        weather.GfsRequest(cycle_date="20260407", cycle_hour=12, forecast_hour=4),
        lat=32.8,
        lon=-96.8,
        download_dir=download_dir,
    )

    assert persisted_grib_path.exists()
    assert persisted_csv_path.exists()
    assert persisted_grib_path.read_bytes() == b"GRIBtest"
    assert persisted_csv_path.read_text(encoding="utf-8") == "value\n"
    assert persisted_grib_path.name.endswith(".grib2")
    assert persisted_csv_path.name.endswith(".csv")
    assert "20260407_t12z_f004" in persisted_grib_path.name


def test_extract_metar_high_cloud_cover_uses_high_layers_only() -> None:
    record = {
        "clouds": [
            {"cover": "SCT", "base": 12000},
            {"cover": "BKN", "base": 25000},
        ]
    }

    payload = weather._extract_metar_high_cloud_cover(record)

    assert payload == pytest.approx(0.75)


def test_extract_metar_precipitation_flags_rain_weather_codes() -> None:
    record = {
        "rawOb": "METAR KDAL 071653Z 15009KT 3SM -RA BKN250 19/02 A3019",
    }

    payload = weather._extract_metar_precipitation(record)

    assert payload == pytest.approx(1.0)


def test_extract_metar_timestamp_uses_iso_field_when_available() -> None:
    record = {
        "obsTime": "2026-04-07T16:53:00Z",
        "rawOb": "METAR KDAL 071653Z 15009KT 10SM BKN250 19/02 A3019",
    }

    payload = weather._extract_metar_timestamp(record)

    assert payload == "20260407 1653z"


def test_extract_metar_timestamp_falls_back_to_raw_report_time() -> None:
    record = {
        "rawOb": "METAR KDAL 071653Z 15009KT 10SM BKN250 19/02 A3019",
    }

    with patch.object(
        weather,
        "_resolve_metar_report_datetime",
        return_value=datetime(2026, 4, 7, 16, 53, tzinfo=timezone.utc),
    ):
        payload = weather._extract_metar_timestamp(record)

    assert payload == "20260407 1653z"


def test_extract_goes_timestamp_reads_start_time_from_key() -> None:
    payload = weather._extract_goes_timestamp(
        "ABI-L2-CCLC/2026/098/01/OR_ABI-L2-CCLC-M6_G19_s20260980123456_e20260980126029_c20260980130110.nc"
    )

    assert payload == "20260408 012345z"


def test_compose_observed_weather_prefers_goes_and_keeps_upper_air_missing() -> None:
    metar_observation = weather.MetarObservation(
        station_identifier="KDAL",
        raw_record={"rawOb": "METAR KDAL 071653Z 15009KT 10SM BKN250 19/02 A3019"},
        precipitation=0.0,
        high_cloud_cover=0.75,
        timestamp="20260407 1653z",
    )
    goes_observation = weather.GoesObservation(
        bucket="noaa-goes19",
        key="sample.nc",
        high_cloud_cover=0.82,
        downloaded_path=Path("/tmp/sample.nc"),
        timestamp="20260407 124617z",
    )

    payload = weather._compose_observed_weather(metar_observation, goes_observation)

    assert math.isnan(payload["temp_250"])
    assert math.isnan(payload["humidity_250"])
    assert payload["cloud_cover_high"] == pytest.approx(0.82)
    assert payload["precipitation"] == pytest.approx(0.0)


def test_compose_observed_weather_uses_forecast_fallback_when_goes_missing() -> None:
    metar_observation = weather.MetarObservation(
        station_identifier="KDAL",
        raw_record={},
        precipitation=1.0,
        high_cloud_cover=0.75,
        timestamp="",
    )
    goes_observation = weather.GoesObservation(
        bucket="",
        key="",
        high_cloud_cover=math.nan,
        downloaded_path=None,
        timestamp="",
    )
    forecast_fallback = weather.WeatherSnapshot(
        weather={
            "temp_250": -42.0,
            "humidity_250": 56.0,
            "cloud_cover_high": 0.33,
            "precipitation": 0.0,
        },
        sources=(weather.SourceAttribution(name="gfs", timestamp="20260407 12z f006"),),
    )

    payload = weather._compose_observed_weather(
        metar_observation,
        goes_observation,
        forecast_fallback=forecast_fallback,
    )

    assert payload == {
        "temp_250": pytest.approx(-42.0),
        "humidity_250": pytest.approx(56.0),
        "cloud_cover_high": pytest.approx(0.75),
        "precipitation": pytest.approx(1.0),
    }


def test_collect_observed_sources_reports_goes_and_metar() -> None:
    metar_observation = weather.MetarObservation(
        station_identifier="KDAL",
        raw_record={"rawOb": "METAR KDAL 071653Z 15009KT 10SM BKN250 19/02 A3019"},
        precipitation=0.0,
        high_cloud_cover=0.75,
        timestamp="20260407 1653z",
    )
    goes_observation = weather.GoesObservation(
        bucket="noaa-goes19",
        key="sample.nc",
        high_cloud_cover=0.82,
        downloaded_path=Path("/tmp/sample.nc"),
        timestamp="20260407 124617z",
    )

    sources = weather._collect_observed_sources(
        metar_observation=metar_observation,
        goes_observation=goes_observation,
        forecast_fallback=None,
    )

    assert sources == (
        weather.SourceAttribution(name="goes-east", timestamp="20260407 124617z"),
        weather.SourceAttribution(name="metar", timestamp="20260407 1653z"),
    )


def test_collect_observed_sources_reports_gfs_fallback_when_goes_missing() -> None:
    metar_observation = weather.MetarObservation(
        station_identifier="KDAL",
        raw_record={"rawOb": "METAR KDAL 071653Z 15009KT 3SM -RA BKN250 19/02 A3019"},
        precipitation=1.0,
        high_cloud_cover=0.75,
        timestamp="20260407 1653z",
    )
    goes_observation = weather.GoesObservation(
        bucket="",
        key="",
        high_cloud_cover=math.nan,
        downloaded_path=None,
        timestamp="",
    )
    forecast_fallback = weather.WeatherSnapshot(
        weather={
            "temp_250": -42.0,
            "humidity_250": 56.0,
            "cloud_cover_high": math.nan,
            "precipitation": math.nan,
        },
        sources=(weather.SourceAttribution(name="gfs", timestamp="20260407 12z f006"),),
    )

    sources = weather._collect_observed_sources(
        metar_observation=metar_observation,
        goes_observation=goes_observation,
        forecast_fallback=forecast_fallback,
    )

    assert sources == (
        weather.SourceAttribution(name="gfs", timestamp="20260407 12z f006"),
        weather.SourceAttribution(name="metar", timestamp="20260407 1653z"),
    )


def test_find_latest_goes_object_prefers_bucket_family_before_recency(monkeypatch) -> None:
    monkeypatch.setattr(weather, "_preferred_goes_buckets", lambda lon: ("east", "west"))

    def fake_list_s3_keys(bucket: str, prefix: str) -> tuple[str, ...]:
        if bucket == "east" and prefix.endswith("/01/"):
            return ("east-file.nc",)
        if bucket == "west" and prefix.endswith("/17/"):
            return ("west-file.nc",)
        return ()

    monkeypatch.setattr(weather, "_list_s3_keys", fake_list_s3_keys)

    bucket, key = weather._find_latest_goes_object(
        datetime(2026, 4, 7, 17, 0, tzinfo=timezone.utc),
        lon=-96.8,
    )

    assert bucket == "east"
    assert key == "east-file.nc"


def test_get_weather_dispatches_observed_mode() -> None:
    expected = weather.WeatherSnapshot(
        weather={
            "temp_250": math.nan,
            "humidity_250": math.nan,
            "cloud_cover_high": 0.75,
            "precipitation": 0.0,
        },
        sources=(
            weather.SourceAttribution(name="goes-east", timestamp="20260407 124617z"),
            weather.SourceAttribution(name="metar", timestamp="20260407 1653z"),
        ),
    )

    with patch.object(weather, "_get_observed_snapshot", return_value=expected) as observed_mock:
        payload = weather.get_weather(32.8, -96.8, mode="observed")

    observed_mock.assert_called_once_with(
        32.8,
        -96.8,
        keep_downloaded_files=False,
        download_dir=None,
    )
    assert math.isnan(payload["temp_250"])
    assert math.isnan(payload["humidity_250"])
    assert payload["cloud_cover_high"] == pytest.approx(0.75)
    assert payload["precipitation"] == pytest.approx(0.0)


def test_get_weather_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError):
        weather.get_weather(32.8, -96.8, mode="nowcast")


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        (-91.0, 0.0),
        (91.0, 0.0),
        (0.0, -181.0),
        (0.0, 181.0),
    ],
)
def test_get_weather_rejects_invalid_coordinates(lat: float, lon: float) -> None:
    with pytest.raises(ValueError):
        weather.get_weather(lat, lon)
