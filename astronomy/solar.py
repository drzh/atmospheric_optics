"""Solar position calculations based on NOAA solar formulas."""

from __future__ import annotations

import math
from datetime import datetime, timezone


def get_solar_position(lat: float, lon: float, time: datetime) -> dict[str, float]:
    """Return solar elevation and azimuth for a specific moment."""

    _validate_coordinates(lat, lon)
    moment = _normalize_time(time)

    julian_century = _julian_century(moment)
    geom_mean_longitude = _geom_mean_longitude_sun(julian_century)
    geom_mean_anomaly = _geom_mean_anomaly_sun(julian_century)
    eccentricity = _earth_orbit_eccentricity(julian_century)
    sun_eq_center = _sun_equation_of_center(julian_century, geom_mean_anomaly)
    sun_true_longitude = geom_mean_longitude + sun_eq_center
    sun_apparent_longitude = _sun_apparent_longitude(julian_century, sun_true_longitude)
    mean_obliquity = _mean_obliquity_ecliptic(julian_century)
    obliquity_correction = _obliquity_correction(julian_century, mean_obliquity)
    solar_declination = _solar_declination(obliquity_correction, sun_apparent_longitude)
    equation_of_time = _equation_of_time(
        julian_century,
        geom_mean_longitude,
        geom_mean_anomaly,
        eccentricity,
        obliquity_correction,
    )

    true_solar_time = _true_solar_time_minutes(moment, lon, equation_of_time)
    hour_angle = _hour_angle(true_solar_time)

    lat_radians = math.radians(lat)
    declination_radians = math.radians(solar_declination)
    hour_angle_radians = math.radians(hour_angle)

    cos_zenith = (
        math.sin(lat_radians) * math.sin(declination_radians)
        + math.cos(lat_radians) * math.cos(declination_radians) * math.cos(hour_angle_radians)
    )
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith = math.degrees(math.acos(cos_zenith))
    elevation = 90.0 - zenith

    azimuth = (
        math.degrees(
            math.atan2(
                math.sin(hour_angle_radians),
                math.cos(hour_angle_radians) * math.sin(lat_radians)
                - math.tan(declination_radians) * math.cos(lat_radians),
            )
        )
        + 180.0
    ) % 360.0

    return {
        "elevation": elevation,
        "azimuth": azimuth,
    }


def _validate_coordinates(lat: float, lon: float) -> None:
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"Latitude must be between -90 and 90. Received {lat}.")
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"Longitude must be between -180 and 180. Received {lon}.")


def _normalize_time(moment: datetime) -> datetime:
    if not isinstance(moment, datetime):
        raise TypeError("time must be a datetime instance.")
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _julian_century(moment: datetime) -> float:
    return (_julian_day(moment) - 2451545.0) / 36525.0


def _julian_day(moment: datetime) -> float:
    year = moment.year
    month = moment.month
    day = moment.day
    hour = moment.hour + (moment.minute / 60.0) + (moment.second / 3600.0) + (moment.microsecond / 3_600_000_000.0)

    if month <= 2:
        year -= 1
        month += 12

    century = math.floor(year / 100)
    leap_correction = 2 - century + math.floor(century / 4)
    julian_day = (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + day
        + leap_correction
        - 1524.5
    )
    return julian_day + (hour / 24.0)


def _geom_mean_longitude_sun(julian_century: float) -> float:
    return (280.46646 + julian_century * (36000.76983 + julian_century * 0.0003032)) % 360.0


def _geom_mean_anomaly_sun(julian_century: float) -> float:
    return 357.52911 + julian_century * (35999.05029 - 0.0001537 * julian_century)


def _earth_orbit_eccentricity(julian_century: float) -> float:
    return 0.016708634 - julian_century * (0.000042037 + 0.0000001267 * julian_century)


def _sun_equation_of_center(julian_century: float, geom_mean_anomaly: float) -> float:
    anomaly_radians = math.radians(geom_mean_anomaly)
    return (
        math.sin(anomaly_radians) * (1.914602 - julian_century * (0.004817 + 0.000014 * julian_century))
        + math.sin(2.0 * anomaly_radians) * (0.019993 - 0.000101 * julian_century)
        + math.sin(3.0 * anomaly_radians) * 0.000289
    )


def _sun_apparent_longitude(julian_century: float, sun_true_longitude: float) -> float:
    omega = 125.04 - 1934.136 * julian_century
    return sun_true_longitude - 0.00569 - 0.00478 * math.sin(math.radians(omega))


def _mean_obliquity_ecliptic(julian_century: float) -> float:
    seconds = 21.448 - julian_century * (
        46.815 + julian_century * (0.00059 - julian_century * 0.001813)
    )
    return 23.0 + ((26.0 + (seconds / 60.0)) / 60.0)


def _obliquity_correction(julian_century: float, mean_obliquity: float) -> float:
    omega = 125.04 - 1934.136 * julian_century
    return mean_obliquity + 0.00256 * math.cos(math.radians(omega))


def _solar_declination(obliquity_correction: float, sun_apparent_longitude: float) -> float:
    return math.degrees(
        math.asin(
            math.sin(math.radians(obliquity_correction))
            * math.sin(math.radians(sun_apparent_longitude))
        )
    )


def _equation_of_time(
    julian_century: float,
    geom_mean_longitude: float,
    geom_mean_anomaly: float,
    eccentricity: float,
    obliquity_correction: float,
) -> float:
    variable_y = math.tan(math.radians(obliquity_correction) / 2.0)
    variable_y *= variable_y

    longitude_radians = math.radians(geom_mean_longitude)
    anomaly_radians = math.radians(geom_mean_anomaly)

    equation = (
        variable_y * math.sin(2.0 * longitude_radians)
        - 2.0 * eccentricity * math.sin(anomaly_radians)
        + 4.0 * eccentricity * variable_y * math.sin(anomaly_radians) * math.cos(2.0 * longitude_radians)
        - 0.5 * variable_y * variable_y * math.sin(4.0 * longitude_radians)
        - 1.25 * eccentricity * eccentricity * math.sin(2.0 * anomaly_radians)
    )

    return math.degrees(equation) * 4.0


def _true_solar_time_minutes(moment: datetime, lon: float, equation_of_time: float) -> float:
    minutes_since_midnight = (
        moment.hour * 60.0
        + moment.minute
        + (moment.second / 60.0)
        + (moment.microsecond / 60_000_000.0)
    )
    return (minutes_since_midnight + equation_of_time + (4.0 * lon)) % 1440.0


def _hour_angle(true_solar_time: float) -> float:
    hour_angle = (true_solar_time / 4.0) - 180.0
    if hour_angle < -180.0:
        return hour_angle + 360.0
    return hour_angle
