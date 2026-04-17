"""Approximate lunar position and illumination calculations."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from solar.solar_position import (
    _julian_century,
    _julian_day,
    _normalize_time,
)


def get_lunar_position(lat: float, lon: float, time: datetime) -> dict[str, float]:
    """Return approximate lunar elevation, azimuth, and illumination fraction."""

    moment = _normalize_time(time)
    days_since_j2000 = _julian_day(moment) - 2451543.5

    ascending_node = math.radians((125.1228 - (0.0529538083 * days_since_j2000)) % 360.0)
    inclination = math.radians(5.1454)
    argument_of_perigee = math.radians((318.0634 + (0.1643573223 * days_since_j2000)) % 360.0)
    eccentricity = 0.0549
    mean_anomaly = math.radians((115.3654 + (13.0649929509 * days_since_j2000)) % 360.0)

    eccentric_anomaly = _solve_kepler(mean_anomaly, eccentricity)
    orbital_x = math.cos(eccentric_anomaly) - eccentricity
    orbital_y = math.sin(eccentric_anomaly) * math.sqrt(1.0 - (eccentricity**2))
    true_anomaly = math.atan2(orbital_y, orbital_x)
    radius = math.hypot(orbital_x, orbital_y)

    ecliptic_x = radius * (
        (math.cos(ascending_node) * math.cos(true_anomaly + argument_of_perigee))
        - (math.sin(ascending_node) * math.sin(true_anomaly + argument_of_perigee) * math.cos(inclination))
    )
    ecliptic_y = radius * (
        (math.sin(ascending_node) * math.cos(true_anomaly + argument_of_perigee))
        + (math.cos(ascending_node) * math.sin(true_anomaly + argument_of_perigee) * math.cos(inclination))
    )
    ecliptic_z = radius * math.sin(true_anomaly + argument_of_perigee) * math.sin(inclination)

    lunar_longitude = math.atan2(ecliptic_y, ecliptic_x)
    lunar_latitude = math.atan2(ecliptic_z, math.hypot(ecliptic_x, ecliptic_y))
    obliquity = math.radians(_mean_obliquity(moment))

    equatorial_x = math.cos(lunar_longitude) * math.cos(lunar_latitude)
    equatorial_y = (
        math.sin(lunar_longitude) * math.cos(lunar_latitude) * math.cos(obliquity)
        - math.sin(lunar_latitude) * math.sin(obliquity)
    )
    equatorial_z = (
        math.sin(lunar_longitude) * math.cos(lunar_latitude) * math.sin(obliquity)
        + math.sin(lunar_latitude) * math.cos(obliquity)
    )

    right_ascension = math.atan2(equatorial_y, equatorial_x)
    declination = math.asin(max(-1.0, min(1.0, equatorial_z)))
    hour_angle = math.radians(_local_sidereal_time_degrees(moment, lon)) - right_ascension

    latitude_radians = math.radians(lat)
    elevation = math.asin(
        max(
            -1.0,
            min(
                1.0,
                (math.sin(declination) * math.sin(latitude_radians))
                + (math.cos(declination) * math.cos(latitude_radians) * math.cos(hour_angle)),
            ),
        )
    )
    azimuth = math.atan2(
        math.sin(hour_angle),
        (math.cos(hour_angle) * math.sin(latitude_radians))
        - (math.tan(declination) * math.cos(latitude_radians)),
    )

    solar_longitude = math.radians(_solar_apparent_longitude_degrees(moment))
    elongation = math.acos(
        max(
            -1.0,
            min(
                1.0,
                (math.cos(lunar_latitude) * math.cos(lunar_longitude - solar_longitude)),
            ),
        )
    )
    phase = 0.5 * (1.0 - math.cos(elongation))

    return {
        "elevation": math.degrees(elevation),
        "azimuth": (math.degrees(azimuth) + 180.0) % 360.0,
        "phase": phase,
        "illuminance": phase**1.5,
    }


def _solve_kepler(mean_anomaly: float, eccentricity: float) -> float:
    eccentric_anomaly = mean_anomaly
    for _ in range(8):
        delta = (
            eccentric_anomaly
            - (eccentricity * math.sin(eccentric_anomaly))
            - mean_anomaly
        ) / (1.0 - (eccentricity * math.cos(eccentric_anomaly)))
        eccentric_anomaly -= delta
        if abs(delta) < 1.0e-8:
            break
    return eccentric_anomaly


def _mean_obliquity(moment: datetime) -> float:
    julian_century = _julian_century(moment)
    seconds = 21.448 - julian_century * (
        46.815 + julian_century * (0.00059 - (julian_century * 0.001813))
    )
    return 23.0 + ((26.0 + (seconds / 60.0)) / 60.0)


def _local_sidereal_time_degrees(moment: datetime, lon: float) -> float:
    julian_day = _julian_day(moment)
    julian_century = (julian_day - 2451545.0) / 36525.0
    greenwich_sidereal = (
        280.46061837
        + (360.98564736629 * (julian_day - 2451545.0))
        + (0.000387933 * (julian_century**2))
        - ((julian_century**3) / 38710000.0)
    )
    return (greenwich_sidereal + lon) % 360.0


def _solar_apparent_longitude_degrees(moment: datetime) -> float:
    julian_century = _julian_century(moment)
    mean_longitude = (280.46646 + (julian_century * (36000.76983 + (julian_century * 0.0003032)))) % 360.0
    mean_anomaly = math.radians(357.52911 + (julian_century * (35999.05029 - (0.0001537 * julian_century))))
    equation_of_center = (
        math.sin(mean_anomaly) * (1.914602 - (julian_century * (0.004817 + (0.000014 * julian_century))))
        + (math.sin(2.0 * mean_anomaly) * (0.019993 - (0.000101 * julian_century)))
        + (math.sin(3.0 * mean_anomaly) * 0.000289)
    )
    true_longitude = mean_longitude + equation_of_center
    omega = 125.04 - (1934.136 * julian_century)
    return true_longitude - 0.00569 - (0.00478 * math.sin(math.radians(omega)))
