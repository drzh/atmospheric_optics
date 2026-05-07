"""Constants for deterministic WMO cloud classification rules."""

from __future__ import annotations

FEET_TO_METERS = 0.3048
LOW_CLOUD_TOP_M = 2_000.0
MID_CLOUD_TOP_M = 6_000.0
HIGH_CLOUD_BASE_M = 6_000.0

COVERAGE_BY_METAR_CODE: dict[str, str] = {
    "FEW": "few",
    "SCT": "scattered",
    "BKN": "broken",
    "OVC": "overcast",
    "VV": "overcast",
}
COVERAGE_FRACTION_BY_NAME: dict[str, float] = {
    "clear": 0.0,
    "few": 0.125,
    "scattered": 0.375,
    "broken": 0.75,
    "overcast": 1.0,
}
WMO_NAME_BY_CODE: dict[str, str] = {
    "Ci": "Cirrus",
    "Cs": "Cirrostratus",
    "Cc": "Cirrocumulus",
    "As": "Altostratus",
    "Ac": "Altocumulus",
    "Ns": "Nimbostratus",
    "Sc": "Stratocumulus",
    "St": "Stratus",
    "Cu": "Cumulus",
    "Cb": "Cumulonimbus",
}
PRECIPITATION_WEATHER_CODES: tuple[str, ...] = (
    "DZ",
    "RA",
    "SN",
    "SG",
    "IC",
    "PL",
    "GR",
    "GS",
    "UP",
    "SHRA",
    "SHSN",
    "TSRA",
    "TSSN",
    "FZRA",
    "FZDZ",
)
CONVECTIVE_WEATHER_CODES: tuple[str, ...] = (
    "TS",
    "TSRA",
    "TSSN",
    "SHRA",
    "SHSN",
)
