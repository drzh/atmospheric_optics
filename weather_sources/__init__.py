"""Data ingestion package for atmospheric optics predictors."""

"""Weather source loading for forecast and observed prediction modes."""

from .snapshots import get_weather, get_weather_snapshot

__all__ = ["get_weather", "get_weather_snapshot"]
