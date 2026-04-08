"""Data ingestion package for atmospheric optics predictors."""

from .weather import get_weather, get_weather_snapshot

__all__ = ["get_weather", "get_weather_snapshot"]
