"""HTTP API for atmospheric optics prediction."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.predictor import predict_all

try:
    from fastapi import FastAPI, HTTPException
except ImportError:  # pragma: no cover - exercised through the WSGI fallback.
    FastAPI = None
    HTTPException = None


FASTAPI_AVAILABLE = FastAPI is not None
WEATHER_MODES = ("forecast", "observed")
ILLUMINATION_MODES = ("solar", "lunar")


def build_prediction_response(
    lat: float,
    lon: float,
    mode: str = "forecast",
    illumination: str = "solar",
    at_time: str | None = None,
    time_window_hours: str | None = None,
    phenomena: str | None = None,
    spatial_resolution_km: float | None = None,
    lightweight: bool = False,
    debug: bool = False,
) -> dict[str, object]:
    """Return a JSON-serializable prediction payload."""

    predictor_kwargs: dict[str, object] = {
        "mode": mode,
        "illumination": _normalize_illumination(illumination),
        "at_time": _parse_at_time(at_time),
        "time_window_hours": _parse_time_window_hours(time_window_hours),
    }
    parsed_phenomena = _parse_csv_values(phenomena)
    if parsed_phenomena is not None:
        predictor_kwargs["phenomena"] = parsed_phenomena
    if spatial_resolution_km is not None:
        predictor_kwargs["spatial_resolution_km"] = spatial_resolution_km
    if lightweight:
        predictor_kwargs["lightweight"] = True
    if debug:
        predictor_kwargs["debug"] = True

    return predict_all(lat, lon, **predictor_kwargs)


if FASTAPI_AVAILABLE:
    app = FastAPI(title="Atmospheric Optics Predictor")

    @app.get("/predict")
    def predict_endpoint(
        lat: float,
        lon: float,
        mode: str = "forecast",
        illumination: str = "solar",
        at_time: str | None = None,
        time_window_hours: str | None = None,
        phenomena: str | None = None,
        spatial_resolution_km: float | None = None,
        lightweight: bool = False,
        debug: bool = False,
    ) -> dict[str, object]:
        try:
            return build_prediction_response(
                lat,
                lon,
                mode=mode,
                illumination=illumination,
                at_time=at_time,
                time_window_hours=time_window_hours,
                phenomena=phenomena,
                spatial_resolution_km=spatial_resolution_km,
                lightweight=lightweight,
                debug=debug,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

else:

    class PredictApplication:
        """Minimal WSGI app for environments without FastAPI installed."""

        def __call__(self, environ: dict[str, str], start_response) -> list[bytes]:
            if environ.get("REQUEST_METHOD") != "GET" or environ.get("PATH_INFO") != "/predict":
                return _json_response(start_response, HTTPStatus.NOT_FOUND, {"detail": "Not Found"})

            query = parse_qs(environ.get("QUERY_STRING", ""))
            try:
                lat = _required_float(query, "lat")
                lon = _required_float(query, "lon")
                mode = _optional_mode(query)
                payload = build_prediction_response(
                    lat,
                    lon,
                    mode=mode,
                    illumination=_optional_illumination(query),
                    at_time=_optional_string(query, "at_time"),
                    time_window_hours=_optional_string(query, "time_window_hours"),
                    phenomena=_optional_string(query, "phenomena"),
                    spatial_resolution_km=_optional_float(query, "spatial_resolution_km"),
                    lightweight=_optional_bool(query, "lightweight"),
                    debug=_optional_bool(query, "debug"),
                )
            except ValueError as exc:
                return _json_response(start_response, HTTPStatus.BAD_REQUEST, {"detail": str(exc)})

            return _json_response(start_response, HTTPStatus.OK, payload)


    def _required_float(query: dict[str, list[str]], key: str) -> float:
        values = query.get(key)
        if not values or not values[0].strip():
            raise ValueError(f"Missing required query parameter: {key}")

        try:
            return float(values[0])
        except ValueError as exc:
            raise ValueError(f"Invalid float for {key}: {values[0]}") from exc


    def _optional_mode(query: dict[str, list[str]]) -> str:
        values = query.get("mode")
        if not values or not values[0].strip():
            return "forecast"

        mode = values[0].strip().lower()
        if mode not in WEATHER_MODES:
            raise ValueError(f"Invalid mode: {mode}. Expected one of {', '.join(WEATHER_MODES)}")
        return mode


    def _optional_illumination(query: dict[str, list[str]]) -> str:
        values = query.get("illumination")
        if not values or not values[0].strip():
            return "solar"

        illumination = values[0].strip().lower()
        if illumination not in ILLUMINATION_MODES:
            raise ValueError(
                f"Invalid illumination: {illumination}. Expected one of {', '.join(ILLUMINATION_MODES)}"
            )
        return illumination


    def _optional_string(query: dict[str, list[str]], key: str) -> str | None:
        values = query.get(key)
        if not values:
            return None
        value = values[0].strip()
        return value or None


    def _optional_float(query: dict[str, list[str]], key: str) -> float | None:
        value = _optional_string(query, key)
        if value is None:
            return None
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid float for {key}: {value}") from exc


    def _optional_bool(query: dict[str, list[str]], key: str) -> bool:
        value = _optional_string(query, key)
        if value is None:
            return False
        normalized = value.lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean for {key}: {value}")


    def _json_response(start_response, status: HTTPStatus, payload: dict[str, object]) -> list[bytes]:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        start_response(
            f"{status.value} {status.phrase}",
            [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ],
        )
        return [body]


    app = PredictApplication()


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the API using the standard library server when FastAPI is unavailable."""

    if FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI is installed. Run this app with an ASGI server such as uvicorn.")

    with make_server(host, port, app) as server:
        server.serve_forever()

def _parse_at_time(value: str | None) -> datetime | None:
    if value is None or not value.strip():
        return None

    parsed_value = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed_value.tzinfo is None:
        return parsed_value.replace(tzinfo=timezone.utc)
    return parsed_value.astimezone(timezone.utc)


def _parse_time_window_hours(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None

    hours: list[int] = []
    for part in value.split(","):
        normalized_part = part.strip()
        if not normalized_part:
            continue
        hours.append(int(normalized_part))
    return tuple(hours)


def _normalize_illumination(value: str) -> str:
    illumination = str(value).strip().lower()
    if illumination not in ILLUMINATION_MODES:
        raise ValueError(f"Invalid illumination: {value}. Expected one of {', '.join(ILLUMINATION_MODES)}")
    return illumination


def _parse_csv_values(value: str | None) -> tuple[str, ...] | None:
    if value is None or not value.strip():
        return None

    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        return None
    return tuple(items)


if __name__ == "__main__":
    run()
