"""HTTP API for atmospheric optics prediction."""

from __future__ import annotations

import json
import sys
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.predictor import predict_all

try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover - exercised through the WSGI fallback.
    FastAPI = None


FASTAPI_AVAILABLE = FastAPI is not None
WEATHER_MODES = ("forecast", "observed")


def build_prediction_response(lat: float, lon: float, mode: str = "forecast") -> dict[str, object]:
    """Return a JSON-serializable prediction payload."""

    return predict_all(lat, lon, mode=mode)


if FASTAPI_AVAILABLE:
    app = FastAPI(title="Atmospheric Optics Predictor")

    @app.get("/predict")
    def predict_endpoint(lat: float, lon: float, mode: str = "forecast") -> dict[str, object]:
        return build_prediction_response(lat, lon, mode=mode)

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
                payload = build_prediction_response(lat, lon, mode=mode)
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


if __name__ == "__main__":
    run()
