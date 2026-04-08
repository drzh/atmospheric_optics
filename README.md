# Atmospheric Optics Predictor

Predict atmospheric optical phenomena such as halos, sundogs, circumzenithal arcs, and rainbows from live weather inputs.

The project supports two weather modes:

- `forecast`: NOAA GFS forecast data
- `observed`: GOES cloud-layer products plus nearby METAR observations, with GFS fallback if satellite data is unavailable

Predictions are returned as smooth probabilities between `0.000` and `1.000`, rounded to 3 decimal places, with source attribution and source timestamps.

## What It Predicts

- `halo`
- `parhelia`
- `cza`
- `rainbow`

## Project Layout

```text
data_ingestion -> feature_engineering -> models -> core -> cli/api
```

Key modules:

- `data_ingestion/weather.py`: GFS, GOES, and METAR ingestion
- `feature_engineering/features.py`: normalized model features
- `models/`: quantitative probability models
- `core/predictor.py`: end-to-end orchestration
- `cli/main.py`: command-line interface
- `api/main.py`: HTTP API

## Dependencies

Base Python packages:

```bash
python3 -m pip install requests pytest
```

Optional Python packages:

- `numpy`, `netCDF4`, `pyproj`: required for GOES-based observed mode
- `fastapi`, `uvicorn`: optional for an ASGI API server

Example:

```bash
python3 -m pip install requests pytest numpy netCDF4 pyproj fastapi uvicorn
```

External binary:

- `wgrib2` is required for GFS GRIB2 parsing in forecast mode

## CLI Usage

Run from the repo root:

```bash
python3 cli/main.py --lat 32.8 --lon -96.8
```

Forecast mode:

```bash
python3 cli/main.py --lat 32.8 --lon -96.8 --mode forecast
```

Observed mode:

```bash
python3 cli/main.py --lat 32.8 --lon -96.8 --mode observed
```

Keep downloaded source artifacts:

```bash
python3 cli/main.py --lat 32.8 --lon -96.8 --mode observed --keep-downloaded-files
python3 cli/main.py --lat 32.8 --lon -96.8 --mode forecast --download-dir /tmp/atmospheric-optics-cache
```

CLI options:

- `--lat`: latitude in decimal degrees
- `--lon`: longitude in decimal degrees
- `--mode`: `forecast` or `observed`
- `--keep-downloaded-files`: retain downloaded source artifacts
- `--download-dir`: custom artifact directory

## Output Format

Example forecast response:

```json
{
  "cza": 0.0,
  "halo": 0.116,
  "parhelia": 0.116,
  "rainbow": 0.023,
  "sources": [
    {
      "name": "gfs",
      "timestamp": "20260407 12z f008"
    }
  ]
}
```

Example observed response:

```json
{
  "cza": 0.0,
  "halo": 0.0,
  "parhelia": 0.0,
  "rainbow": 0.023,
  "sources": [
    {
      "name": "goes-east",
      "timestamp": "20260407 124617z"
    },
    {
      "name": "metar",
      "timestamp": "20260407 1953z"
    }
  ]
}
```

Timestamp conventions:

- GFS: `YYYYMMDD HHz fNNN`
- GOES: `YYYYMMDD HHMMSSz`
- METAR: `YYYYMMDD HHMMz`

## API Usage

If `fastapi` is not installed, you can run the built-in WSGI server:

```bash
python3 api/main.py
```

Then call:

```bash
curl "http://127.0.0.1:8000/predict?lat=32.8&lon=-96.8&mode=forecast"
```

If `fastapi` and `uvicorn` are installed, use:

```bash
uvicorn api.main:app --reload
```

## Data Sources

Forecast mode:

- NOAA NOMADS GFS

Observed mode:

- NOAA GOES cloud-layer products
- Aviation Weather Center METAR observations
- NOAA/NWS station discovery

Behavior notes:

- Forecast mode tries the newest GFS cycle first and falls back to older cycles if the latest file is not published yet.
- Observed mode prefers GOES for cirrus/high-cloud detection and METAR for precipitation/cloud layers.
- If GOES is unavailable, observed mode falls back to GFS for the missing upper-air and high-cloud inputs.

## Model Notes

The current implementation uses a smooth, physics-informed probability model:

- `PhysicalCondition`
- `Visibility`
- `Geometry`

Each final prediction is a weighted combination of those terms using sigmoid and gaussian helper functions. The detailed specification lives in `MODELS.md`.

## Tests

Run the full test suite with:

```bash
pytest -q tests
```

## Repo Docs

- `ARCHITECTURE.md`: system design
- `DATA_SOURCES.md`: source selection rules
- `MODELS.md`: quantitative model definition
- `TASKS.md`: implementation phases
- `TESTS.md`: test expectations
- `CODING_GUIDELINES.md`: local coding conventions
