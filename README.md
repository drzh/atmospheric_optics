# Atmospheric Optics Predictor

Predict atmospheric optical phenomena such as halos, sundogs, circumzenithal arcs, and rainbows from live weather inputs.

The current implementation combines:

- weather ingestion from forecast and observed sources
- solar position calculation
- normalized feature engineering
- smooth quantitative probability models
- CLI and HTTP API entry points

Predictions are returned as probabilities between `0.000` and `1.000`, rounded to 3 decimal places, with source attribution and source timestamps in UTC.

## Supported Phenomena

- `halo`
- `parhelia`
- `cza`
- `rainbow`

## Weather Modes

- `forecast`
  Uses NOAA GFS forecast data.
- `observed`
  Uses GOES cloud-layer products plus nearby METAR observations.
  If GOES is unavailable, the pipeline falls back to GFS for the missing upper-air and high-cloud inputs.

## Project Layout

```text
data_ingestion -> feature_engineering -> models -> core -> cli/api
```

Key modules:

- `data_ingestion/weather.py`: GFS, GOES, METAR, and source attribution
- `feature_engineering/features.py`: normalized model inputs
- `models/`: quantitative probability models
- `core/predictor.py`: end-to-end orchestration and output shaping
- `cli/main.py`: command-line interface
- `api/main.py`: HTTP API

## Requirements

Python:

- Python 3.10+

Core runtime package:

```bash
python3 -m pip install requests
```

Forecast mode requirement:

- `wgrib2` is required to parse NOAA GFS GRIB2 data

Observed mode extras for full GOES support:

```bash
python3 -m pip install numpy netCDF4 pyproj
```

API extras:

```bash
python3 -m pip install fastapi uvicorn
```

Test dependency:

```bash
python3 -m pip install pytest
```

Full example:

```bash
python3 -m pip install requests numpy netCDF4 pyproj fastapi uvicorn pytest
```

## Quick Start

Run commands from the repo root:

```bash
cd /home/celaeno/script/atmospheric_optics
python3 cli/main.py --lat 32.8 --lon -96.8
```

Important:

- The CLI and API currently predict for the current UTC time.
- Output can change from run to run as source data and solar geometry change.

## CLI Usage

Default forecast mode:

```bash
python3 cli/main.py --lat 32.8 --lon -96.8
```

Explicit forecast mode:

```bash
python3 cli/main.py --lat 32.8 --lon -96.8 --mode forecast
```

Observed mode:

```bash
python3 cli/main.py --lat 32.8 --lon -96.8 --mode observed
```

Keep downloaded artifacts:

```bash
python3 cli/main.py --lat 32.8 --lon -96.8 --mode observed --keep-downloaded-files
python3 cli/main.py --lat 32.8 --lon -96.8 --mode forecast --download-dir /tmp/atmospheric-optics-cache
```

CLI options:

- `--lat`: latitude in decimal degrees
- `--lon`: longitude in decimal degrees
- `--mode`: `forecast` or `observed`
- `--keep-downloaded-files`: retain downloaded weather-source artifacts
- `--download-dir`: custom artifact directory

Default artifact directories:

- forecast: `data_cache/noaa_gfs`
- observed: `data_cache/nws_observations`

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

Timestamp formats:

- GFS: `YYYYMMDD HHz fNNN`
- GOES: `YYYYMMDD HHMMSSz`
- METAR: `YYYYMMDD HHMMz`

Possible source names:

- `gfs`
- `goes-east`
- `goes-west`
- `metar`

## API Usage

If `fastapi` is not installed, the repo includes a standard-library WSGI server:

```bash
python3 api/main.py
```

Then request:

```bash
curl "http://127.0.0.1:8000/predict?lat=32.8&lon=-96.8&mode=forecast"
```

If `fastapi` and `uvicorn` are installed, run:

```bash
uvicorn api.main:app --reload
```

## Data Source Behavior

Forecast mode:

- NOAA NOMADS GFS
- newest cycle attempted first
- older cycles used automatically if the newest file is not published yet

Observed mode:

- NOAA GOES cloud-layer products for cirrus and high-cloud conditions
- Aviation Weather Center METAR observations for precipitation and cloud layers
- NOAA/NWS station discovery for locating nearby observing stations

Fallback behavior:

- If GOES data is unavailable, observed mode falls back to GFS for missing upper-air and high-cloud inputs.
- If the latest GFS cycle is unavailable, forecast mode falls back to older cycles quietly before failing.

## Model Summary

The current implementation follows the quantitative model described in `MODELS.md`.

Each probability is built from smooth components such as:

- `PhysicalCondition`
- `Visibility`
- `Geometry`

The model uses sigmoid and gaussian helper functions plus weighted combination terms, rather than hard threshold rules.

## Tests

Run the full suite:

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
