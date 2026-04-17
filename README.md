# Atmospheric Optics Predictor

Predict atmospheric optical phenomena from live weather inputs for both solar and lunar illumination modes.

The current implementation combines:

- weather ingestion from forecast and observed sources
- solar and lunar position calculation
- derived feature engineering with multi-layer ice support and wind stability
- smooth quantitative probability models with weighted log combination
- adaptive km-based spatial sampling with distance and directional weighting
- asymmetric peak-preserving temporal smoothing across the output timeline
- CLI and HTTP API entry points

Predictions are returned as a top-level object with `generated_at`, `request`, `sources`, and `phenomena`. Each phenomenon entry contains nested `current`, `peak`, and `timeline` objects so the format can grow cleanly as new phenomena, data sources, and supporting metadata are added. Numeric values are rounded to 3 decimal places where applicable, and source timestamps are emitted in UTC. When requested, the payload also includes per-phenomenon `debug` terms for the physical (`P`), visibility (`V`), and geometry (`G`) components.

## Supported Phenomena

Solar mode:

- `halo`
- `parhelia`
- `cza`
- `circumhorizontal_arc`
- `upper_tangent_arc`
- `sun_pillar`
- `crepuscular_rays`
- `rainbow`
- `fogbow`

Lunar mode:

- `lunar_halo`
- `paraselenae`
- `lunar_pillar`
- `lunar_corona`
- `moonbow`

## Weather Modes

- `forecast`
  Uses NOAA GFS forecast data, including short-horizon timeline sampling.
- `observed`
  Uses GOES cloud-layer products, GOES cloud optical depth when available, nearby METAR observations, and GFS upper-air fallback fields.
  If GOES optical-depth data is unavailable, the pipeline falls back to a GFS condensate proxy and then to the legacy cirrus-coverage approximation.

## Project Layout

```text
data_ingestion -> feature_engineering -> models -> core -> cli/api
```

Key modules:

- `data_ingestion/weather.py`: GFS, GOES, METAR, and source attribution
- `feature_engineering/features.py`: normalized and derived model inputs
- `feature_engineering/cirrus.py`: thin-cirrus and ice-presence features
- `feature_engineering/dynamics.py`: plate-alignment and cloud-variability features
- `models/`: quantitative probability models
- `models/combine.py`: weighted log combination helper
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

- The CLI and API default to the current UTC time, but `--at-time` / `at_time` can override that.
- Output can change from run to run as source data and solar geometry change.

## CLI Usage

Default forecast mode:

```bash
python3 cli/main.py --lat 32.8 --lon -96.8
```

Lunar mode:

```bash
python3 cli/main.py --lat 32.8 --lon -96.8 --illumination lunar
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
- `--illumination`: `solar` or `lunar`
- `--at-time`: optional ISO 8601 prediction time
- `--time-window-hours`: optional comma-separated horizon offsets such as `0,1,2,3`
- `--phenomena`: optional comma-separated subset such as `halo,fogbow`
- `--spatial-resolution-km`: optional 3x3 sample spacing inside the adaptive spatial radius
- `--lightweight`: skip spatial sampling and evaluate only the center point
- `--debug`: include per-phenomenon `P`/`V`/`G` model terms in the output
- `--keep-downloaded-files`: retain downloaded weather-source artifacts
- `--download-dir`: custom artifact directory

Default artifact directories:

- forecast: `data_cache/noaa_gfs`
- observed: `data_cache/nws_observations`

Cache behavior:

- Downloaded artifact directories are treated as short-lived caches rather than permanent archives.
- Forecast cache files are pruned to the active `time_window_hours` for the current request.
- Observed-mode cached GOES and persisted observation files are pruned once they age out of the active time scale.
- Transient GOES download folders are removed automatically when files are not being persisted.
- In the sibling deployment, a daily cron maintenance job also prunes user-owned `/tmp` entries older than 3 days to keep scratch download directories from accumulating indefinitely.

## Output Format

The predictor returns one JSON object per request:

```json
{
  "generated_at": "2026-04-07T18:00:04Z",
  "request": {
    "location": {
      "lat": 32.8,
      "lon": -96.8
    },
    "mode": "forecast",
    "prediction_time": "2026-04-07T18:00:00Z",
    "time_window_hours": [0, 1, 2, 3],
    "options": {
      "lightweight": false,
      "debug": false,
      "illumination": "solar",
      "phenomena": ["halo", "fogbow"]
    }
  },
  "sources": [
    {
      "id": "gfs",
      "label": "GFS",
      "kind": "forecast_model",
      "timestamp": "20260407 12z f006"
    }
  ],
  "phenomena": [
    {
      "id": "halo",
      "label": "Halo",
      "category": "ice_crystal",
      "current": {
        "probability": 0.178,
        "confidence": 0.925,
        "confidence_components": {
          "data": 0.901,
          "spatial": 0.992,
          "temporal": 0.944,
          "feature": 0.874
        },
        "reason": "Thin icy cirrus with visible sun and favorable solar elevation supports halo formation.",
        "spatial_context": {
          "radius_km": 40.0,
          "aggregation": "weighted_blend",
          "center_probability": 0.181,
          "mean_probability": 0.178,
          "max_probability": 0.214,
          "min_probability": 0.146,
          "spatial_variance": 0.001,
          "spatial_consistency": 0.999,
          "spatial_gradient": 0.068,
          "edge_signal": 0.033
        }
      },
      "peak": {
        "probability": 0.251,
        "time": "2026-04-07T20:00:00Z"
      },
      "timeline": [
        {
          "label": "now",
          "offset_hours": 0,
          "probability": 0.178
        },
        {
          "label": "1h",
          "offset_hours": 1,
          "probability": 0.214
        },
        {
          "label": "2h",
          "offset_hours": 2,
          "probability": 0.251
        },
        {
          "label": "3h",
          "offset_hours": 3,
          "probability": 0.239
        }
      ]
    },
    {
      "id": "fogbow",
      "label": "Fogbow",
      "category": "water_droplet",
      "current": {
        "probability": 0.488,
        "confidence": 0.942,
        "reason": "Fog or mist with visible sun and low visibility favors fogbow formation."
      },
      "peak": {
        "probability": 0.548,
        "time": "2026-04-07T21:00:00Z"
      },
      "timeline": [
        {
          "label": "now",
          "offset_hours": 0,
          "probability": 0.488
        },
        {
          "label": "1h",
          "offset_hours": 1,
          "probability": 0.522
        }
      ]
    }
  ]
}
```

Top-level fields:

- `generated_at`: UTC timestamp for when the predictor finished building the payload.
- `request`: normalized request metadata, including the resolved location, mode, illumination, prediction time, requested horizon, and runtime options.
- `sources`: list of source objects used for the prediction. Each source has `id`, `label`, `kind`, and `timestamp`.
- `phenomena`: list of per-phenomenon result objects. A list is used instead of top-level keyed fields so the format can grow without changing the surrounding envelope.

Each phenomenon entry contains:

- `id`, `label`, and `category`
- `current`: the current-site probability, confidence, reason text, `spatial_context`, and optional `debug`
- `peak`: the strongest timeline slot with nested `probability` and `time`
- `timeline`: ordered per-slot probabilities, each with `label`, `offset_hours`, and `probability`

Notes:

- `timeline` values are temporally smoothed across the requested horizon with an asymmetric kernel that preserves sharp onset and slower decay.
- `peak.time` stays pinned to the raw peak slot when the unsmoothed peak already exceeds `0.6`.
- `confidence_components` breaks the final confidence into explicit data, spatial, temporal, and feature terms.
- `spatial_context` summarizes the weighted 3x3 sampling field for that phenomenon.
- `debug` is optional and only appears when the CLI/API `debug` flag is enabled.

### Using Spatial Context

`spatial_context` is meant to answer a different question than `current.probability`.

- `current.probability` is the final site-level score after spatial aggregation.
- `spatial_context` explains whether that score comes from a broad favorable field, a localized pocket, or a structured cloud pattern near the site.

Field meanings:

- `radius_km`: the adaptive radius of influence used for the weighted 3x3 sampling field.
- `aggregation`: the current spatial merge rule. It is `weighted_blend`, which combines weighted mean support and weighted local peaks.
- `center_probability`: the weighted probability at the requested latitude/longitude.
- `mean_probability`: the weighted mean of the sampled neighborhood.
- `max_probability`: the strongest weighted sampled probability in the neighborhood.
- `min_probability`: the weakest weighted sampled probability in the neighborhood.
- `spatial_variance`: how uneven the neighborhood is. Higher values mean the favorable signal is patchy or sharply changing across the grid.
- `spatial_consistency`: `1 - spatial_variance`, clamped to `0..1`. Higher values mean neighboring points agree more strongly.
- `spatial_gradient`: `max_probability - min_probability`, useful for spotting sharp nearby transitions.
- `edge_signal`: `max_probability - center_probability`, useful for spotting stronger nearby pockets than the exact site.

How to interpret it:

- If `center_probability`, `mean_probability`, and `max_probability` are all similar and `spatial_consistency` is high, the phenomenon is spatially broad and stable near the site.
- If `max_probability` is much higher than `center_probability`, a stronger signal exists nearby than directly overhead. This is useful for edge-of-cloud or approaching-field situations.
- If `center_probability` is higher than `mean_probability`, the site is favorable but the surrounding area is less supportive.
- If `spatial_variance` is elevated and `spatial_consistency` is lower, treat the output as more localized or less uniform across the nearby area.
- If `spatial_gradient` or `edge_signal` is large, the strongest signal is nearby rather than centered exactly over the observer.

Current spatial behavior:

- Base radii remain phenomenon-specific, but the effective `radius_km` is adaptive and scales with `cloud_variability` and `wind_shear_250`.
- All phenomena use the same weighted-blend aggregation after distance weighting.
- `rainbow`, `crepuscular_rays`, `sun_pillar`, `moonbow`, and `lunar_pillar` also apply directional weighting using the active source azimuth.

Practical guidance:

- For alerts, keep using `peak.probability` or `current.probability` as the main trigger, and use `spatial_context` as supporting evidence.
- For a map or dashboard, `center_probability` is the cleanest "right here" number, while `mean_probability` and `max_probability` show whether the surrounding area is weaker or stronger.
- For brief human-readable summaries, mention when `max_probability` is much higher than `center_probability`; that usually indicates the phenomenon is nearby rather than centered exactly over the observer.

## API Usage

If `fastapi` is not installed, the repo includes a standard-library WSGI server:

```bash
python3 api/main.py
```

Then request:

```bash
curl "http://127.0.0.1:8000/predict?lat=32.8&lon=-96.8&mode=forecast"
```

Optional API query parameters:

- `at_time=2026-04-13T18:00:00Z`
- `illumination=lunar`
- `time_window_hours=0,1,2,3`
- `phenomena=halo,fogbow`
- `spatial_resolution_km=10`
- `lightweight=true`
- `debug=true`

If `fastapi` and `uvicorn` are installed, run:

```bash
uvicorn api.main:app --reload
```

## Data Source Behavior

Forecast mode:

- NOAA NOMADS GFS
- newest cycle attempted first
- older cycles used automatically if the newest file is not published yet
- short-horizon timeline slots re-request forecast fields at the target hour when possible
- each time slot is evaluated across an adaptive 3x3 km-based grid before weighted aggregation

Observed mode:

- NOAA GOES cloud-layer products for cirrus and high-cloud conditions
- NOAA GOES cloud optical depth products when available
- Aviation Weather Center METAR observations for precipitation and cloud layers
- NOAA/NWS station discovery for locating nearby observing stations
- NOAA GFS fallback for upper-air, condensate, and other unavailable observed-mode inputs
- observed-mode timelines reuse the spatially sampled weather field and evolve only solar geometry through the horizon

Fallback behavior:

- If GOES cloud optical depth is unavailable, the predictor falls back to a hybrid optical-thickness proxy built from GFS condensate support and `humidity_250`.
- If both GOES optical depth and GFS condensate support are unavailable, cloud optical thickness falls back to `cirrus_coverage * 0.5`.
- Observed-mode time-window output uses persistence for the weather fields and updates solar geometry across the horizon.
- If the latest GFS cycle is unavailable, forecast mode falls back to older cycles quietly before failing.
- Confidence now combines explicit data, feature, spatial, and temporal components, with source quality folded into the data term.
- Lunar mode uses the same weather field but switches the active geometry and visibility source to the Moon, including moon-phase and sky-darkness scaling.

## Model Summary

The implementation uses quantitative rule-based models in `models/`.

Each probability is built from three smooth components:

- `PhysicalCondition`
- `Visibility`
- `Geometry`

The models use a weighted log-combination step rather than direct multiplication.

Shared implementation details:

- `temp_250` is the 250 mb temperature in Celsius.
- `humidity_250` is normalized to a 0-1 fraction.
- `cirrus_coverage` is sourced from GOES/GFS high cloud coverage and normalized to 0-1.
- `cloud_optical_thickness` uses this priority:
  GOES cloud optical depth -> hybrid fallback `0.6 * condensate_proxy + 0.4 * humidity_250` -> `cirrus_coverage * 0.5`
- `thin_cirrus = cirrus_coverage * exp(-2.0 * cloud_optical_thickness)`
- `ice_presence = 0.3 * ice_300mb + 0.4 * ice_250mb + 0.3 * ice_200mb` when multi-layer ice support is available, otherwise it falls back to the earlier thin-cirrus / ice-fraction proxy
- `plate_alignment = exp(-wind_shear_250) * exp(-vertical_velocity_variance)` when both inputs are available
- `wind_stability = exp(-wind_shear_250 - vertical_velocity_variance)`
- `cloud_variability` is derived from a 3x3 cloud-cover neighborhood when present, otherwise fallback `0.3`
- `source_visible = sigmoid(source_elevation; 1.5, -2.0)` with lunar mode further scaled by moon phase and sky darkness
- `brightness_factor = 1.0` in solar mode, and approximately `moon_phase^1.5 * sky_darkness` in lunar mode
- `precipitation` is a non-negative precipitation signal.
- `surface_visibility` and `fog_presence` are populated from METAR parsing, so they are most informative in `observed` mode.
- Predictor output also includes:
  `confidence`, a weighted blend of the explicit `confidence_components`
  `timeline`, a 0-3 hour forecast window with `now`, `1h`, `2h`, and `3h`
  nested `peak.probability` and `peak.time`, the strongest horizon slot and its earliest timestamp
  `reason`, a short explanation string derived from the strongest supporting or limiting factors
  optional `debug` terms for `P`, `V`, and `G`

Helper formulas:

```text
sigmoid(x; k, x0) = 1 / (1 + exp(-k * (x - x0)))
gaussian(x; mu, sigma) = exp(-((x - mu)^2) / (2 * sigma^2))

combine_log(P, V, G) =
  sigmoid(
      1.2 * log(clamp(P) + 1e-6)
    + 1.0 * log(clamp(V) + 1e-6)
    + 1.1 * log(clamp(G) + 1e-6)
  )
```

Where `clamp()` limits a value to the `0..1` interval.

### Phenomenon Models

`halo`

```text
P = ice_presence * thin_cirrus
V = sun_visible * exp(-cloud_optical_thickness)
G = gaussian(solar_elevation; 20.0, 20.0)
Probability = combine_log(P, V, G)
```

Rule summary: halos favor thin icy cirrus, visible sun, and a broad mid-elevation solar geometry window.

`parhelia`

```text
P = ice_presence * thin_cirrus * plate_alignment
V = sun_visible * exp(-cloud_optical_thickness)
G = gaussian(solar_elevation; 20.0, 15.0)
Probability = combine_log(P, V, G)
```

Rule summary: unlike v1, parhelia is no longer identical to halo and adds a plate-alignment factor.

`cza`

```text
P = ice_presence * thin_cirrus
V = sun_visible * exp(-cloud_optical_thickness)
G = gaussian(solar_elevation; 22.0, 5.0)
Probability = combine_log(P, V, G)
```

Rule summary: same ice-and-thin-cirrus support as halo, but with a much tighter solar-elevation window.

`circumhorizontal_arc`

```text
P = ice_presence * thin_cirrus
V = sun_visible * exp(-cloud_optical_thickness)
G = sigmoid(solar_elevation; 0.6, 58.0)
  * gaussian(solar_elevation; 66.0, 8.0)
Probability = combine_log(P, V, G)
```

Rule summary: the same thin icy cirrus support is required, but the sun must be quite high.

`upper_tangent_arc`

```text
P = ice_presence * thin_cirrus
V = sun_visible * exp(-cloud_optical_thickness)
G = gaussian(solar_elevation; 12.0, 7.0)
  * (1 - sigmoid(solar_elevation; 0.35, 32.0))
Probability = combine_log(P, V, G)
```

Rule summary: upper tangent arcs favor low sun and thin icy cirrus, with explicit suppression once the sun climbs too high.

`sun_pillar`

```text
P = ice_presence * max(thin_cirrus, humidity_250)
V = sun_visible * exp(-0.5 * cloud_optical_thickness)
G = gaussian(solar_elevation; 2.0, 4.0)
  * (1 - sigmoid(solar_elevation; 0.4, 14.0))
Probability = combine_log(P, V, G)
```

Rule summary: sun pillars emphasize near-horizon geometry and allow humid upper air to help sustain the physical term.

`crepuscular_rays`

```text
P = gaussian(cirrus_coverage; 0.5, 0.3)
  * gaussian(cloud_variability; 0.5, 0.3)
V = sun_visible * exp(-0.5 * cloud_optical_thickness)
G = gaussian(solar_elevation; 6.0, 6.0)
Probability = combine_log(P, V, G)
```

Rule summary: crepuscular rays now use explicit cloud-structure variability instead of raw optical thickness in the physical term.

`rainbow`

```text
P = sigmoid(precipitation; 5.0, 0.1)
V = sun_visible
G = gaussian(solar_elevation; 20.0, 15.0)
Probability = combine_log(P, V, G)
```

Rule summary: rainbows remain precipitation-driven and use the same weighted log combination as the other phenomenon models.

`fogbow`

```text
P = max(
      fog_presence,
      gaussian(surface_visibility; 0.8, 0.8)
    )
  * (1 - sigmoid(precipitation; 8.0, 0.6))
V = sun_visible * exp(-0.25 * cloud_optical_thickness)
G = gaussian(solar_elevation; 15.0, 12.0)
Probability = combine_log(P, V, G)
```

Rule summary: fogbows favor fog or mist, low visibility, visible sun, and only limited competing precipitation.

Practical note: `fogbow` is best supported in `observed` mode because `surface_visibility` and `fog_presence` come from METAR parsing.

Lunar reuse summary:

- `lunar_halo` reuses the halo model with lunar source geometry and brightness scaling
- `paraselenae` reuses the parhelia model with lunar source geometry and brightness scaling
- `lunar_pillar` reuses the sun-pillar model with lunar source geometry and brightness scaling
- `moonbow` reuses the rainbow model with lunar source geometry and brightness scaling
- `lunar_corona` is a lunar-specific thin-cloud diffraction model

## Alert Integration

The sibling alert project can invoke this predictor directly through the `atmospheric_optics` provider.

Example `alerts.toml` target:

```toml
[[sources]]
name = "atmospheric_optics"
provider = "atmospheric_optics"
db_file = "/home/celaeno/alert/atmospheric_optics.db"

[[sources.targets]]
name = "Home"
url = "atmospheric-optics://home"
threshold = 0.8
timeout_seconds = 180
lat = 32.847
lon = -96.806
mode = "observed"
illumination = "solar"
project_dir = "/home/celaeno/script/atmospheric_optics"
phenomena = [
  "halo",
  "parhelia",
  "cza",
  "circumhorizontal_arc",
  "upper_tangent_arc",
  "sun_pillar",
  "crepuscular_rays",
  "rainbow",
  "fogbow",
]
```

Notes:

- `threshold` is applied to each phenomenon's `peak.probability`.
- `timeout_seconds` can be raised for observed-mode targets so GOES and METAR ingestion have enough time to complete in cron jobs.
- If `phenomena` is omitted, the alert provider defaults to all supported phenomena.
- `illumination` is optional and defaults to `solar`; set it to `lunar` to run the moonlit phenomenon set.
- Optional target options `at_time` and `time_window_hours` are passed through to the CLI.
- Alert items use the phenomenon object's `peak.probability`, `peak.time`, and `current.reason`.
- Exported JSON preserves the nested per-phenomenon structure, including `spatial_context`, instead of flattening it.

Web export example:

```bash
python3 /home/celaeno/script/alert/export_atmospheric_optics_json.py \
  --config /home/celaeno/script/alert/alerts.toml \
  --source atmospheric_optics \
  --prediction-only \
  --output /home/celaeno/web/astro/table/atmospheric_optics.json
```

## Tests

Run the full suite:

```bash
pytest -q tests
```
