# New forecast implementation — October 24, 2026

The predictor now supports `solar_corona`, `anticrepuscular_rays`, and `secondary_rainbow` in forecast and observed modes. Combined solar/lunar output contains 17 phenomena; lunar-only selection remains unchanged. Existing output fields are retained, and new fields are additive.

| Forecast | Model and gates | Viewing direction |
| --- | --- | --- |
| Solar corona | Shared thin-cloud diffraction core with the lunar corona. Positive optical-depth evidence, visible sunlight, and an above-horizon Sun are required. Lunar phase and sky-darkness factors are not used. | Solar azimuth |
| Anticrepuscular rays | Source-side cloud and gaps provide shadow support; opposite-side cloud clearance supplies viewing support. Low-Sun geometry is favored. Missing directional evidence, entirely clear/overcast source fields, an obscured opposite field, and deep night give zero support. | Solar azimuth + 180 degrees |
| Secondary rainbow | Shared rain and illumination terms, independent geometry through the 51–54 degree color band, with a taper as the remaining visible colors disappear above 51 degrees. Dry conditions, absent illumination, night, and a source elevation of 54 degrees or more give zero support. | Solar azimuth + 180 degrees |

## Data additions

The new cloud-shadow model needs total cloud coverage rather than only high cirrus. Existing NOAA feeds supply it:

- GOES `TCF` is read from the cloud-layer file already downloaded. It uses netCDF automatic scaling to fractions in `0..1`; this differs from the percentage-valued `CF1`–`CF5` layers. Masked and invalid samples are excluded. No additional GOES product download is needed for this field.
- GFS requests now include `TCDC` at the entire-atmosphere level. Percentages are converted to fractions, and neighboring samples are retained. A live request confirmed that the field is returned.
- Observed mode prefers GOES total coverage, then METAR layer/clear-sky evidence, then GFS total coverage. Missing observations remain unavailable. The maximum reported METAR layer amount is a total-coverage proxy rather than an independent measurement of the union of all layers.

The weather payload gains `cloud_cover_total` and `total_cloud_cover_grid`. GOES reads are serialized inside the predictor process to avoid overlapping netCDF file access from spatial sampling threads. Total-cloud input is attributed to GOES even when high-cloud coverage is unavailable.

For anticrepuscular rays, noncentral grid samples are separated into sunward and antisolar hemispheres using the solar azimuth. Projection-weighted mean cloud fractions are computed for each side. The local center is not counted as evidence for either horizon. Reversing the cloud pattern therefore changes the score even if the neighborhood's overall mean is identical. A single center-only point cannot establish both sides, so `--lightweight` reports zero support and explains the missing directional evidence.

## Model assumptions

[WMO describes coronae around either the Sun or Moon](https://cloudatlas.wmo.int/en/corona.html), supporting a shared diffraction model with separate illumination handling. Existing `LunarCoronaParameters`, `DEFAULT_LUNAR_CORONA_PARAMETERS`, and `predict_lunar_corona` imports remain available. Solar and lunar wrappers explicitly select their own source visibility, preventing sunlight from substituting for moonlight or the reverse.

[NOAA describes anticrepuscular rays](https://www.weather.gov/lmk/AntiCrepuscularRaysThisMorning) as low-angle sunlight blocked by clouds and seen on the opposite horizon. This model distinguishes source-side obstruction from viewing-side clearance. Its normalized shadow-pattern support is `4 * source_cloud_fraction * (1 - source_cloud_fraction)`. Visibility includes opposite-side clearance and the existing optical-thickness attenuation proxy. Local samples do not resolve the full cloud-shadow path, atmospheric scattering, or terrain horizon.

[WMO's secondary-bow description](https://cloudatlas.wmo.int/secondary-bow.html) gives reversed colors and a larger angular range than the primary bow. The new model does not inherit the primary 42-degree cutoff: at a 45-degree source elevation, primary above-horizon support is zero while secondary support can remain positive. The model's visible-band proxy tapers between 51 and 54 degrees. The visibility exponent is `2.0` instead of the primary model's `1.0`, making weaker illumination/contrast more limiting without multiplying every result by a fixed low brightness factor. This exponent and the existing Gaussian preferences are heuristic choices, not fitted occurrence probabilities or a complete radiative-transfer calculation.

Cloud optical depth and variability remain proxies for diffraction conditions. Effective particle-size data are not required to run this initial model, but would be a useful future input after quality assessment and calibration against labeled observations.

## Use and integration

```bash
python3 cli/command.py --lat 32.847 --lon -96.806 --mode observed \
  --illumination solar \
  --phenomena solar_corona,anticrepuscular_rays,secondary_rainbow --debug
```

The HTTP API accepts the same comma-separated IDs through `phenomena`. New entries include `current.viewing_azimuth`, measured clockwise from north; anticrepuscular-ray spatial context adds the available `source_cloud_fraction` and `antisolar_cloud_fraction` values. Explanations describe missing directional evidence, nighttime gates, and poor visibility.

The sibling alert provider already consumes payload-defined phenomenon IDs. With no explicit `phenomena` list, it follows all supported output and includes these additions. Explicit lists must include the desired new IDs. No SMTP or history changes are required.

## Validation

The predictor suite passes **178 tests, with 1 skipped**; the skip is the existing WSGI fallback test when FastAPI is installed. The sibling alert suite passes **56 tests**. Coverage includes geometry boundaries, lunar compatibility, separate solar visibility, directional cloud reversal and rotation, missing horizon samples, real packed-netCDF scaling, fallback priority, source attribution, CLI/API selection, and finite JSON output.

The live CLI produced all 17 entries from current GOES, GFS, and METAR data. At the nighttime check, all three new solar forecasts correctly scored zero. Library HDF5 attribute diagnostics were captured separately; the live process completed successfully.

Controlled end-to-end cases, using actual feature/model/spatial/temporal processing with fixed weather and source positions, gave:

| Favorable case | Peak suitability |
| --- | --- |
| Solar corona | 0.914 |
| Secondary rainbow | 0.980 |
| Anticrepuscular rays | 0.975 |

Each corresponding payload triggered one alert item at the existing `0.8` threshold through the real sibling alert runner, then deduplicated on repetition. Validation used temporary SQLite history and a preview notifier: three preview messages, no real email, and no changes to production history. The scores illustrate model behavior and reachability; they are not observed event frequencies.
