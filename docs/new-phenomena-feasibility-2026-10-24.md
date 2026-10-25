# Additional forecast phenomena — feasibility review, October 24, 2026

**Implementation update:** solar corona, anticrepuscular rays, and secondary rainbows are now registered forecasts, bringing the combined output to 17 entries. See the [implementation guide](new-phenomena-implementation-2026-10-24.md). The assessment below records the investigation that preceded implementation.

The recommendations below are engineering judgments based on the present code and the cited physical descriptions. Being able to compute a suitability score does not establish that the score is a calibrated event probability.

## Current capabilities

The predictor already has solar/lunar elevation and azimuth, cloud optical-depth estimates, spatial cloud coverage and variability, precipitation/fog evidence, upper-air ice proxies, and wind-stability proxies. It evaluates a local spatial neighborhood and a horizon of up to three hours. Observed mode persists the current weather field and updates the illumination geometry; forecast mode can request GFS fields at the future times.

The current pipeline does not ingest cloud effective particle radius or particle-size dispersion, a directional horizon mask, observer elevation relative to surrounding clouds, a detailed near-surface refractive profile, or mesospheric conditions. These are concrete limitations when distinguishing some otherwise similar optical phenomena.

## Candidate assessment

| Candidate / proposed ID | Fit with current inputs | Main limitation | Assessment |
| --- | --- | --- | --- |
| Solar corona / `solar_corona` | Strong: reuse the existing lunar-corona diffraction core with solar visibility and daytime geometry. | Cloud optical depth/variability do not directly measure particle size or uniformity. | First experimental addition; a new weather feed is not required for an initial suitability model. |
| Anticrepuscular rays / `anticrepuscular_rays` | Moderate: low-Sun geometry, structured cloud, solar azimuth, and spatial sampling already exist. | Requires treating the cloud that blocks sunlight separately from visibility toward the opposite horizon; local cloud data do not resolve the entire light path. | Useful follow-on, with explicit directional modeling and experimental confidence. |
| Secondary rainbow / `secondary_rainbow` | Moderate: precipitation, source visibility, cloud attenuation, and antisolar sampling already exist. | Needs its own angular range and a brightness/contrast treatment. Its visibility cannot be obtained by copying the primary score. | Useful follow-on after refactoring shared bow ingredients. |
| Cloud iridescence / `cloud_iridescence` | Partial: thin-cloud and solar inputs provide broad supporting conditions. | Current data cannot reliably distinguish iridescence from a corona or measure the relevant particle-size structure. | Experimental diffraction watch, or defer a separate event score until particle-size inputs are available. |
| Earth's shadow / Belt of Venus | Good for a viewing-time window: solar geometry and cloud proxies are available. | Horizon obstruction, haze, and illumination/contrast along the viewing direction remain uncertain. | Add as a twilight viewing-window product rather than presenting geometry alone as an occurrence probability. |
| 46-degree halo / parhelic circle | Ice-bearing-cloud and alignment proxies provide general support. | These fields do not reliably distinguish the crystal configurations producing the separate halo forms. | Defer separate forecasts until observations or better crystal-habit evidence support differentiation. |
| Glory / Brocken spectre | Partial: cloud/fog evidence is available. | Needs observer position relative to cloud, an elevated/downward view, and appropriate viewing geometry. | Requires additional request inputs and terrain/cloud-relative geometry. |
| Green flash | Solar rise/set timing can be calculated. | Needs a clear horizon and a near-surface temperature/refractive structure; the event is much shorter than the current hourly timeline. | Defer an event forecast; sunrise/sunset timing alone is insufficient. |
| Noctilucent / nacreous clouds | The illumination window can be calculated. | Formation depends on upper-atmospheric conditions outside the ingested fields. | Defer until appropriate mesospheric or stratospheric data are available. |

## Physical basis and implications

**Solar corona.** WMO describes [coronae around either the Sun or Moon](https://cloudatlas.wmo.int/en/corona.html), produced by diffraction through small particles in thin cloud, mist, or fog, with particle size affecting the ring structure. An initial solar counterpart is therefore a reasonable extension of the existing lunar model. It should share the cloud physics, use solar source visibility, exclude a below-horizon Sun, and omit lunar phase/darkness factors. A solar corona is distinct from the existing ice-refraction `halo` forecast.

**Anticrepuscular rays.** NOAA documents [rays produced by low-angle sunlight blocked by clouds and seen toward the opposite horizon](https://www.weather.gov/lmk/AntiCrepuscularRaysThisMorning). The viewing azimuth should be `(solar_azimuth + 180) % 360`. Simply reversing every cloud weight would conflate the shadow-producing obstruction with the viewing direction. A useful model should combine source-side cloud/shadow support with opposite-side viewing/scattering support; the latter is only coarsely approximated by current inputs.

**Secondary rainbow.** WMO describes a [fainter, reversed-color secondary bow](https://cloudatlas.wmo.int/secondary-bow.html), with the red edge at about 51 degrees and the violet edge at about 54 degrees. The primary model's 42-degree cutoff must not be reused. At some source elevations above 42 degrees a secondary arc can remain above the horizon while the primary does not. Shared rain/illumination terms should be extracted below the primary geometry gate, with separate secondary geometry and a documented treatment of reduced visibility. An arbitrary constant multiplier would not supply empirical calibration.

**Iridescence.** WMO describes [cloud iridescence](https://cloudatlas.wmo.int/en/irisation-or-iridescence.html) over a range of angular separations from the Sun. Current spatial cloud variability is not a particle-size-distribution measurement. Reusing the corona score under another name would not justify an independent, distinguishable forecast.

**Twilight bands.** NOAA describes [Earth's shadow and the Belt of Venus](https://www.weather.gov/owlie/weird-weather) near the horizon opposite the rising or setting Sun. A geometry-based watch window is feasible, but its weather-dependent appearance and intensity remain uncertain. This should be distinguished in the output from a measured or calibrated likelihood.

**More data-intensive candidates.** WMO's [glory description](https://cloudatlas.wmo.int/en/glory.html) identifies the elevated observer/cloud-relative geometry. Its [green-flash description](https://cloudatlas.wmo.int/en/green-flash.html) identifies clear-horizon and temperature-structure requirements and very short durations. WMO places [noctilucent and polar stratospheric clouds](https://cloudatlas.wmo.int/en/upper-atmospheric-clouds.html) in atmospheric regions beyond the current upper-air fields. These requirements support deferring those event forecasts. The distinct geometry and appearance of [46-degree halos and supralateral arcs](https://cloudatlas.wmo.int/en/differentiation-characteristics-of-the-46-degree-halo-and-Supralateral-arc.html) also illustrate why one general ice-cloud score cannot reliably classify every halo form.

## Solar-corona feasibility probe

An offline probe reused `predict_lunar_corona` with features computed for solar illumination, plus an explicit proposed above-horizon gate. It exercised model reuse without registering a new forecast:

| Input case | Reused-core suitability score |
| --- | --- |
| Daylight, optical thickness 0.3, variability 0.3 | 0.914 |
| Clear sky, optical thickness 0 | 0.000 |
| Thick cloud, optical thickness 0.9 | 0.003 |
| Missing cloud evidence | 0.000 |
| Below-horizon Sun, explicit proposed gate | 0.000 |

These synthetic cases confirm that the existing components can provide the desired qualitative gates and a usable score range. They do not measure how frequently real solar coronas occur, nor independently validate particle-size assumptions.

## Concrete implementation path

1. Extract a shared corona model from `optics_models/lunar.py`, retaining the existing lunar entry point for compatibility. Add a solar entry point with an above-horizon source gate and solar brightness/visibility inputs.
2. Register `solar_corona` in the solar list, metadata, feature requirements, model mapping, and explanation logic in `prediction/pipeline.py`. Add an explicit spatial radius in `prediction/spatial_sampling.py`, initially matching the lunar counterpart as a documented heuristic.
3. Add selection, combined-illumination, clear/missing-cloud, nighttime, and temporal-gate regressions. Check that the lunar model retains its behavior.
4. Verify CLI/API payloads and the sibling alert provider with a preview notifier. The current schema already supports additional phenomenon objects. The alert provider accepts payload-defined solar IDs; explicit configured phenomenon lists would need the new ID, while omitted lists follow all available phenomena. Account for that behavior during rollout.
5. Collect timestamped outcomes and input snapshots before treating the score as a calibrated probability or setting candidate-specific alert thresholds.

The next data improvement worth evaluating is NOAA's [GOES cloud effective particle-size retrieval](https://www.goes-r.gov/products/baseline-cloud-particle-size-dist.html), which is retrieved alongside optical depth. Its [product guide](https://www.goes-r.gov/users/docs/PUG-L2%2B-vol5.pdf) describes effective radius and quality flags, including degraded conditions near the terminator. This is a candidate additional input for diffraction models, not a substitute for an observed particle-size distribution or outcome validation. Its operational availability and retrieval quality would need checking during implementation.

The initial investigation was completed as a feasibility assessment. Its first three candidates have subsequently been implemented; the remaining candidates retain the data and validation limitations described above.
