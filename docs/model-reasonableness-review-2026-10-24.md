# Model reasonableness review — October 24, 2026

This review follows the spatial-normalization fix and checks whether model scores respect required physical ingredients, geometric limits, and the meaning of observed weather inputs. The changes preserve the existing output schema and alert threshold.

## Physical ingredients and geometry

The [WMO circumzenithal-arc description](https://cloudatlas.wmo.int/en/circumzenithal-arc.html) limits the light-source elevation to less than 32 degrees. Its [circumhorizontal-arc description](https://cloudatlas.wmo.int/en/circumhorizontal-arc.html) requires more than 58 degrees. The previous Gaussian/sigmoid curves assigned substantial support outside those limits. Explicit geometry gates now make these invalid cases zero while retaining the existing smooth preferences inside the valid ranges.

[NOAA describes the 42-degree limit](https://www.weather.gov/fgz/Rainbow) for rainbows visible above the horizon. Rainbow, moonbow, and fogbow geometry now enforce a light-source elevation from zero to less than 42 degrees for this ground-observer model. Below-horizon bows seen from aircraft or above waterfalls are outside its scope.

Rainbow precipitation previously used a shifted logistic response with positive support at zero rain. Subtracting and normalizing its dry baseline retains a smooth increasing response while making dry or missing precipitation zero. Rainbow and moonbow visibility also now use the cloud-thickness attenuation already used by the other models. This is a proxy for attenuation, not a measurement of direct sunlight along the observer’s line of sight.

The [WMO fogbow description](https://cloudatlas.wmo.int/en/fog-bow.html) requires a screen of fog or mist droplets. Previously, missing visibility became numerical zero, and low visibility alone could supply fog support even in haze or smoke. Fog/mist evidence is now required. A measured visibility can modulate that evidence; if visibility is unavailable, an explicit fog report can still contribute.

The [WMO corona description](https://cloudatlas.wmo.int/en/corona.html) describes diffraction through small particles in thin cloud, fog, or mist, and notes other particle sources such as pollen. This implementation models the thin-cloud case. It previously produced a positive score when cloud optical thickness was absent or zero; those cases now have zero physical support. Pollen coronas and explicit particle-size modeling remain outside the available inputs.

## Weather decoding, numerical behavior, and timing

[Aviation Weather Center documentation](https://aviationweather.gov/help/data/) distinguishes present-weather groups from the station header and remarks, and distinguishes thunderstorms from precipitation. The old substring checks interpreted station `KRAL` as rain, station `KBRD` as mist, and bare `TS` as liquid precipitation. Shared METAR parsing now validates weather-group syntax, skips the header through its report time, and ignores remarks when extracting current weather. Liquid rain/drizzle requires `RA` or `DZ`; bare thunder, snow, and unspecified vicinity showers do not supply that signal. Missing weather remains missing rather than being declared dry. Structured `wxString` and combined weather groups remain supported.

The shared component combiner now returns exactly zero when any required physical, visibility, or geometry component is zero. After spatial aggregation, temporal smoothing preserves slots with zero support; a later rain event or valid arc elevation cannot introduce a currently impossible prediction.

The lunar brightness feature now honors a supplied illuminance value. When none is supplied, the existing phase-based estimate remains the fallback. Lunar sky-darkness scaling still applies. Logistic calculations also use a numerically stable form to avoid overflow on extreme finite inputs.

## Reproduced behavior

The following pre-revision scores were reproduced with favorable remaining inputs:

| Case | Previous behavior | Revised behavior |
| --- | --- | --- |
| Dry rainbow | Score about 0.311 | Zero |
| Fogbow with neither fog nor visibility evidence | Score about 0.543 | Zero |
| Clear-sky lunar corona | Score about 0.259 | Zero |
| Circumzenithal arc at 32.1 degrees | Score about 0.106 | Zero |
| Circumhorizontal arc at 57.9 degrees | Score about 0.257 | Zero |
| Dry `KRAL` METAR | Liquid-precipitation signal 1 | Signal 0 |
| Thunder without reported rain | Liquid-precipitation signal 1 | Signal 0 |

Favorable halo, arc, fogbow, and rainy/sunny cases remain capable of scoring above the existing `0.8` threshold. Tests also check that future favorable conditions do not leak into a currently gated slot.

Validation: **155 predictor tests passed, 1 skipped**, and **56 sibling alert tests passed**. The skipped test is the WSGI fallback, which is inapplicable when FastAPI is installed. A live CLI run at 10:49 PM CDT produced all 14 phenomenon entries from current GOES, GFS, and METAR inputs; the highest peak was lunar corona at `0.179`. The sibling alert provider parsed all entries and correctly found none above `0.8`. No real email was sent by these checks. Existing HDF5 attribute diagnostics were captured separately; the live run completed successfully.

## Calibration limits

These changes fix demonstrable mathematical and input-interpretation defects. The scores remain heuristic suitability estimates rather than measured event frequencies. Gaussian preferences, relative weights, precipitation accumulation versus occurrence proxies, and crystal-alignment assumptions need labeled observations for calibration. Cloud fraction and optical-depth proxies do not establish whether the Sun or Moon is visible along a particular viewing direction, and satellite or METAR data do not resolve all relevant particle shapes and sizes. Those limitations are documented instead of fitting new constants without outcome data.
