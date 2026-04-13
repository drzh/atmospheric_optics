"""End-to-end predictor orchestration."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Callable, Iterable

from core.spatial import (
    SpatialSample,
    adaptive_radius,
    aggregate_probabilities,
    aggregation_for,
    apply_spatial_weights,
    generate_samples,
    radius_for,
    spatial_context as build_spatial_context,
)
from core.temporal import (
    PEAK_PRESERVATION_THRESHOLD,
    resolve_peak_index,
    smooth_probabilities,
    temporal_consistency,
    temporal_stability,
)
from data_ingestion.weather import (
    SourceAttribution,
    WeatherSnapshot,
    cleanup_cached_artifacts,
    get_weather_snapshot,
)
from feature_engineering.features import compute_features
from models.combine import ModelComponents
from models.cza_model import predict_cza
from models.halo_model import predict_halo, predict_parhelia
from models.ice_crystal_model import (
    predict_circumhorizontal_arc,
    predict_sun_pillar,
    predict_upper_tangent_arc,
)
from models.rainbow_model import predict_rainbow
from models.scattering_model import predict_crepuscular_rays, predict_fogbow
from solar.solar_position import get_solar_position

OUTPUT_DECIMALS = 3
MAX_TIME_WINDOW_HOURS = 3
TIME_WINDOW_HOURS: tuple[int, ...] = (0, 1, 2, 3)
PredictionFunction = Callable[[dict[str, float]], float]
PointKey = tuple[float, float]
PHENOMENA: tuple[str, ...] = (
    "halo",
    "parhelia",
    "cza",
    "circumhorizontal_arc",
    "upper_tangent_arc",
    "sun_pillar",
    "crepuscular_rays",
    "rainbow",
    "fogbow",
)
PHENOMENON_METADATA: dict[str, dict[str, str]] = {
    "halo": {"label": "Halo", "category": "ice_crystal"},
    "parhelia": {"label": "Parhelia", "category": "ice_crystal"},
    "cza": {"label": "Circumzenithal Arc", "category": "ice_crystal"},
    "circumhorizontal_arc": {"label": "Circumhorizontal Arc", "category": "ice_crystal"},
    "upper_tangent_arc": {"label": "Upper Tangent Arc", "category": "ice_crystal"},
    "sun_pillar": {"label": "Sun Pillar", "category": "ice_crystal"},
    "crepuscular_rays": {"label": "Crepuscular Rays", "category": "cloud_shadow"},
    "rainbow": {"label": "Rainbow", "category": "water_droplet"},
    "fogbow": {"label": "Fogbow", "category": "water_droplet"},
}
SOURCE_METADATA: dict[str, dict[str, str]] = {
    "goes-east": {"label": "GOES East", "kind": "satellite"},
    "goes-west": {"label": "GOES West", "kind": "satellite"},
    "goes": {"label": "GOES", "kind": "satellite"},
    "metar": {"label": "METAR", "kind": "surface_observation"},
    "gfs": {"label": "GFS", "kind": "forecast_model"},
}
PHENOMENON_FEATURES: dict[str, tuple[str, ...]] = {
    "halo": ("ice_presence", "thin_cirrus", "sun_visible", "cloud_optical_thickness", "solar_elevation"),
    "parhelia": (
        "ice_presence",
        "thin_cirrus",
        "plate_alignment",
        "wind_stability",
        "sun_visible",
        "cloud_optical_thickness",
        "solar_elevation",
    ),
    "cza": ("ice_presence", "thin_cirrus", "sun_visible", "cloud_optical_thickness", "solar_elevation"),
    "circumhorizontal_arc": ("ice_presence", "thin_cirrus", "sun_visible", "cloud_optical_thickness", "solar_elevation"),
    "upper_tangent_arc": ("ice_presence", "thin_cirrus", "sun_visible", "cloud_optical_thickness", "solar_elevation"),
    "sun_pillar": (
        "ice_presence",
        "thin_cirrus",
        "humidity_250",
        "wind_stability",
        "sun_visible",
        "cloud_optical_thickness",
        "solar_elevation",
    ),
    "crepuscular_rays": ("cirrus_coverage", "cloud_variability", "sun_visible", "cloud_optical_thickness", "solar_elevation"),
    "rainbow": ("precipitation", "sun_visible", "solar_elevation"),
    "fogbow": ("fog_presence", "surface_visibility", "precipitation", "sun_visible", "cloud_optical_thickness", "solar_elevation"),
}


@dataclass
class PredictorCaches:
    snapshot_cache: dict[tuple[float, float, str, str], WeatherSnapshot] = field(default_factory=dict)
    point_cache: dict[tuple[float, float, str, str], "PointEvaluation"] = field(default_factory=dict)
    snapshot_lock: Lock = field(default_factory=Lock)
    point_lock: Lock = field(default_factory=Lock)


@dataclass(frozen=True)
class PointEvaluation:
    weather: dict[str, object]
    sources: tuple[SourceAttribution, ...]
    features: dict[str, float]
    probabilities: dict[str, float]
    debug_components: dict[str, ModelComponents]


@dataclass(frozen=True)
class PhenomenonTimeEvaluation:
    probability: float
    spatial_context: dict[str, float | str]
    reason_features: dict[str, float]
    confidence_components: dict[str, float]
    debug: dict[str, float]


@dataclass(frozen=True)
class TimeSlotEvaluation:
    label: str
    target_time: datetime
    phenomena: dict[str, PhenomenonTimeEvaluation]


def predict_all(
    lat: float,
    lon: float,
    at_time: datetime | None = None,
    mode: str = "forecast",
    keep_downloaded_files: bool = False,
    download_dir: str | Path | None = None,
    time_window_hours: Iterable[int] | None = None,
    spatial_resolution_km: float | None = None,
    lightweight: bool = False,
    debug: bool = False,
    phenomena: Iterable[str] | None = None,
) -> dict[str, object]:
    """Predict all supported atmospheric optics probabilities."""

    generated_at = datetime.now(timezone.utc)
    prediction_time = _resolve_prediction_time(at_time)
    resolved_time_window_hours = _normalize_time_window_hours(time_window_hours)
    resolved_spatial_resolution_km = _normalize_spatial_resolution_km(spatial_resolution_km)
    selected_phenomena = _normalize_selected_phenomena(phenomena)
    predictors = _prediction_functions()
    caches = PredictorCaches()
    center_snapshot = _load_weather_snapshot(
        lat,
        lon,
        mode=mode,
        target_time=prediction_time,
        prediction_time=prediction_time,
        keep_downloaded_files=keep_downloaded_files,
        download_dir=download_dir,
        caches=caches,
    )
    time_slot_evaluations = [
        _evaluate_time_slot(
            lat,
            lon,
            label=_time_window_label(hour_offset),
            target_time=prediction_time + timedelta(hours=hour_offset),
            mode=mode,
            prediction_time=prediction_time,
            keep_downloaded_files=keep_downloaded_files,
            download_dir=download_dir,
            predictors=predictors,
            caches=caches,
            spatial_resolution_km=resolved_spatial_resolution_km,
            lightweight=lightweight,
            include_debug=debug,
        )
        for hour_offset in resolved_time_window_hours
    ]
    request_payload: dict[str, object] = {
        "location": {
            "lat": _round_output_float(lat),
            "lon": _round_output_float(lon),
        },
        "mode": mode,
        "prediction_time": prediction_time.isoformat().replace("+00:00", "Z"),
        "time_window_hours": list(resolved_time_window_hours),
        "options": {
            "lightweight": bool(lightweight),
            "debug": bool(debug),
        },
    }
    if resolved_spatial_resolution_km is not None:
        request_payload["options"]["spatial_resolution_km"] = _round_output_float(resolved_spatial_resolution_km)
    if tuple(selected_phenomena) != PHENOMENA:
        request_payload["options"]["phenomena"] = list(selected_phenomena)

    result: dict[str, object] = {
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        "request": request_payload,
        "sources": [_source_payload(source) for source in center_snapshot.sources],
        "phenomena": [],
    }

    for phenomenon in selected_phenomena:
        raw_timeline = [
            time_slot.phenomena[phenomenon].probability
            for time_slot in time_slot_evaluations
        ]
        smoothed_timeline = smooth_probabilities(
            resolved_time_window_hours,
            raw_timeline,
            preserve_peak=True,
            preserve_peak_threshold=PEAK_PRESERVATION_THRESHOLD,
        )
        timeline_stability = temporal_stability(resolved_time_window_hours, raw_timeline)
        timeline_consistency = temporal_consistency(raw_timeline)
        temporal_component = _clamp_unit_interval((0.6 * timeline_stability) + (0.4 * timeline_consistency))
        peak_index = resolve_peak_index(raw_timeline, smoothed_timeline) or 0
        current_evaluation = time_slot_evaluations[0].phenomena[phenomenon]
        current_probability = _round_output_float(smoothed_timeline[0])
        phenomenon_timeline = {
            time_slot.label: _round_output_float(smoothed_timeline[index])
            for index, time_slot in enumerate(time_slot_evaluations)
        }
        peak_probability_value = smoothed_timeline[peak_index]
        if raw_timeline[peak_index] > PEAK_PRESERVATION_THRESHOLD:
            peak_probability_value = max(peak_probability_value, raw_timeline[peak_index])
        peak_probability = _round_output_float(peak_probability_value)
        peak_time = time_slot_evaluations[peak_index].target_time.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        confidence_components = {
            "data": current_evaluation.confidence_components["data"],
            "spatial": current_evaluation.confidence_components["spatial"],
            "temporal": temporal_component,
            "feature": current_evaluation.confidence_components["feature"],
        }
        confidence = _round_output_float(_compose_confidence(confidence_components))
        current_payload: dict[str, object] = {
            "probability": current_probability,
            "confidence": confidence,
            "confidence_components": _round_numeric_mapping(confidence_components),
            "reason": _build_reason(phenomenon, current_evaluation.reason_features, current_probability),
            "spatial_context": _round_numeric_mapping(current_evaluation.spatial_context),
        }
        if debug and current_evaluation.debug:
            current_payload["debug"] = _round_numeric_mapping(current_evaluation.debug)
        result["phenomena"].append(
            {
                "id": phenomenon,
                "label": PHENOMENON_METADATA[phenomenon]["label"],
                "category": PHENOMENON_METADATA[phenomenon]["category"],
                "current": current_payload,
                "peak": {
                    "probability": peak_probability,
                    "time": peak_time,
                },
                "timeline": [
                    {
                        "label": time_slot.label,
                        "offset_hours": resolved_time_window_hours[index],
                        "probability": phenomenon_timeline[time_slot.label],
                    }
                    for index, time_slot in enumerate(time_slot_evaluations)
                ],
            }
        )

    cleanup_cached_artifacts(
        lat=lat,
        lon=lon,
        mode=mode,
        keep_downloaded_files=keep_downloaded_files,
        download_dir=download_dir,
        prediction_time=prediction_time,
        time_window_hours=resolved_time_window_hours,
    )

    return result


def _resolve_prediction_time(at_time: datetime | None) -> datetime:
    if at_time is None:
        return datetime.now(timezone.utc)
    if at_time.tzinfo is None:
        return at_time.replace(tzinfo=timezone.utc)
    return at_time.astimezone(timezone.utc)


def _round_output_float(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(value, OUTPUT_DECIMALS)


def _round_numeric_mapping(values: dict[str, float | str]) -> dict[str, float | str]:
    rounded: dict[str, float | str] = {}
    for key, value in values.items():
        if isinstance(value, str):
            rounded[key] = value
            continue
        try:
            rounded[key] = _round_output_float(float(value))
        except (TypeError, ValueError):
            continue
    return rounded


def _normalize_time_window_hours(time_window_hours: Iterable[int] | None) -> tuple[int, ...]:
    if time_window_hours is None:
        return TIME_WINDOW_HOURS

    normalized_hours = {0}
    for value in time_window_hours:
        try:
            hour = int(value)
        except (TypeError, ValueError):
            continue
        if hour < 0 or hour > MAX_TIME_WINDOW_HOURS:
            continue
        normalized_hours.add(hour)

    return tuple(sorted(normalized_hours))


def _normalize_spatial_resolution_km(spatial_resolution_km: float | None) -> float | None:
    if spatial_resolution_km is None:
        return None
    try:
        numeric_value = float(spatial_resolution_km)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value) or numeric_value <= 0.0:
        return None
    return numeric_value


def _prediction_functions() -> dict[str, PredictionFunction]:
    return {
        "halo": predict_halo,
        "parhelia": predict_parhelia,
        "cza": predict_cza,
        "circumhorizontal_arc": predict_circumhorizontal_arc,
        "upper_tangent_arc": predict_upper_tangent_arc,
        "sun_pillar": predict_sun_pillar,
        "crepuscular_rays": predict_crepuscular_rays,
        "rainbow": predict_rainbow,
        "fogbow": predict_fogbow,
    }


def _normalize_selected_phenomena(phenomena: Iterable[str] | None) -> tuple[str, ...]:
    if phenomena is None:
        return PHENOMENA

    selected: list[str] = []
    invalid: list[str] = []
    for value in phenomena:
        name = str(value).strip().lower()
        if not name:
            continue
        if name not in PHENOMENA:
            invalid.append(name)
            continue
        if name not in selected:
            selected.append(name)
    if invalid:
        raise ValueError("Unsupported phenomena: " + ", ".join(sorted(invalid)))
    if not selected:
        raise ValueError("At least one phenomenon must be selected when phenomena is provided.")
    return tuple(selected)


def _source_payload(source: SourceAttribution) -> dict[str, str]:
    metadata = SOURCE_METADATA.get(source.name, {})
    label = metadata.get("label", source.name.replace("-", " ").title())
    kind = metadata.get("kind", "other")
    return {
        "id": source.name,
        "label": label,
        "kind": kind,
        "timestamp": source.timestamp,
    }


def _point_key(lat: float, lon: float) -> PointKey:
    return (round(lat, 4), round(lon, 4))


def _evaluate_time_slot(
    lat: float,
    lon: float,
    *,
    label: str,
    target_time: datetime,
    mode: str,
    prediction_time: datetime,
    keep_downloaded_files: bool,
    download_dir: str | Path | None,
    predictors: dict[str, PredictionFunction],
    caches: PredictorCaches,
    spatial_resolution_km: float | None,
    lightweight: bool,
    include_debug: bool,
) -> TimeSlotEvaluation:
    center_point_evaluation = _evaluate_point(
        lat,
        lon,
        target_time=target_time,
        mode=mode,
        prediction_time=prediction_time,
        keep_downloaded_files=keep_downloaded_files,
        download_dir=download_dir,
        predictors=predictors,
        caches=caches,
        include_debug=include_debug,
    )
    samples_by_phenomenon, radii_by_phenomenon = _samples_by_phenomenon(
        lat,
        lon,
        center_features=center_point_evaluation.features,
        spatial_resolution_km=spatial_resolution_km,
        lightweight=lightweight,
    )
    unique_points = _unique_sample_points(samples_by_phenomenon)
    center_key = _point_key(lat, lon)
    point_evaluations: dict[PointKey, PointEvaluation] = {
        center_key: center_point_evaluation,
    }

    remaining_points = [
        (point_key, point_lat, point_lon)
        for point_key, (point_lat, point_lon) in unique_points.items()
        if point_key != center_key
    ]
    if remaining_points:
        if len(remaining_points) > 1:
            with ThreadPoolExecutor(max_workers=min(len(remaining_points), 9)) as executor:
                future_map = {
                    executor.submit(
                        _evaluate_point,
                        point_lat,
                        point_lon,
                        target_time=target_time,
                        mode=mode,
                        prediction_time=prediction_time,
                        keep_downloaded_files=keep_downloaded_files,
                        download_dir=download_dir,
                        predictors=predictors,
                        caches=caches,
                        include_debug=include_debug,
                    ): point_key
                    for point_key, point_lat, point_lon in remaining_points
                }
                for future, point_key in future_map.items():
                    point_evaluations[point_key] = future.result()
        else:
            point_key, point_lat, point_lon = remaining_points[0]
            point_evaluations[point_key] = _evaluate_point(
                point_lat,
                point_lon,
                target_time=target_time,
                mode=mode,
                prediction_time=prediction_time,
                keep_downloaded_files=keep_downloaded_files,
                download_dir=download_dir,
                predictors=predictors,
                caches=caches,
                include_debug=include_debug,
            )

    solar_azimuth = float(center_point_evaluation.features.get("solar_azimuth", math.nan))
    center_source_names = tuple(source.name for source in center_point_evaluation.sources)
    phenomenon_evaluations: dict[str, PhenomenonTimeEvaluation] = {}
    for phenomenon in PHENOMENA:
        samples = samples_by_phenomenon[phenomenon]
        grid_evaluations = [
            point_evaluations[_point_key(sample.lat, sample.lon)]
            for sample in samples
        ]
        raw_probabilities = [
            point_evaluation.probabilities[phenomenon]
            for point_evaluation in grid_evaluations
        ]
        weighted_probabilities = apply_spatial_weights(
            phenomenon,
            raw_probabilities,
            samples,
            radius_km=radii_by_phenomenon[phenomenon],
            solar_azimuth=solar_azimuth,
        )
        spatial_context = build_spatial_context(
            phenomenon,
            weighted_probabilities,
            radius_km=radii_by_phenomenon[phenomenon],
            aggregation=aggregation_for(phenomenon),
        )
        center_evaluation = _center_point_evaluation(grid_evaluations)
        confidence_components = _base_confidence_components(
            phenomenon,
            features=center_evaluation.features,
            weather=center_evaluation.weather,
            source_names=center_source_names,
            predictor=predictors[phenomenon],
            spatial_consistency=float(spatial_context.get("spatial_consistency", 0.0)),
        )
        phenomenon_evaluations[phenomenon] = PhenomenonTimeEvaluation(
            probability=aggregate_probabilities(weighted_probabilities, aggregation_for(phenomenon)),
            spatial_context=spatial_context,
            reason_features=dict(center_evaluation.features),
            confidence_components=confidence_components,
            debug=_debug_payload(center_evaluation.debug_components.get(phenomenon)),
        )

    return TimeSlotEvaluation(
        label=label,
        target_time=target_time,
        phenomena=phenomenon_evaluations,
    )


def _samples_by_phenomenon(
    lat: float,
    lon: float,
    *,
    center_features: dict[str, float],
    spatial_resolution_km: float | None,
    lightweight: bool,
) -> tuple[dict[str, list[SpatialSample]], dict[str, float]]:
    samples_by_phenomenon: dict[str, list[SpatialSample]] = {}
    radii_by_phenomenon: dict[str, float] = {}
    cloud_variability = _numeric_or_nan(center_features.get("cloud_variability"))
    wind_shear = _numeric_or_nan(center_features.get("wind_shear_250"))
    center_sample = [SpatialSample(lat=round(lat, 6), lon=round(lon, 6), dx_km=0.0, dy_km=0.0, distance_km=0.0)]
    for phenomenon in PHENOMENA:
        radius_km = adaptive_radius(radius_for(phenomenon), cloud_variability, wind_shear)
        radii_by_phenomenon[phenomenon] = radius_km
        if lightweight:
            samples_by_phenomenon[phenomenon] = center_sample
            continue
        samples_by_phenomenon[phenomenon] = generate_samples(
            lat,
            lon,
            radius_km,
            spatial_resolution_km=spatial_resolution_km,
        )
    return samples_by_phenomenon, radii_by_phenomenon


def _unique_sample_points(
    samples_by_phenomenon: dict[str, list[SpatialSample]],
) -> dict[PointKey, tuple[float, float]]:
    unique_points: dict[PointKey, tuple[float, float]] = {}
    for samples in samples_by_phenomenon.values():
        for sample in samples:
            unique_points.setdefault(_point_key(sample.lat, sample.lon), (sample.lat, sample.lon))
    return unique_points


def _center_point_evaluation(grid_evaluations: list[PointEvaluation]) -> PointEvaluation:
    if len(grid_evaluations) > 4:
        return grid_evaluations[4]
    return grid_evaluations[0]


def _evaluate_point(
    lat: float,
    lon: float,
    *,
    target_time: datetime,
    mode: str,
    prediction_time: datetime,
    keep_downloaded_files: bool,
    download_dir: str | Path | None,
    predictors: dict[str, PredictionFunction],
    caches: PredictorCaches,
    include_debug: bool,
) -> PointEvaluation:
    cache_key = (
        round(lat, 4),
        round(lon, 4),
        mode,
        target_time.astimezone(timezone.utc).isoformat(),
    )
    with caches.point_lock:
        cached_evaluation = caches.point_cache.get(cache_key)
    if cached_evaluation is not None:
        return cached_evaluation

    weather_snapshot = _load_weather_snapshot(
        lat,
        lon,
        mode=mode,
        target_time=target_time,
        prediction_time=prediction_time,
        keep_downloaded_files=keep_downloaded_files,
        download_dir=download_dir,
        caches=caches,
    )
    features = _compute_features_for_time(
        lat,
        lon,
        weather=weather_snapshot.weather,
        prediction_time=target_time,
    )
    probabilities: dict[str, float] = {}
    debug_components: dict[str, ModelComponents] = {}
    for phenomenon in PHENOMENA:
        predictor = predictors[phenomenon]
        probabilities[phenomenon] = predictor(features)
        if include_debug:
            components = _model_components_for_debug(predictor, features)
            if components is not None:
                debug_components[phenomenon] = components

    point_evaluation = PointEvaluation(
        weather=weather_snapshot.weather,
        sources=weather_snapshot.sources,
        features=features,
        probabilities=probabilities,
        debug_components=debug_components,
    )
    with caches.point_lock:
        caches.point_cache.setdefault(cache_key, point_evaluation)
        return caches.point_cache[cache_key]


def _model_components_for_debug(
    predictor: PredictionFunction,
    features: dict[str, float],
) -> ModelComponents | None:
    try:
        evaluation = predictor(features, return_components=True)  # type: ignore[misc,call-arg]
    except TypeError:
        return None
    return evaluation if isinstance(evaluation, ModelComponents) else None


def _load_weather_snapshot(
    lat: float,
    lon: float,
    *,
    mode: str,
    target_time: datetime,
    prediction_time: datetime,
    keep_downloaded_files: bool,
    download_dir: str | Path | None,
    caches: PredictorCaches,
) -> WeatherSnapshot:
    snapshot_time = target_time if mode == "forecast" else prediction_time
    cache_key = (
        round(lat, 4),
        round(lon, 4),
        mode,
        snapshot_time.astimezone(timezone.utc).isoformat(),
    )
    with caches.snapshot_lock:
        cached_snapshot = caches.snapshot_cache.get(cache_key)
    if cached_snapshot is not None:
        return cached_snapshot

    loaded_snapshot = get_weather_snapshot(
        lat,
        lon,
        mode=mode,
        keep_downloaded_files=keep_downloaded_files,
        download_dir=download_dir,
        at_time=snapshot_time,
    )
    with caches.snapshot_lock:
        caches.snapshot_cache.setdefault(cache_key, loaded_snapshot)
        return caches.snapshot_cache[cache_key]


def _compute_features_for_time(
    lat: float,
    lon: float,
    *,
    weather: dict[str, object],
    prediction_time: datetime,
) -> dict[str, float]:
    solar = get_solar_position(lat, lon, prediction_time)
    return compute_features(weather, solar)


def _time_window_label(hour_offset: int) -> str:
    if hour_offset == 0:
        return "now"
    return f"{hour_offset}h"


def _base_confidence_components(
    phenomenon: str,
    *,
    features: dict[str, float],
    weather: dict[str, object],
    source_names: tuple[str, ...],
    predictor: PredictionFunction,
    spatial_consistency: float,
) -> dict[str, float]:
    return {
        "data": _compute_data_component(
            phenomenon,
            features=features,
            weather=weather,
            source_names=source_names,
        ),
        "spatial": _clamp_unit_interval(spatial_consistency),
        "feature": _compute_feature_stability(
            phenomenon,
            features=features,
            predictor=predictor,
        ),
    }


def _compose_confidence(confidence_components: dict[str, float]) -> float:
    return _clamp_unit_interval(
        (0.4 * confidence_components["data"])
        + (0.2 * confidence_components["feature"])
        + (0.2 * confidence_components["spatial"])
        + (0.2 * confidence_components["temporal"])
    )


def _debug_payload(components: ModelComponents | None) -> dict[str, float]:
    if components is None:
        return {}
    return {
        "P": components.physical,
        "V": components.visibility,
        "G": components.geometry,
    }


def _compute_data_component(
    phenomenon: str,
    *,
    features: dict[str, float],
    weather: dict[str, object],
    source_names: tuple[str, ...],
) -> float:
    required_features = PHENOMENON_FEATURES[phenomenon]
    data_completeness = (
        sum(_feature_quality(feature_name, features=features, weather=weather) for feature_name in required_features)
        / len(required_features)
    )
    source_quality = _source_quality(source_names)
    return _clamp_unit_interval((0.7 * data_completeness) + (0.3 * source_quality))


def _feature_is_available(value: object) -> float:
    try:
        return 1.0 if math.isfinite(float(value)) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _feature_quality(
    feature_name: str,
    *,
    features: dict[str, float],
    weather: dict[str, object],
) -> float:
    cirrus_source = _raw_fraction(weather.get("cirrus_coverage", weather.get("cloud_cover_high")))
    optical_thickness = _raw_fraction(weather.get("cloud_optical_thickness"))
    humidity_250 = _raw_fraction(weather.get("humidity_250"))
    condensate_proxy = _raw_fraction(weather.get("condensate_proxy"))
    ice_fraction = _raw_fraction(weather.get("ice_cloud_fraction"))
    ice_levels = [
        _raw_fraction(weather.get("ice_300mb")),
        _raw_fraction(weather.get("ice_250mb")),
        _raw_fraction(weather.get("ice_200mb")),
    ]
    vertical_velocity_variance = _raw_fraction(weather.get("vertical_velocity_variance"))
    wind_shear = _raw_nonnegative(weather.get("wind_shear_250"))

    if feature_name == "cirrus_coverage":
        return 1.0 if math.isfinite(cirrus_source) else 0.0
    if feature_name == "cloud_optical_thickness":
        if math.isfinite(optical_thickness):
            return 1.0
        if math.isfinite(condensate_proxy) or math.isfinite(humidity_250):
            return 0.7
        if math.isfinite(cirrus_source):
            return 0.4
        return 0.0
    if feature_name == "humidity_250":
        return 1.0 if math.isfinite(humidity_250) else 0.0
    if feature_name == "thin_cirrus":
        return (
            _feature_quality("cirrus_coverage", features=features, weather=weather)
            + _feature_quality("cloud_optical_thickness", features=features, weather=weather)
        ) / 2.0
    if feature_name == "ice_presence":
        layer_quality_terms = [1.0 for value in ice_levels if math.isfinite(value)]
        if layer_quality_terms:
            return sum(layer_quality_terms) / 3.0
        thin_cirrus_quality = _feature_quality("thin_cirrus", features=features, weather=weather)
        if math.isfinite(ice_fraction):
            return (thin_cirrus_quality + 1.0) / 2.0
        return thin_cirrus_quality
    if feature_name == "plate_alignment":
        if math.isfinite(wind_shear) and math.isfinite(vertical_velocity_variance):
            return 1.0
        if math.isfinite(wind_shear):
            return 0.8
        return 0.5
    if feature_name == "wind_stability":
        if math.isfinite(wind_shear) and math.isfinite(vertical_velocity_variance):
            return 1.0
        if math.isfinite(wind_shear) or math.isfinite(vertical_velocity_variance):
            return 0.8
        return 0.5
    if feature_name == "cloud_variability":
        cloud_grid = weather.get("cloud_cover_grid", weather.get("cloud_variability"))
        if _has_cloud_variability_input(cloud_grid):
            return 1.0
        return 0.5 if math.isfinite(float(features.get("cloud_variability", math.nan))) else 0.0
    if feature_name in {"solar_elevation", "sun_visible", "sun_visibility"}:
        return 1.0 if math.isfinite(float(features.get("solar_elevation", math.nan))) else 0.0
    if feature_name == "precipitation":
        return 1.0 if math.isfinite(_raw_nonnegative(weather.get("precipitation"))) else 0.0
    if feature_name == "surface_visibility":
        return 1.0 if math.isfinite(_raw_nonnegative(weather.get("surface_visibility"))) else 0.0
    if feature_name == "fog_presence":
        return 1.0 if math.isfinite(_raw_fraction(weather.get("fog_presence"))) else 0.0
    return _feature_is_available(features.get(feature_name))


def _source_quality(source_names: tuple[str, ...]) -> float:
    source_set = {source_name.strip().lower() for source_name in source_names if source_name}
    goes_available = 1.0 if any(source_name.startswith("goes") for source_name in source_set) else 0.0
    metar_available = 1.0 if "metar" in source_set else 0.0
    gfs_available = 1.0 if "gfs" in source_set else 0.0
    return _clamp_unit_interval(((1.0 * goes_available) + (0.9 * metar_available) + (0.7 * gfs_available)) / 3.0)


def _compute_feature_stability(
    phenomenon: str,
    *,
    features: dict[str, float],
    predictor: PredictionFunction,
) -> float:
    baseline_probability = predictor(features)
    required_features = PHENOMENON_FEATURES[phenomenon]
    sensitivities: list[float] = []

    for feature_name in required_features:
        value = _numeric_or_nan(features.get(feature_name))
        if not math.isfinite(value):
            continue

        delta = _feature_perturbation_step(feature_name, value)
        if delta <= 0.0:
            continue

        lower_features = dict(features)
        lower_features[feature_name] = _perturb_feature_value(feature_name, value, -delta)
        upper_features = dict(features)
        upper_features[feature_name] = _perturb_feature_value(feature_name, value, delta)
        sensitivities.append(abs(predictor(lower_features) - baseline_probability))
        sensitivities.append(abs(predictor(upper_features) - baseline_probability))

    if not sensitivities:
        return 0.35

    mean_sensitivity = sum(sensitivities) / len(sensitivities)
    return _clamp_unit_interval(1.0 - (mean_sensitivity / 0.25))


def _feature_perturbation_step(feature_name: str, value: float) -> float:
    if feature_name == "solar_elevation":
        return 2.0
    if feature_name == "precipitation":
        return max(0.1, value * 0.25)
    if feature_name == "surface_visibility":
        return max(0.25, value * 0.2)
    return 0.05


def _perturb_feature_value(feature_name: str, value: float, delta: float) -> float:
    perturbed_value = value + delta
    if feature_name in {
        "cirrus_coverage",
        "cloud_optical_thickness",
        "condensate_proxy",
        "humidity_250",
        "ice_presence",
        "ice_300mb",
        "ice_250mb",
        "ice_200mb",
        "thin_cirrus",
        "plate_alignment",
        "wind_stability",
        "cloud_variability",
        "sun_visible",
        "sun_visibility",
        "fog_presence",
    }:
        return _clamp_unit_interval(perturbed_value)
    if feature_name in {"precipitation", "surface_visibility", "wind_shear_250", "vertical_velocity_variance"}:
        return max(0.0, perturbed_value)
    return perturbed_value


def _numeric_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _build_reason(phenomenon: str, features: dict[str, float], probability: float) -> str:
    if phenomenon == "halo":
        return _halo_reason(features)
    if phenomenon == "parhelia":
        return _parhelia_reason(features)
    if phenomenon == "cza":
        return _cza_reason(features)
    if phenomenon == "circumhorizontal_arc":
        return _circumhorizontal_arc_reason(features)
    if phenomenon == "upper_tangent_arc":
        return _upper_tangent_arc_reason(features)
    if phenomenon == "sun_pillar":
        return _sun_pillar_reason(features)
    if phenomenon == "crepuscular_rays":
        return _crepuscular_rays_reason(features)
    if phenomenon == "rainbow":
        return _rainbow_reason(features)
    if phenomenon == "fogbow":
        return _fogbow_reason(features)
    return f"Probability {probability:.3f} computed from physical, temporal, spatial, visibility, and geometry terms."


def _halo_reason(features: dict[str, float]) -> str:
    if _thin_icy_cirrus_available(features):
        if _sun_obscured(features):
            return "Thin icy cirrus is present, but visibility is reduced by thicker cloud or low sun."
        return "Thin icy cirrus with visible sun and favorable solar elevation supports halo formation."
    return "Halo potential is limited by weak thin-cirrus or ice-presence support."


def _parhelia_reason(features: dict[str, float]) -> str:
    if not _thin_icy_cirrus_available(features):
        return "Parhelia potential is limited by weak thin-cirrus or ice-presence support."
    if min(float(features.get("plate_alignment", 0.5)), float(features.get("wind_stability", 0.5))) < 0.5:
        return "Thin icy cirrus is present, but weak crystal alignment stability limits parhelia formation."
    return "Thin icy cirrus with aligned crystals and visible sun favors parhelia."


def _cza_reason(features: dict[str, float]) -> str:
    if not _thin_icy_cirrus_available(features):
        return "Circumzenithal arc potential is limited by weak thin-cirrus or ice-presence support."
    solar_elevation = float(features.get("solar_elevation", math.nan))
    if 15.0 <= solar_elevation <= 30.0:
        return "Thin icy cirrus with solar elevation near the circumzenithal arc window supports CZA formation."
    return "Solar elevation is outside the narrow circumzenithal arc window."


def _circumhorizontal_arc_reason(features: dict[str, float]) -> str:
    if not _thin_icy_cirrus_available(features):
        return "Circumhorizontal arc potential is limited by weak thin-cirrus or ice-presence support."
    if float(features.get("solar_elevation", math.nan)) >= 58.0:
        return "High sun with thin icy cirrus favors a circumhorizontal arc."
    return "The sun is not yet high enough for a strong circumhorizontal arc signal."


def _upper_tangent_arc_reason(features: dict[str, float]) -> str:
    if not _thin_icy_cirrus_available(features):
        return "Upper tangent arc potential is limited by weak thin-cirrus or ice-presence support."
    if float(features.get("solar_elevation", math.nan)) <= 32.0:
        return "Low sun with thin icy cirrus favors an upper tangent arc."
    return "The sun is too high for a strong upper tangent arc signal."


def _sun_pillar_reason(features: dict[str, float]) -> str:
    if float(features.get("solar_elevation", math.nan)) < -6.0:
        return "The sun is too far below the horizon for a strong sun pillar signal."
    if float(features.get("solar_elevation", math.nan)) > 14.0:
        return "The sun is too high for a strong sun pillar despite any ice support."
    if _sun_obscured(features):
        return "Near-horizon geometry is favorable, but cloud thickness reduces direct sunlight for a sun pillar."
    if float(features.get("wind_stability", 0.5)) < 0.45:
        return "Near-horizon geometry is favorable, but unstable upper-level flow weakens the sun-pillar signal."
    return "Near-horizon sun with ice-bearing thin cloud supports a sun pillar."


def _crepuscular_rays_reason(features: dict[str, float]) -> str:
    if _sun_obscured(features):
        return "Low-sun geometry may be favorable, but sunlight is too obscured for strong crepuscular rays."
    if float(features.get("cloud_variability", 0.3)) >= 0.35:
        return "Structured cloud cover with low sun favors crepuscular rays."
    return "Cloud structure is too uniform for a strong crepuscular-ray signal."


def _rainbow_reason(features: dict[str, float]) -> str:
    if float(features.get("precipitation", 0.0)) <= 0.1:
        return "Rainbow potential is limited by weak precipitation."
    if _sun_obscured(features):
        return "Precipitation is present, but direct sunlight is too limited for a strong rainbow."
    return "Rain with visible sun at moderate solar elevation favors rainbow formation."


def _fogbow_reason(features: dict[str, float]) -> str:
    fog_signal = max(
        float(features.get("fog_presence", 0.0)),
        math.exp(-((float(features.get("surface_visibility", math.nan)) - 0.8) ** 2) / (2.0 * (0.8**2)))
        if math.isfinite(float(features.get("surface_visibility", math.nan)))
        else 0.0,
    )
    if fog_signal < 0.35:
        return "Fogbow potential is limited by weak fog or mist support."
    if float(features.get("precipitation", 0.0)) > 0.6:
        return "Fog or mist is present, but stronger precipitation suppresses fogbow conditions."
    if _sun_obscured(features):
        return "Fog or mist is present, but direct sunlight is too limited for a strong fogbow."
    return "Fog or mist with visible sun and low visibility favors fogbow formation."


def _thin_icy_cirrus_available(features: dict[str, float]) -> bool:
    return (
        float(features.get("thin_cirrus", 0.0)) >= 0.2
        and float(features.get("ice_presence", 0.0)) >= 0.2
    )


def _sun_obscured(features: dict[str, float]) -> bool:
    return (
        float(features.get("sun_visible", 0.0)) <= 0.15
        or float(features.get("cloud_optical_thickness", 0.0)) >= 0.7
    )


def _has_cloud_variability_input(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return len(value) >= 2
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _raw_fraction(value: object) -> float:
    if value is None:
        return math.nan
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(numeric_value):
        return math.nan
    if numeric_value > 1.0:
        numeric_value /= 100.0
    return _clamp_unit_interval(numeric_value)


def _raw_nonnegative(value: object) -> float:
    if value is None:
        return math.nan
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(numeric_value):
        return math.nan
    return max(0.0, numeric_value)


def _clamp_unit_interval(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
