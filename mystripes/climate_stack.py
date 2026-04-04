from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from queue import Empty, Queue
from typing import Any

import pandas as pd

from mystripes.cds import (
    ERA5_LAND_MONTHLY_DATASET,
    ERA5_MONTHLY_DATASET,
    TEMPERATURE_GRID_REQUEST_CACHE_DIR,
    TEMPERATURE_REQUEST_CACHE_DIR,
    TEMPERATURE_TIMELINE_CACHE_DIR,
    CDSConfig,
    DatasetWindow,
    ProgressCallback,
    _emit_progress,
    _request_area,
    _load_cached_temperature_series,
    _missing_temperature_ranges,
    _temperature_series_cache_path,
    _temperature_grid_request_cache_path,
    _temperature_timeline_cache_path,
    fetch_temperature_grid_frame,
    fetch_point_temperature_series,
    fetch_saved_temperature_series,
    get_dataset_window,
)
from mystripes.models import LifePeriod
from mystripes.twcr import (
    TWCR_DISPLAY_NAME,
    TWCR_MAX_END,
    TWCR_MIN_START,
    TWCR_RAW_YEAR_CACHE_DIR,
    TWCR_REQUEST_CACHE_DIR,
    TWCR_SOURCE_ID,
    TWCR_TIMELINE_CACHE_DIR,
    _twcr_timeline_cache_path,
    _twcr_window_cache_path,
    estimate_missing_twcr_years,
    ensure_twcr_year_cached,
    fetch_saved_twcr_temperature_series,
    fetch_twcr_temperature_series,
)

CALIBRATION_BASELINE_START = date(1961, 1, 1)
CALIBRATION_BASELINE_END = date(1990, 12, 31)
ERA5_BRIDGE_END = date(1949, 12, 31)
ERA5_LAND_START = date(1950, 1, 1)
TWCR_END = date(1939, 12, 31)


@dataclass(frozen=True)
class ClimateSourceSlice:
    source_id: str
    label: str
    start_date: date
    end_date: date


@dataclass(frozen=True)
class LocationClimateRequest:
    location_key: str
    display_name: str
    latitude: float
    longitude: float
    boundary_geojson: Mapping[str, Any] | None
    boundary_bbox: tuple[float, float, float, float] | None
    start_date: date
    end_date: date


@dataclass(frozen=True)
class ClimateDownloadEstimate:
    uses_historical_fallback: bool
    uncached_era5_bridge_fetches: int
    uncached_twcr_years: tuple[int, ...]


@dataclass(frozen=True)
class SharedClimateTask:
    work_id: str
    source_id: str
    label: str
    task_kind: str
    start_date: date | None = None
    end_date: date | None = None
    year: int | None = None
    dataset: Any | None = None
    area: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class ClimateBatchPlan:
    location_requests: tuple[LocationClimateRequest, ...]
    shared_tasks: tuple[SharedClimateTask, ...]


def get_climate_dataset_window() -> DatasetWindow:
    era5_land_window = get_dataset_window(ERA5_LAND_MONTHLY_DATASET)
    return DatasetWindow(min_start=TWCR_MIN_START, max_end=era5_land_window.max_end)


def build_climate_source_slices(start_date: date, end_date: date) -> list[ClimateSourceSlice]:
    if start_date > end_date:
        raise ValueError("Start date must be on or before end date.")
    if start_date < TWCR_MIN_START:
        raise ValueError(f"Historical fallback only covers {TWCR_MIN_START.isoformat()} onward.")

    slices: list[ClimateSourceSlice] = []
    if start_date <= TWCR_END:
        twcr_end = min(end_date, TWCR_END)
        slices.append(
            ClimateSourceSlice(
                source_id=TWCR_SOURCE_ID,
                label=TWCR_DISPLAY_NAME,
                start_date=start_date,
                end_date=twcr_end,
            )
        )

    era5_start = max(start_date, date(1940, 1, 1))
    if era5_start <= min(end_date, ERA5_BRIDGE_END):
        slices.append(
            ClimateSourceSlice(
                source_id="era5_bridge",
                label=ERA5_MONTHLY_DATASET.display_name,
                start_date=era5_start,
                end_date=min(end_date, ERA5_BRIDGE_END),
            )
        )

    era5_land_start = max(start_date, ERA5_LAND_START)
    if era5_land_start <= end_date:
        slices.append(
            ClimateSourceSlice(
                source_id="era5_land",
                label=ERA5_LAND_MONTHLY_DATASET.display_name,
                start_date=era5_land_start,
                end_date=end_date,
            )
        )

    return slices


def build_location_climate_requests(
    periods: Sequence[LifePeriod],
    *,
    fetch_start: date,
    fetch_end: date,
) -> list[LocationClimateRequest]:
    requests_by_location: dict[str, LocationClimateRequest] = {}
    for period in periods:
        requests_by_location.setdefault(
            period.location_key,
            LocationClimateRequest(
                location_key=period.location_key,
                display_name=period.display_name,
                latitude=period.latitude,
                longitude=period.longitude,
                boundary_geojson=period.boundary_geojson,
                boundary_bbox=period.bounding_box,
                start_date=fetch_start,
                end_date=fetch_end,
            ),
        )
    return list(requests_by_location.values())


def estimate_climate_downloads(
    location_requests: Sequence[LocationClimateRequest],
    *,
    spatial_mode: str,
    radius_km: float | None = None,
) -> ClimateDownloadEstimate:
    batch_plan = build_climate_batch_plan(
        location_requests,
        spatial_mode=spatial_mode,
        radius_km=radius_km,
    )
    uses_historical_fallback = any(
        any(source.source_id != "era5_land" for source in build_climate_source_slices(request.start_date, request.end_date))
        for request in location_requests
    )
    uncached_era5_bridge_fetches = sum(
        1
        for task in batch_plan.shared_tasks
        if task.source_id == "era5_bridge"
    )
    uncached_twcr_years = {
        int(task.year)
        for task in batch_plan.shared_tasks
        if task.source_id == TWCR_SOURCE_ID and task.year is not None
    }

    return ClimateDownloadEstimate(
        uses_historical_fallback=uses_historical_fallback,
        uncached_era5_bridge_fetches=uncached_era5_bridge_fetches,
        uncached_twcr_years=tuple(sorted(uncached_twcr_years)),
    )


def build_climate_batch_plan(
    location_requests: Sequence[LocationClimateRequest],
    *,
    spatial_mode: str,
    radius_km: float | None = None,
) -> ClimateBatchPlan:
    shared_tasks: dict[str, SharedClimateTask] = {}

    for request in location_requests:
        request_area_era5_land = _request_area_for_location_request(
            request,
            spatial_mode=spatial_mode,
            radius_km=radius_km,
            dataset=ERA5_LAND_MONTHLY_DATASET,
        )
        request_area_era5 = _request_area_for_location_request(
            request,
            spatial_mode=spatial_mode,
            radius_km=radius_km,
            dataset=ERA5_MONTHLY_DATASET,
        )
        source_slices = build_climate_source_slices(request.start_date, request.end_date)
        for source_slice in source_slices:
            if source_slice.source_id == "era5_land":
                _add_missing_cds_timeline_tasks(
                    shared_tasks,
                    request=request,
                    dataset=ERA5_LAND_MONTHLY_DATASET,
                    area=request_area_era5_land,
                    start_date=source_slice.start_date,
                    end_date=source_slice.end_date,
                    spatial_mode=spatial_mode,
                    radius_km=radius_km,
                )
                continue

            if source_slice.source_id == "era5_bridge":
                _add_missing_cds_timeline_tasks(
                    shared_tasks,
                    request=request,
                    dataset=ERA5_MONTHLY_DATASET,
                    area=request_area_era5,
                    start_date=source_slice.start_date,
                    end_date=source_slice.end_date,
                    spatial_mode=spatial_mode,
                    radius_km=radius_km,
                )
                _add_missing_cds_window_task(
                    shared_tasks,
                    request=request,
                    dataset=ERA5_MONTHLY_DATASET,
                    area=request_area_era5,
                    start_date=CALIBRATION_BASELINE_START,
                    end_date=CALIBRATION_BASELINE_END,
                    spatial_mode=spatial_mode,
                    radius_km=radius_km,
                    source_id="era5_bridge",
                    label=ERA5_MONTHLY_DATASET.display_name,
                )
                _add_missing_cds_window_task(
                    shared_tasks,
                    request=request,
                    dataset=ERA5_LAND_MONTHLY_DATASET,
                    area=request_area_era5_land,
                    start_date=CALIBRATION_BASELINE_START,
                    end_date=CALIBRATION_BASELINE_END,
                    spatial_mode=spatial_mode,
                    radius_km=radius_km,
                    source_id="era5_land",
                    label=ERA5_LAND_MONTHLY_DATASET.display_name,
                )
                continue

            if source_slice.source_id == TWCR_SOURCE_ID:
                _add_missing_twcr_timeline_tasks(
                    shared_tasks,
                    request=request,
                    start_date=source_slice.start_date,
                    end_date=source_slice.end_date,
                    spatial_mode=spatial_mode,
                    radius_km=radius_km,
                )
                _add_missing_twcr_window_tasks(
                    shared_tasks,
                    request=request,
                    start_date=CALIBRATION_BASELINE_START,
                    end_date=CALIBRATION_BASELINE_END,
                    spatial_mode=spatial_mode,
                    radius_km=radius_km,
                )
                _add_missing_cds_window_task(
                    shared_tasks,
                    request=request,
                    dataset=ERA5_LAND_MONTHLY_DATASET,
                    area=request_area_era5_land,
                    start_date=CALIBRATION_BASELINE_START,
                    end_date=CALIBRATION_BASELINE_END,
                    spatial_mode=spatial_mode,
                    radius_km=radius_km,
                    source_id="era5_land",
                    label=ERA5_LAND_MONTHLY_DATASET.display_name,
                )

    return ClimateBatchPlan(
        location_requests=tuple(location_requests),
        shared_tasks=tuple(shared_tasks.values()),
    )


def fetch_saved_climate_series_batch(
    config: CDSConfig,
    location_requests: Sequence[LocationClimateRequest],
    *,
    spatial_mode: str = "single_cell",
    radius_km: float | None = None,
    progress_callback: ProgressCallback | None = None,
    max_workers: int | None = None,
) -> dict[str, pd.DataFrame]:
    batch_plan = build_climate_batch_plan(
        location_requests,
        spatial_mode=spatial_mode,
        radius_km=radius_km,
    )
    _emit_progress(
        progress_callback,
        "batch_plan",
        total_locations=len(batch_plan.location_requests),
        total_shared_tasks=len(batch_plan.shared_tasks),
    )

    event_queue: Queue[dict[str, Any]] = Queue()
    results: dict[str, pd.DataFrame] = {}
    effective_workers = max_workers or max(
        1,
        min(8, len(batch_plan.location_requests) + len(batch_plan.shared_tasks)),
    )

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        if batch_plan.shared_tasks:
            shared_futures = {
                executor.submit(_run_shared_task, config, task, event_queue.put): task
                for task in batch_plan.shared_tasks
            }
            _drain_futures(
                shared_futures,
                event_queue=event_queue,
                progress_callback=progress_callback,
            )

        location_futures = {
            executor.submit(
                _run_location_task,
                config,
                request,
                spatial_mode,
                radius_km,
                event_queue.put,
            ): request
            for request in batch_plan.location_requests
        }
        results.update(
            _drain_futures(
                location_futures,
                event_queue=event_queue,
                progress_callback=progress_callback,
                returns_location_frames=True,
            )
        )

    _drain_progress_queue(event_queue, progress_callback)
    return results


def _drain_futures(
    future_map: dict[Future[Any], Any],
    *,
    event_queue: Queue[dict[str, Any]],
    progress_callback: ProgressCallback | None,
    returns_location_frames: bool = False,
) -> dict[str, pd.DataFrame]:
    pending = set(future_map)
    results: dict[str, pd.DataFrame] = {}

    while pending:
        done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
        _drain_progress_queue(event_queue, progress_callback)
        for future in done:
            payload = future_map[future]
            result = future.result()
            if returns_location_frames:
                results[payload.location_key] = result
        if not done:
            _drain_progress_queue(event_queue, progress_callback)

    _drain_progress_queue(event_queue, progress_callback)
    return results


def _drain_progress_queue(
    event_queue: Queue[dict[str, Any]],
    progress_callback: ProgressCallback | None,
) -> None:
    while True:
        try:
            event = event_queue.get_nowait()
        except Empty:
            return
        if progress_callback is not None:
            progress_callback(event)


def _run_shared_task(
    config: CDSConfig,
    task: SharedClimateTask,
    progress_callback: ProgressCallback | None,
) -> None:
    _emit_progress(
        progress_callback,
        "shared_task_started",
        work_id=task.work_id,
        source_id=task.source_id,
        source_label=task.label,
        task_kind=task.task_kind,
    )
    if task.task_kind == "cds_grid":
        fetch_temperature_grid_frame(
            config=config,
            start_date=task.start_date,
            end_date=task.end_date,
            area=task.area,
            dataset=task.dataset,
            target_latitude=0.0,
            target_longitude=0.0,
            progress_callback=progress_callback,
        )
    elif task.task_kind == "twcr_year":
        ensure_twcr_year_cached(
            int(task.year),
            raw_year_cache_dir=TWCR_RAW_YEAR_CACHE_DIR,
            progress_callback=progress_callback,
        )
    else:  # pragma: no cover - defensive guard.
        raise ValueError(f"Unsupported shared climate task kind: {task.task_kind}")
    _emit_progress(
        progress_callback,
        "shared_task_finished",
        work_id=task.work_id,
        source_id=task.source_id,
        source_label=task.label,
        task_kind=task.task_kind,
    )


def _run_location_task(
    config: CDSConfig,
    request: LocationClimateRequest,
    spatial_mode: str,
    radius_km: float | None,
    progress_callback: ProgressCallback | None,
) -> pd.DataFrame:
    _emit_progress(
        progress_callback,
        "location_started",
        location_key=request.location_key,
        location_label=request.display_name,
    )
    frame = fetch_saved_climate_series(
        config=config,
        latitude=request.latitude,
        longitude=request.longitude,
        start_date=request.start_date,
        end_date=request.end_date,
        spatial_mode=spatial_mode,
        radius_km=radius_km,
        boundary_geojson=request.boundary_geojson,
        boundary_bbox=request.boundary_bbox,
        progress_callback=_forward_location_progress(progress_callback, request),
    )
    _emit_progress(
        progress_callback,
        "location_finished",
        location_key=request.location_key,
        location_label=request.display_name,
    )
    return frame


def _forward_location_progress(
    progress_callback: ProgressCallback | None,
    request: LocationClimateRequest,
):
    def _forward(event: dict[str, Any]) -> None:
        if progress_callback is None:
            return
        forwarded = dict(event)
        forwarded.setdefault("location_key", request.location_key)
        forwarded.setdefault("location_label", request.display_name)
        progress_callback(forwarded)

    return _forward


def _request_area_for_location_request(
    request: LocationClimateRequest,
    *,
    spatial_mode: str,
    radius_km: float | None,
    dataset,
) -> tuple[float, float, float, float]:
    return _request_area(
        latitude=request.latitude,
        longitude=request.longitude,
        spatial_mode=spatial_mode,
        radius_km=radius_km,
        boundary_geojson=request.boundary_geojson,
        boundary_bbox=request.boundary_bbox,
        grid_step_degrees=dataset.grid_step_degrees,
    )


def _add_missing_cds_timeline_tasks(
    shared_tasks: dict[str, SharedClimateTask],
    *,
    request: LocationClimateRequest,
    dataset,
    area: tuple[float, float, float, float],
    start_date: date,
    end_date: date,
    spatial_mode: str,
    radius_km: float | None,
) -> None:
    timeline_cache_path = _temperature_timeline_cache_path(
        cache_dir=TEMPERATURE_TIMELINE_CACHE_DIR,
        dataset=dataset,
        latitude=request.latitude,
        longitude=request.longitude,
        spatial_mode=spatial_mode,
        radius_km=radius_km,
        boundary_geojson=request.boundary_geojson,
        boundary_bbox=request.boundary_bbox,
    )
    cached_frame = _load_cached_temperature_series(timeline_cache_path)
    missing_ranges = _missing_temperature_ranges(cached_frame, start_date=start_date, end_date=end_date)
    source_id = "era5_land" if dataset.name == ERA5_LAND_MONTHLY_DATASET.name else "era5_bridge"
    for missing_start, missing_end in missing_ranges:
        work_path = _temperature_grid_request_cache_path(
            cache_dir=TEMPERATURE_GRID_REQUEST_CACHE_DIR,
            dataset=dataset,
            start_date=missing_start,
            end_date=missing_end,
            area=area,
        )
        if work_path.exists():
            continue
        shared_tasks.setdefault(
            str(work_path),
            SharedClimateTask(
                work_id=str(work_path),
                source_id=source_id,
                label=dataset.display_name,
                task_kind="cds_grid",
                start_date=missing_start,
                end_date=missing_end,
                dataset=dataset,
                area=area,
            ),
        )


def _add_missing_cds_window_task(
    shared_tasks: dict[str, SharedClimateTask],
    *,
    request: LocationClimateRequest,
    dataset,
    area: tuple[float, float, float, float],
    start_date: date,
    end_date: date,
    spatial_mode: str,
    radius_km: float | None,
    source_id: str,
    label: str,
) -> None:
    if not _needs_cds_window_fetch(
        dataset=dataset,
        latitude=request.latitude,
        longitude=request.longitude,
        start_date=start_date,
        end_date=end_date,
        spatial_mode=spatial_mode,
        radius_km=radius_km,
        boundary_geojson=request.boundary_geojson,
        boundary_bbox=request.boundary_bbox,
    ):
        return
    work_path = _temperature_grid_request_cache_path(
        cache_dir=TEMPERATURE_GRID_REQUEST_CACHE_DIR,
        dataset=dataset,
        start_date=start_date,
        end_date=end_date,
        area=area,
    )
    if work_path.exists():
        return
    shared_tasks.setdefault(
        str(work_path),
        SharedClimateTask(
            work_id=str(work_path),
            source_id=source_id,
            label=label,
            task_kind="cds_grid",
            start_date=start_date,
            end_date=end_date,
            dataset=dataset,
            area=area,
        ),
    )


def _add_missing_twcr_timeline_tasks(
    shared_tasks: dict[str, SharedClimateTask],
    *,
    request: LocationClimateRequest,
    start_date: date,
    end_date: date,
    spatial_mode: str,
    radius_km: float | None,
) -> None:
    timeline_cache_path = _twcr_timeline_cache_path(
        cache_dir=TWCR_TIMELINE_CACHE_DIR,
        latitude=request.latitude,
        longitude=request.longitude,
        spatial_mode=spatial_mode,
        radius_km=radius_km,
        boundary_geojson=request.boundary_geojson,
        boundary_bbox=request.boundary_bbox,
    )
    cached_frame = _load_cached_temperature_series(timeline_cache_path)
    missing_ranges = _missing_temperature_ranges(cached_frame, start_date=start_date, end_date=end_date)
    for missing_start, missing_end in missing_ranges:
        _register_twcr_year_tasks(shared_tasks, start_date=missing_start, end_date=missing_end)


def _add_missing_twcr_window_tasks(
    shared_tasks: dict[str, SharedClimateTask],
    *,
    request: LocationClimateRequest,
    start_date: date,
    end_date: date,
    spatial_mode: str,
    radius_km: float | None,
) -> None:
    if not _needs_twcr_window_fetch(
        latitude=request.latitude,
        longitude=request.longitude,
        start_date=start_date,
        end_date=end_date,
        spatial_mode=spatial_mode,
        radius_km=radius_km,
        boundary_geojson=request.boundary_geojson,
        boundary_bbox=request.boundary_bbox,
    ):
        return
    _register_twcr_year_tasks(shared_tasks, start_date=start_date, end_date=end_date)


def _register_twcr_year_tasks(
    shared_tasks: dict[str, SharedClimateTask],
    *,
    start_date: date,
    end_date: date,
) -> None:
    for year in estimate_missing_twcr_years(
        start_date,
        end_date,
        raw_year_cache_dir=TWCR_RAW_YEAR_CACHE_DIR,
    ):
        work_id = str(TWCR_RAW_YEAR_CACHE_DIR / f"air.2m.mon.mean.{year:04d}.nc")
        shared_tasks.setdefault(
            work_id,
            SharedClimateTask(
                work_id=work_id,
                source_id=TWCR_SOURCE_ID,
                label=TWCR_DISPLAY_NAME,
                task_kind="twcr_year",
                year=year,
            ),
        )


def fetch_saved_climate_series(
    config: CDSConfig,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    spatial_mode: str = "single_cell",
    radius_km: float | None = None,
    boundary_geojson: Mapping[str, Any] | None = None,
    boundary_bbox: tuple[float, float, float, float] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    source_slices = build_climate_source_slices(start_date, end_date)
    if not source_slices:
        raise ValueError("No climate-data source covers the requested period.")

    anchor_calibration_frame: pd.DataFrame | None = None
    source_calibration_frames: dict[str, pd.DataFrame] = {}
    frames: list[pd.DataFrame] = []

    for source_slice in source_slices:
        _emit_progress(
            progress_callback,
            "source_started",
            source_id=source_slice.source_id,
            source_label=source_slice.label,
            source_start=source_slice.start_date.isoformat(),
            source_end=source_slice.end_date.isoformat(),
        )

        if source_slice.source_id == "era5_land":
            frame = fetch_saved_temperature_series(
                config=config,
                latitude=latitude,
                longitude=longitude,
                start_date=source_slice.start_date,
                end_date=source_slice.end_date,
                dataset=ERA5_LAND_MONTHLY_DATASET,
                spatial_mode=spatial_mode,
                radius_km=radius_km,
                boundary_geojson=boundary_geojson,
                boundary_bbox=boundary_bbox,
                progress_callback=_forward_source_progress(progress_callback, source_slice),
            )
        elif source_slice.source_id == "era5_bridge":
            raw_frame = fetch_saved_temperature_series(
                config=config,
                latitude=latitude,
                longitude=longitude,
                start_date=source_slice.start_date,
                end_date=source_slice.end_date,
                dataset=ERA5_MONTHLY_DATASET,
                spatial_mode=spatial_mode,
                radius_km=radius_km,
                boundary_geojson=boundary_geojson,
                boundary_bbox=boundary_bbox,
                progress_callback=_forward_source_progress(progress_callback, source_slice),
            )
            anchor_calibration_frame = anchor_calibration_frame or _fetch_anchor_calibration_frame(
                config=config,
                latitude=latitude,
                longitude=longitude,
                spatial_mode=spatial_mode,
                radius_km=radius_km,
                boundary_geojson=boundary_geojson,
                boundary_bbox=boundary_bbox,
                progress_callback=_forward_calibration_progress(progress_callback, source_slice),
            )
            source_calibration_frames.setdefault(
                source_slice.source_id,
                fetch_point_temperature_series(
                    config=config,
                    latitude=latitude,
                    longitude=longitude,
                    start_date=CALIBRATION_BASELINE_START,
                    end_date=CALIBRATION_BASELINE_END,
                    dataset=ERA5_MONTHLY_DATASET,
                    spatial_mode=spatial_mode,
                    radius_km=radius_km,
                    boundary_geojson=boundary_geojson,
                    boundary_bbox=boundary_bbox,
                    cache_dir=TEMPERATURE_REQUEST_CACHE_DIR,
                    progress_callback=_forward_calibration_progress(progress_callback, source_slice),
                ),
            )
            frame = _calibrate_source_monthly_frame(
                raw_frame,
                anchor_calibration_frame=anchor_calibration_frame,
                source_calibration_frame=source_calibration_frames[source_slice.source_id],
            )
        elif source_slice.source_id == TWCR_SOURCE_ID:
            raw_frame = fetch_saved_twcr_temperature_series(
                latitude=latitude,
                longitude=longitude,
                start_date=source_slice.start_date,
                end_date=source_slice.end_date,
                spatial_mode=spatial_mode,
                radius_km=radius_km,
                boundary_geojson=boundary_geojson,
                boundary_bbox=boundary_bbox,
                progress_callback=_forward_source_progress(progress_callback, source_slice),
            )
            anchor_calibration_frame = anchor_calibration_frame or _fetch_anchor_calibration_frame(
                config=config,
                latitude=latitude,
                longitude=longitude,
                spatial_mode=spatial_mode,
                radius_km=radius_km,
                boundary_geojson=boundary_geojson,
                boundary_bbox=boundary_bbox,
                progress_callback=_forward_calibration_progress(progress_callback, source_slice),
            )
            source_calibration_frames.setdefault(
                source_slice.source_id,
                fetch_twcr_temperature_series(
                    latitude=latitude,
                    longitude=longitude,
                    start_date=CALIBRATION_BASELINE_START,
                    end_date=CALIBRATION_BASELINE_END,
                    spatial_mode=spatial_mode,
                    radius_km=radius_km,
                    boundary_geojson=boundary_geojson,
                    boundary_bbox=boundary_bbox,
                    progress_callback=_forward_calibration_progress(progress_callback, source_slice),
                ),
            )
            frame = _calibrate_source_monthly_frame(
                raw_frame,
                anchor_calibration_frame=anchor_calibration_frame,
                source_calibration_frame=source_calibration_frames[source_slice.source_id],
            )
        else:  # pragma: no cover - defensive guard.
            raise ValueError(f"Unsupported climate source: {source_slice.source_id}")

        _emit_progress(
            progress_callback,
            "source_finished",
            source_id=source_slice.source_id,
            source_label=source_slice.label,
            source_start=source_slice.start_date.isoformat(),
            source_end=source_slice.end_date.isoformat(),
        )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp")
    return combined.reset_index(drop=True)


def climate_stack_note(start_date: date, end_date: date) -> str | None:
    source_slices = build_climate_source_slices(start_date, end_date)
    if all(source.source_id == "era5_land" for source in source_slices):
        return None

    parts: list[str] = []
    if any(source.source_id == "era5_bridge" for source in source_slices):
        parts.append("ERA5 monthly single levels for 1940-1949")
    if any(source.source_id == TWCR_SOURCE_ID for source in source_slices):
        parts.append("NOAA 20CRv3 monthly 2 m air temperature before 1940")
    detail = " and ".join(parts)
    return (
        f"Historical fallback uses {detail}. Non-ERA5-Land slices are anomaly-aligned to "
        "ERA5-Land with an internal 1961-1990 monthly calibration."
    )


def preflight_download_message(estimate: ClimateDownloadEstimate) -> tuple[str, str] | None:
    if not estimate.uses_historical_fallback:
        return None
    if not estimate.uncached_era5_bridge_fetches and not estimate.uncached_twcr_years:
        return None

    twcr_year_count = len(estimate.uncached_twcr_years)
    level = "warning" if twcr_year_count > 10 or estimate.uncached_era5_bridge_fetches > 2 else "info"
    message_parts: list[str] = []
    if twcr_year_count:
        message_parts.append(f"{twcr_year_count} uncached 20CRv3 yearly monthly files")
    if estimate.uncached_era5_bridge_fetches:
        noun = "fetch" if estimate.uncached_era5_bridge_fetches == 1 else "fetches"
        message_parts.append(f"{estimate.uncached_era5_bridge_fetches} ERA5 bridge {noun}")
    summary = " and ".join(message_parts)
    return (
        level,
        f"The current story line will need {summary}. Later runs reuse the local cache.",
    )


def _fetch_anchor_calibration_frame(
    *,
    config: CDSConfig,
    latitude: float,
    longitude: float,
    spatial_mode: str,
    radius_km: float | None,
    boundary_geojson: Mapping[str, Any] | None,
    boundary_bbox: tuple[float, float, float, float] | None,
    progress_callback: ProgressCallback | None,
) -> pd.DataFrame:
    return fetch_point_temperature_series(
        config=config,
        latitude=latitude,
        longitude=longitude,
        start_date=CALIBRATION_BASELINE_START,
        end_date=CALIBRATION_BASELINE_END,
        dataset=ERA5_LAND_MONTHLY_DATASET,
        spatial_mode=spatial_mode,
        radius_km=radius_km,
        boundary_geojson=boundary_geojson,
        boundary_bbox=boundary_bbox,
        cache_dir=TEMPERATURE_REQUEST_CACHE_DIR,
        progress_callback=progress_callback,
    )


def _forward_source_progress(
    progress_callback: ProgressCallback | None,
    source_slice: ClimateSourceSlice,
):
    def _forward(event: dict[str, Any]) -> None:
        if progress_callback is None:
            return
        forwarded = dict(event)
        forwarded.setdefault("purpose", "source")
        forwarded.setdefault("source_id", source_slice.source_id)
        forwarded.setdefault("source_label", source_slice.label)
        forwarded.setdefault("source_start", source_slice.start_date.isoformat())
        forwarded.setdefault("source_end", source_slice.end_date.isoformat())
        progress_callback(forwarded)

    return _forward


def _forward_calibration_progress(
    progress_callback: ProgressCallback | None,
    source_slice: ClimateSourceSlice,
):
    def _forward(event: dict[str, Any]) -> None:
        if progress_callback is None:
            return
        forwarded = dict(event)
        forwarded["purpose"] = "calibration"
        forwarded.setdefault("source_id", source_slice.source_id)
        forwarded.setdefault("source_label", source_slice.label)
        progress_callback(forwarded)

    return _forward


def _calibrate_source_monthly_frame(
    frame: pd.DataFrame,
    *,
    anchor_calibration_frame: pd.DataFrame,
    source_calibration_frame: pd.DataFrame,
) -> pd.DataFrame:
    calibration_table = _monthly_calibration_table(
        anchor_calibration_frame=anchor_calibration_frame,
        source_calibration_frame=source_calibration_frame,
    )
    corrected = frame.copy()
    corrected["month"] = corrected["timestamp"].dt.month
    corrected = corrected.merge(calibration_table, on="month", how="left")

    anchor_mean = pd.to_numeric(corrected["anchor_mean_c"], errors="coerce")
    source_mean = pd.to_numeric(corrected["source_mean_c"], errors="coerce")
    scale = pd.to_numeric(corrected["scale"], errors="coerce").fillna(1.0)
    can_shift = anchor_mean.notna() & source_mean.notna()
    corrected_temperatures = corrected["temperature_c"].astype(float).copy()
    corrected_temperatures.loc[can_shift] = (
        anchor_mean.loc[can_shift]
        + scale.loc[can_shift] * (corrected_temperatures.loc[can_shift] - source_mean.loc[can_shift])
    )
    corrected["temperature_c"] = corrected_temperatures
    return corrected.drop(columns=["month", "anchor_mean_c", "source_mean_c", "scale"])


def _monthly_calibration_table(
    *,
    anchor_calibration_frame: pd.DataFrame,
    source_calibration_frame: pd.DataFrame,
) -> pd.DataFrame:
    anchor_stats = _monthly_temperature_stats(anchor_calibration_frame).rename(
        columns={"mean_c": "anchor_mean_c", "std_c": "anchor_std_c"}
    )
    source_stats = _monthly_temperature_stats(source_calibration_frame).rename(
        columns={"mean_c": "source_mean_c", "std_c": "source_std_c"}
    )
    merged = anchor_stats.merge(source_stats, on="month", how="outer")
    scale = pd.Series(1.0, index=merged.index, dtype=float)
    valid = (
        pd.to_numeric(merged["anchor_std_c"], errors="coerce").notna()
        & pd.to_numeric(merged["source_std_c"], errors="coerce").notna()
        & (pd.to_numeric(merged["source_std_c"], errors="coerce").abs() > 1e-9)
    )
    scale.loc[valid] = (
        pd.to_numeric(merged.loc[valid, "anchor_std_c"], errors="coerce")
        / pd.to_numeric(merged.loc[valid, "source_std_c"], errors="coerce")
    )
    merged["scale"] = scale.fillna(1.0)
    return merged[["month", "anchor_mean_c", "source_mean_c", "scale"]]


def _monthly_temperature_stats(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["month"] = working["timestamp"].dt.month
    grouped = (
        working.groupby("month", as_index=False)
        .agg(
            mean_c=("temperature_c", "mean"),
            std_c=("temperature_c", "std"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    return grouped


def _needs_cds_timeline_fetch(
    *,
    dataset,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    spatial_mode: str,
    radius_km: float | None,
    boundary_geojson: Mapping[str, Any] | None,
    boundary_bbox: tuple[float, float, float, float] | None,
) -> bool:
    cache_path = _temperature_timeline_cache_path(
        cache_dir=TEMPERATURE_TIMELINE_CACHE_DIR,
        dataset=dataset,
        latitude=latitude,
        longitude=longitude,
        spatial_mode=spatial_mode,
        radius_km=radius_km,
        boundary_geojson=boundary_geojson,
        boundary_bbox=boundary_bbox,
    )
    cached_frame = _load_cached_temperature_series(cache_path)
    return bool(_missing_temperature_ranges(cached_frame, start_date=start_date, end_date=end_date))


def _needs_cds_window_fetch(
    *,
    dataset,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    spatial_mode: str,
    radius_km: float | None,
    boundary_geojson: Mapping[str, Any] | None,
    boundary_bbox: tuple[float, float, float, float] | None,
) -> bool:
    cache_path = _temperature_series_cache_path(
        cache_dir=TEMPERATURE_REQUEST_CACHE_DIR,
        dataset=dataset,
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        spatial_mode=spatial_mode,
        radius_km=radius_km,
        boundary_geojson=boundary_geojson,
        boundary_bbox=boundary_bbox,
    )
    return not cache_path.exists()


def _needs_twcr_window_fetch(
    *,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    spatial_mode: str,
    radius_km: float | None,
    boundary_geojson: Mapping[str, Any] | None,
    boundary_bbox: tuple[float, float, float, float] | None,
) -> bool:
    cache_path = _twcr_window_cache_path(
        cache_dir=TWCR_REQUEST_CACHE_DIR,
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        spatial_mode=spatial_mode,
        radius_km=radius_km,
        boundary_geojson=boundary_geojson,
        boundary_bbox=boundary_bbox,
    )
    return not cache_path.exists()
