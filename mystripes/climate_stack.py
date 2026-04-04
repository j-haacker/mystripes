from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
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
    _slice_temperature_series,
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
    TWCR_GRID_REQUEST_CACHE_DIR,
    TWCR_GRID_STEP_DEGREES,
    TWCR_MAX_END,
    TWCR_MIN_START,
    TWCR_SOURCE_ID,
    TWCR_TIMELINE_CACHE_DIR,
    _twcr_grid_window_request_path,
    _twcr_timeline_cache_path,
    ensure_twcr_grid_window_cached,
    fetch_saved_twcr_temperature_series,
)

CALIBRATION_BASELINE_START = date(1961, 1, 1)
CALIBRATION_BASELINE_END = date(1990, 12, 31)
ERA5_BRIDGE_END = date(1949, 12, 31)
ERA5_LAND_START = date(1950, 1, 1)
TWCR_END = date(1939, 12, 31)
TWCR_SHARED_BBOX_MAX_GRID_CELLS = 16
TWCR_SHARED_BBOX_MAX_EXTRA_CELLS = 4


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
    uncached_twcr_fetches: int
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
    location_dependencies: dict[str, tuple[str, ...]]
    location_twcr_tasks: dict[str, tuple[SharedClimateTask, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class _TWCRRequestCandidate:
    location_key: str
    start_date: date
    end_date: date
    area: tuple[float, float, float, float]


@dataclass
class _TWCRRequestCluster:
    start_date: date
    end_date: date
    area: tuple[float, float, float, float]
    member_cell_sum: int
    members: list[_TWCRRequestCandidate]


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
    reference_start: date,
    reference_end: date,
    first_period_history_days: int = 0,
) -> list[LocationClimateRequest]:
    requests_by_location: dict[str, LocationClimateRequest] = {}
    if not periods:
        return []

    first_period = periods[0]
    active_ranges_by_location: dict[str, tuple[date, date]] = {}
    for period in periods:
        current_range = active_ranges_by_location.get(period.location_key)
        current_start = period.start_date
        if period.location_key == first_period.location_key and period.start_date == first_period.start_date:
            current_start = period.start_date - timedelta(days=first_period_history_days)
        if current_range is None:
            active_ranges_by_location[period.location_key] = (current_start, period.end_date)
            continue
        active_ranges_by_location[period.location_key] = (
            min(current_range[0], current_start),
            max(current_range[1], period.end_date),
        )

    for period in periods:
        active_start, active_end = active_ranges_by_location[period.location_key]
        requests_by_location.setdefault(
            period.location_key,
            LocationClimateRequest(
                location_key=period.location_key,
                display_name=period.display_name,
                latitude=period.latitude,
                longitude=period.longitude,
                boundary_geojson=period.boundary_geojson,
                boundary_bbox=period.bounding_box,
                start_date=min(active_start, reference_start),
                end_date=max(active_end, reference_end),
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
    uncached_twcr_fetches = sum(
        1
        for task in batch_plan.shared_tasks
        if task.source_id == TWCR_SOURCE_ID
    )
    uncached_twcr_years = {
        year
        for task in batch_plan.shared_tasks
        if task.source_id == TWCR_SOURCE_ID
        for year in _task_years(task)
    }

    return ClimateDownloadEstimate(
        uses_historical_fallback=uses_historical_fallback,
        uncached_era5_bridge_fetches=uncached_era5_bridge_fetches,
        uncached_twcr_fetches=uncached_twcr_fetches,
        uncached_twcr_years=tuple(sorted(uncached_twcr_years)),
    )


def build_climate_batch_plan(
    location_requests: Sequence[LocationClimateRequest],
    *,
    spatial_mode: str,
    radius_km: float | None = None,
) -> ClimateBatchPlan:
    shared_tasks: dict[str, SharedClimateTask] = {}
    location_dependencies: dict[str, set[str]] = {}
    twcr_candidates: list[_TWCRRequestCandidate] = []

    for request in location_requests:
        request_dependencies = location_dependencies.setdefault(request.location_key, set())
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
        land_slice_covers_baseline = _source_slices_include_anchor_land_baseline(source_slices)
        for source_slice in source_slices:
            if source_slice.source_id == "era5_land":
                _add_missing_cds_timeline_tasks(
                    shared_tasks,
                    location_dependencies=request_dependencies,
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
                    location_dependencies=request_dependencies,
                    request=request,
                    dataset=ERA5_MONTHLY_DATASET,
                    area=request_area_era5,
                    start_date=min(source_slice.start_date, CALIBRATION_BASELINE_START),
                    end_date=max(source_slice.end_date, CALIBRATION_BASELINE_END),
                    spatial_mode=spatial_mode,
                    radius_km=radius_km,
                )
                if not land_slice_covers_baseline:
                    _add_missing_cds_window_task(
                        shared_tasks,
                        location_dependencies=request_dependencies,
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
                _collect_missing_twcr_combined_requests(
                    twcr_candidates,
                    request=request,
                    visible_start_date=source_slice.start_date,
                    visible_end_date=source_slice.end_date,
                    spatial_mode=spatial_mode,
                    radius_km=radius_km,
                )
                if not land_slice_covers_baseline:
                    _add_missing_cds_window_task(
                        shared_tasks,
                        location_dependencies=request_dependencies,
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

    location_twcr_tasks = _register_twcr_shared_tasks(
        shared_tasks,
        location_dependencies=location_dependencies,
        candidates=twcr_candidates,
    )
    return ClimateBatchPlan(
        location_requests=tuple(location_requests),
        shared_tasks=tuple(shared_tasks.values()),
        location_dependencies={
            location_key: tuple(sorted(dependencies))
            for location_key, dependencies in location_dependencies.items()
        },
        location_twcr_tasks={
            location_key: tuple(
                sorted(
                    tasks,
                    key=lambda task: (
                        task.start_date or date.min,
                        task.end_date or date.min,
                        task.work_id,
                    ),
                )
            )
            for location_key, tasks in location_twcr_tasks.items()
        },
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
        shared_futures = {
            executor.submit(_run_shared_task, config, task, event_queue.put): task
            for task in batch_plan.shared_tasks
        }
        location_futures: dict[Future[Any], LocationClimateRequest] = {}
        pending_locations = {request.location_key: request for request in batch_plan.location_requests}
        completed_shared_work_ids: set[str] = set()
        batch_plan_location_dependencies = getattr(batch_plan, "location_dependencies", {}) or {}
        location_dependencies = {
            request.location_key: set(batch_plan_location_dependencies.get(request.location_key, ()))
            for request in batch_plan.location_requests
        }

        def _submit_ready_locations() -> None:
            for location_key, request in list(pending_locations.items()):
                dependencies = location_dependencies.get(location_key, set())
                if not dependencies.issubset(completed_shared_work_ids):
                    continue
                location_futures[
                    executor.submit(
                        _run_location_task,
                        config,
                        request,
                        spatial_mode,
                        radius_km,
                        batch_plan.location_twcr_tasks.get(location_key, ()),
                        event_queue.put,
                    )
                ] = request
                pending_locations.pop(location_key, None)

        _submit_ready_locations()

        while shared_futures or location_futures:
            pending = set(shared_futures) | set(location_futures)
            done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            _drain_progress_queue(event_queue, progress_callback)
            for future in done:
                if future in shared_futures:
                    task = shared_futures.pop(future)
                    future.result()
                    completed_shared_work_ids.add(task.work_id)
                    _submit_ready_locations()
                    continue

                request = location_futures.pop(future)
                results[request.location_key] = future.result()
            if not done:
                _drain_progress_queue(event_queue, progress_callback)

        while pending_locations:
            _, request = pending_locations.popitem()
            results[request.location_key] = _run_location_task(
                config,
                request,
                spatial_mode,
                radius_km,
                batch_plan.location_twcr_tasks.get(request.location_key, ()),
                event_queue.put,
            )

    _drain_progress_queue(event_queue, progress_callback)
    missing_locations = [
        request.location_key
        for request in batch_plan.location_requests
        if request.location_key not in results
    ]
    if missing_locations:
        raise ValueError(
            "Climate-data loading finished without timelines for: "
            + ", ".join(sorted(missing_locations))
        )
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
    elif task.task_kind == "twcr_grid":
        ensure_twcr_grid_window_cached(
            task.start_date,
            task.end_date,
            area=task.area,
            grid_request_cache_dir=TWCR_GRID_REQUEST_CACHE_DIR,
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
    twcr_request_tasks: Sequence[SharedClimateTask],
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
        twcr_request_tasks=twcr_request_tasks,
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


def _task_years(task: SharedClimateTask) -> tuple[int, ...]:
    if task.year is not None:
        return (int(task.year),)
    if task.start_date is None or task.end_date is None:
        return ()
    return tuple(range(task.start_date.year, task.end_date.year + 1))


def _source_slices_include_anchor_land_baseline(source_slices: Sequence[ClimateSourceSlice]) -> bool:
    return any(
        source_slice.source_id == "era5_land"
        and source_slice.start_date <= CALIBRATION_BASELINE_START
        and source_slice.end_date >= CALIBRATION_BASELINE_END
        for source_slice in source_slices
    )


def _anchor_land_source_slice(
    source_slices: Sequence[ClimateSourceSlice],
) -> ClimateSourceSlice | None:
    for source_slice in source_slices:
        if (
            source_slice.source_id == "era5_land"
            and source_slice.start_date <= CALIBRATION_BASELINE_START
            and source_slice.end_date >= CALIBRATION_BASELINE_END
        ):
            return source_slice
    return None


def _add_missing_cds_timeline_tasks(
    shared_tasks: dict[str, SharedClimateTask],
    *,
    location_dependencies: set[str],
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
        location_dependencies.add(str(work_path))
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
    location_dependencies: set[str],
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
    location_dependencies.add(str(work_path))
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


def _collect_missing_twcr_combined_requests(
    candidates: list[_TWCRRequestCandidate],
    *,
    request: LocationClimateRequest,
    visible_start_date: date,
    visible_end_date: date,
    spatial_mode: str,
    radius_km: float | None,
) -> None:
    fetch_start = min(visible_start_date, CALIBRATION_BASELINE_START)
    fetch_end = max(visible_end_date, CALIBRATION_BASELINE_END)
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
    missing_ranges = _missing_temperature_ranges(cached_frame, start_date=fetch_start, end_date=fetch_end)
    for missing_start, missing_end in missing_ranges:
        candidates.append(
            _TWCRRequestCandidate(
                location_key=request.location_key,
                start_date=missing_start,
                end_date=missing_end,
                area=_request_area_for_twcr_location_request(
                    request,
                    spatial_mode=spatial_mode,
                    radius_km=radius_km,
                ),
            )
        )


def _request_area_for_twcr_location_request(
    request: LocationClimateRequest,
    *,
    spatial_mode: str,
    radius_km: float | None,
) -> tuple[float, float, float, float]:
    return _request_area(
        latitude=request.latitude,
        longitude=request.longitude,
        spatial_mode=spatial_mode,
        radius_km=radius_km,
        boundary_geojson=request.boundary_geojson,
        boundary_bbox=request.boundary_bbox,
        grid_step_degrees=TWCR_GRID_STEP_DEGREES,
    )


def _register_twcr_shared_tasks(
    shared_tasks: dict[str, SharedClimateTask],
    *,
    location_dependencies: dict[str, set[str]],
    candidates: Sequence[_TWCRRequestCandidate],
) -> dict[str, list[SharedClimateTask]]:
    location_twcr_tasks: dict[str, list[SharedClimateTask]] = {}
    for cluster in _cluster_twcr_candidates(candidates):
        work_path = _twcr_grid_window_request_path(
            TWCR_GRID_REQUEST_CACHE_DIR,
            start_date=cluster.start_date,
            end_date=cluster.end_date,
            area=cluster.area,
        )
        task = SharedClimateTask(
            work_id=str(work_path),
            source_id=TWCR_SOURCE_ID,
            label=TWCR_DISPLAY_NAME,
            task_kind="twcr_grid",
            start_date=cluster.start_date,
            end_date=cluster.end_date,
            area=cluster.area,
        )
        if not work_path.exists():
            shared_tasks.setdefault(task.work_id, task)
        for candidate in cluster.members:
            location_twcr_tasks.setdefault(candidate.location_key, []).append(task)
            if not work_path.exists():
                location_dependencies.setdefault(candidate.location_key, set()).add(task.work_id)
    return location_twcr_tasks


def _cluster_twcr_candidates(
    candidates: Sequence[_TWCRRequestCandidate],
) -> list[_TWCRRequestCluster]:
    grouped: dict[tuple[date, date], list[_TWCRRequestCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault((candidate.start_date, candidate.end_date), []).append(candidate)

    clusters: list[_TWCRRequestCluster] = []
    for (start_date, end_date), window_candidates in grouped.items():
        window_clusters: list[_TWCRRequestCluster] = []
        for candidate in sorted(
            window_candidates,
            key=lambda item: (
                item.area[2],
                item.area[1],
                item.area[0],
                item.area[3],
                item.location_key,
            ),
        ):
            candidate_cells = _twcr_area_grid_cell_count(candidate.area)
            best_cluster: _TWCRRequestCluster | None = None
            best_union_area: tuple[float, float, float, float] | None = None
            best_union_cells: int | None = None
            for cluster in window_clusters:
                union_area = _merge_request_areas(cluster.area, candidate.area)
                union_cells = _twcr_area_grid_cell_count(union_area)
                if union_cells > TWCR_SHARED_BBOX_MAX_GRID_CELLS:
                    continue
                if union_cells > cluster.member_cell_sum + candidate_cells + TWCR_SHARED_BBOX_MAX_EXTRA_CELLS:
                    continue
                if best_union_cells is None or union_cells < best_union_cells:
                    best_cluster = cluster
                    best_union_area = union_area
                    best_union_cells = union_cells

            if best_cluster is None or best_union_area is None:
                window_clusters.append(
                    _TWCRRequestCluster(
                        start_date=start_date,
                        end_date=end_date,
                        area=candidate.area,
                        member_cell_sum=candidate_cells,
                        members=[candidate],
                    )
                )
                continue

            best_cluster.area = best_union_area
            best_cluster.member_cell_sum += candidate_cells
            best_cluster.members.append(candidate)
        clusters.extend(window_clusters)

    return clusters


def _merge_request_areas(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (
        max(left[0], right[0]),
        min(left[1], right[1]),
        min(left[2], right[2]),
        max(left[3], right[3]),
    )


def _twcr_area_grid_cell_count(area: tuple[float, float, float, float]) -> int:
    north, west, south, east = area
    latitude_cells = max(1, int(round((north - south) / TWCR_GRID_STEP_DEGREES)) + 1)
    longitude_cells = max(1, int(round((east - west) / TWCR_GRID_STEP_DEGREES)) + 1)
    return latitude_cells * longitude_cells


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
    twcr_request_tasks: Sequence[SharedClimateTask] = (),
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    source_slices = build_climate_source_slices(start_date, end_date)
    if not source_slices:
        raise ValueError("No climate-data source covers the requested period.")

    anchor_land_source_slice = _anchor_land_source_slice(source_slices)
    anchor_calibration_frame: pd.DataFrame | None = None
    source_calibration_frames: dict[str, pd.DataFrame] = {}
    source_frames: dict[str, pd.DataFrame] = {}
    frames: list[pd.DataFrame] = []

    def _get_anchor_land_frame(current_source_slice: ClimateSourceSlice) -> pd.DataFrame | None:
        if anchor_land_source_slice is None:
            return None
        if "era5_land" in source_frames:
            return source_frames["era5_land"]
        source_frames["era5_land"] = fetch_saved_temperature_series(
            config=config,
            latitude=latitude,
            longitude=longitude,
            start_date=anchor_land_source_slice.start_date,
            end_date=anchor_land_source_slice.end_date,
            dataset=ERA5_LAND_MONTHLY_DATASET,
            spatial_mode=spatial_mode,
            radius_km=radius_km,
            boundary_geojson=boundary_geojson,
            boundary_bbox=boundary_bbox,
            progress_callback=_forward_source_progress(progress_callback, current_source_slice),
        )
        return source_frames["era5_land"]

    def _ensure_anchor_calibration_frame(current_source_slice: ClimateSourceSlice) -> pd.DataFrame:
        nonlocal anchor_calibration_frame
        if anchor_calibration_frame is not None:
            return anchor_calibration_frame
        anchor_land_frame = _get_anchor_land_frame(current_source_slice)
        if anchor_land_frame is not None:
            anchor_calibration_frame = _slice_temperature_series(
                anchor_land_frame,
                start_date=CALIBRATION_BASELINE_START,
                end_date=CALIBRATION_BASELINE_END,
            )
            return anchor_calibration_frame
        anchor_calibration_frame = _fetch_anchor_calibration_frame(
            config=config,
            latitude=latitude,
            longitude=longitude,
            spatial_mode=spatial_mode,
            radius_km=radius_km,
            boundary_geojson=boundary_geojson,
            boundary_bbox=boundary_bbox,
            progress_callback=_forward_calibration_progress(progress_callback, current_source_slice),
        )
        return anchor_calibration_frame

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
            frame = source_frames.get("era5_land")
            if frame is None:
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
                source_frames["era5_land"] = frame
            if anchor_calibration_frame is None and anchor_land_source_slice == source_slice:
                anchor_calibration_frame = _slice_temperature_series(
                    frame,
                    start_date=CALIBRATION_BASELINE_START,
                    end_date=CALIBRATION_BASELINE_END,
                )
        elif source_slice.source_id == "era5_bridge":
            combined_era5_bridge_frame = fetch_saved_temperature_series(
                config=config,
                latitude=latitude,
                longitude=longitude,
                start_date=min(source_slice.start_date, CALIBRATION_BASELINE_START),
                end_date=max(source_slice.end_date, CALIBRATION_BASELINE_END),
                dataset=ERA5_MONTHLY_DATASET,
                spatial_mode=spatial_mode,
                radius_km=radius_km,
                boundary_geojson=boundary_geojson,
                boundary_bbox=boundary_bbox,
                progress_callback=_forward_source_progress(progress_callback, source_slice),
            )
            raw_frame = _slice_temperature_series(
                combined_era5_bridge_frame,
                start_date=source_slice.start_date,
                end_date=source_slice.end_date,
            )
            source_calibration_frames.setdefault(
                source_slice.source_id,
                _slice_temperature_series(
                    combined_era5_bridge_frame,
                    start_date=CALIBRATION_BASELINE_START,
                    end_date=CALIBRATION_BASELINE_END,
                ),
            )
            frame = _calibrate_source_monthly_frame(
                raw_frame,
                anchor_calibration_frame=_ensure_anchor_calibration_frame(source_slice),
                source_calibration_frame=source_calibration_frames[source_slice.source_id],
            )
        elif source_slice.source_id == TWCR_SOURCE_ID:
            twcr_request_area_overrides = {
                (task.start_date, task.end_date): task.area
                for task in twcr_request_tasks
                if task.source_id == TWCR_SOURCE_ID
                and task.start_date is not None
                and task.end_date is not None
                and task.area is not None
            }
            combined_twcr_frame = fetch_saved_twcr_temperature_series(
                latitude=latitude,
                longitude=longitude,
                start_date=min(source_slice.start_date, CALIBRATION_BASELINE_START),
                end_date=max(source_slice.end_date, CALIBRATION_BASELINE_END),
                spatial_mode=spatial_mode,
                radius_km=radius_km,
                boundary_geojson=boundary_geojson,
                boundary_bbox=boundary_bbox,
                request_area_overrides=twcr_request_area_overrides,
                progress_callback=_forward_source_progress(progress_callback, source_slice),
            )
            raw_frame = _slice_temperature_series(
                combined_twcr_frame,
                start_date=source_slice.start_date,
                end_date=source_slice.end_date,
            )
            source_calibration_frames.setdefault(
                source_slice.source_id,
                _slice_temperature_series(
                    combined_twcr_frame,
                    start_date=CALIBRATION_BASELINE_START,
                    end_date=CALIBRATION_BASELINE_END,
                ),
            )
            frame = _calibrate_source_monthly_frame(
                raw_frame,
                anchor_calibration_frame=_ensure_anchor_calibration_frame(source_slice),
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
    if not estimate.uncached_era5_bridge_fetches and not estimate.uncached_twcr_fetches:
        return None

    twcr_fetch_count = int(estimate.uncached_twcr_fetches)
    level = "warning" if twcr_fetch_count > 10 or estimate.uncached_era5_bridge_fetches > 2 else "info"
    message_parts: list[str] = []
    if twcr_fetch_count:
        noun = "bbox download" if twcr_fetch_count == 1 else "bbox downloads"
        message_parts.append(f"{twcr_fetch_count} uncached 20CRv3 {noun}")
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
