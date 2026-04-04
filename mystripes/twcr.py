from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from mystripes.cds import (
    CDSRequestError,
    ProgressCallback,
    TEMPERATURE_CACHE_DIR,
    _aggregate_spatial_selection,
    cache_path_lock,
    _emit_progress,
    _load_cached_temperature_series,
    _missing_temperature_ranges,
    _request_area,
    _slice_temperature_series,
    _store_cached_temperature_series,
    parse_temperature_file,
)

TWCR_SOURCE_ID = "20crv3"
TWCR_DISPLAY_NAME = "NOAA 20CRv3 monthly 2 m air temperature"
TWCR_DATASET_URL = "https://psl.noaa.gov/thredds/catalog/Datasets/20thC_ReanV3/Monthlies/2mSI-MO/catalog.html"
TWCR_NCSS_URL = (
    "https://psl.noaa.gov/thredds/ncss/grid/"
    "Datasets/20thC_ReanV3/Monthlies/2mSI-MO/air.2m.mon.mean.nc"
)
TWCR_MIN_START = date(1836, 1, 1)
TWCR_MAX_END = date(2015, 12, 31)
TWCR_CACHE_DIR = TEMPERATURE_CACHE_DIR / TWCR_SOURCE_ID
TWCR_REQUEST_CACHE_DIR = TWCR_CACHE_DIR / "window-cache"
TWCR_TIMELINE_CACHE_DIR = TWCR_CACHE_DIR / "timeline-cache"
TWCR_GRID_REQUEST_CACHE_DIR = TWCR_CACHE_DIR / "grid-year-requests"
TWCR_GRID_STEP_DEGREES = 1.0


@dataclass(frozen=True)
class TWCRDownloadWork:
    work_id: str
    task_kind: str
    year: int | None = None
    start_date: date | None = None
    end_date: date | None = None
    area: tuple[float, float, float, float] | None = None


def fetch_twcr_temperature_series(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    spatial_mode: str = "single_cell",
    radius_km: float | None = None,
    boundary_geojson: dict[str, Any] | None = None,
    boundary_bbox: tuple[float, float, float, float] | None = None,
    request_area_override: tuple[float, float, float, float] | None = None,
    cache_dir: Path | None = TWCR_REQUEST_CACHE_DIR,
    grid_request_cache_dir: Path | None = TWCR_GRID_REQUEST_CACHE_DIR,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    _validate_twcr_date_range(start_date, end_date)

    cache_path = None
    if cache_dir is not None:
        cache_path = _twcr_window_cache_path(
            cache_dir=cache_dir,
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            spatial_mode=spatial_mode,
            radius_km=radius_km,
            boundary_geojson=boundary_geojson,
            boundary_bbox=boundary_bbox,
        )
        cached_frame = _load_cached_temperature_series(cache_path)
        if cached_frame is not None:
            _emit_progress(
                progress_callback,
                "request_cache_hit",
                dataset=TWCR_SOURCE_ID,
                dataset_label=TWCR_DISPLAY_NAME,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                request_count=0,
            )
            return cached_frame

    _emit_progress(
        progress_callback,
        "request_plan",
        dataset=TWCR_SOURCE_ID,
        dataset_label=TWCR_DISPLAY_NAME,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        request_count=1,
    )

    request_area = request_area_override or _twcr_request_area(
        latitude=latitude,
        longitude=longitude,
        spatial_mode=spatial_mode,
        radius_km=radius_km,
        boundary_geojson=boundary_geojson,
        boundary_bbox=boundary_bbox,
    )
    request_path, request_scope, request_origin = _resolve_twcr_window_source_path(
        start_date=start_date,
        end_date=end_date,
        request_area=request_area,
        grid_request_cache_dir=grid_request_cache_dir,
        progress_callback=progress_callback,
        request_index=1,
        request_count=1,
    )
    _emit_progress(
        progress_callback,
        "request_started",
        dataset=TWCR_SOURCE_ID,
        dataset_label=TWCR_DISPLAY_NAME,
        request_index=1,
        request_count=1,
        request_year_start=str(start_date.year),
        request_year_end=str(end_date.year),
        request_start=start_date.isoformat(),
        request_end=end_date.isoformat(),
        month_count=_month_count_in_range(start_date, end_date),
        request_origin=request_origin,
        request_scope=request_scope,
        work_id=str(request_path),
    )

    grid_frame, request_path, request_scope, request_origin = _load_twcr_grid_frame_with_recovery(
        start_date=start_date,
        end_date=end_date,
        request_path=request_path,
        request_scope=request_scope,
        request_origin=request_origin,
        request_area=request_area,
        target_latitude=latitude,
        target_longitude=longitude,
        grid_request_cache_dir=grid_request_cache_dir,
        progress_callback=progress_callback,
        request_index=1,
        request_count=1,
    )
    combined = _aggregate_spatial_selection(
        grid_frame=grid_frame,
        latitude=latitude,
        longitude=longitude,
        spatial_mode=spatial_mode,
        radius_km=radius_km,
        boundary_geojson=boundary_geojson,
        boundary_bbox=boundary_bbox,
    )
    _emit_progress(
        progress_callback,
        "request_finished",
        dataset=TWCR_SOURCE_ID,
        dataset_label=TWCR_DISPLAY_NAME,
        request_index=1,
        request_count=1,
        request_year_start=str(start_date.year),
        request_year_end=str(end_date.year),
        request_start=start_date.isoformat(),
        request_end=end_date.isoformat(),
        month_count=_month_count_in_range(start_date, end_date),
        request_origin=request_origin,
        request_scope=request_scope,
        work_id=str(request_path),
    )

    combined = _slice_temperature_series(combined, start_date=start_date, end_date=end_date)
    if cache_path is not None:
        _store_cached_temperature_series(cache_path, combined)
    _emit_progress(
        progress_callback,
        "point_fetch_completed",
        dataset=TWCR_SOURCE_ID,
        dataset_label=TWCR_DISPLAY_NAME,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        request_count=1,
    )
    return combined


def fetch_saved_twcr_temperature_series(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    spatial_mode: str = "single_cell",
    radius_km: float | None = None,
    boundary_geojson: dict[str, Any] | None = None,
    boundary_bbox: tuple[float, float, float, float] | None = None,
    request_area_overrides: Mapping[tuple[date, date], tuple[float, float, float, float]] | None = None,
    cache_dir: Path | None = TWCR_TIMELINE_CACHE_DIR,
    request_cache_dir: Path | None = TWCR_REQUEST_CACHE_DIR,
    grid_request_cache_dir: Path | None = TWCR_GRID_REQUEST_CACHE_DIR,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    _validate_twcr_date_range(start_date, end_date)

    cache_path = None
    cached_frame = None
    if cache_dir is not None:
        cache_path = _twcr_timeline_cache_path(
            cache_dir=cache_dir,
            latitude=latitude,
            longitude=longitude,
            spatial_mode=spatial_mode,
            radius_km=radius_km,
            boundary_geojson=boundary_geojson,
            boundary_bbox=boundary_bbox,
        )
        cached_frame = _load_cached_temperature_series(cache_path)

    missing_ranges = _missing_temperature_ranges(cached_frame, start_date=start_date, end_date=end_date)
    if not missing_ranges:
        if cached_frame is None:
            raise CDSRequestError("20CRv3 cache reported full coverage but no cached data was loaded.")
        _emit_progress(
            progress_callback,
            "timeline_cache_hit",
            dataset=TWCR_SOURCE_ID,
            dataset_label=TWCR_DISPLAY_NAME,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            missing_range_count=0,
        )
        return _slice_temperature_series(cached_frame, start_date=start_date, end_date=end_date)

    _emit_progress(
        progress_callback,
        "timeline_fetch_plan",
        dataset=TWCR_SOURCE_ID,
        dataset_label=TWCR_DISPLAY_NAME,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        missing_range_count=len(missing_ranges),
        has_cached_data=bool(cached_frame is not None and not cached_frame.empty),
    )

    frames: list[pd.DataFrame] = []
    if cached_frame is not None and not cached_frame.empty:
        frames.append(cached_frame)

    total_missing_ranges = len(missing_ranges)
    for range_index, (missing_start, missing_end) in enumerate(missing_ranges, start=1):
        _emit_progress(
            progress_callback,
            "missing_range_started",
            dataset=TWCR_SOURCE_ID,
            dataset_label=TWCR_DISPLAY_NAME,
            range_index=range_index,
            range_count=total_missing_ranges,
            range_start=missing_start.isoformat(),
            range_end=missing_end.isoformat(),
        )

        def _forward_progress(event: dict[str, Any]) -> None:
            if progress_callback is None:
                return
            forwarded_event = dict(event)
            forwarded_event.setdefault("range_index", range_index)
            forwarded_event.setdefault("range_count", total_missing_ranges)
            forwarded_event.setdefault("range_start", missing_start.isoformat())
            forwarded_event.setdefault("range_end", missing_end.isoformat())
            progress_callback(forwarded_event)

        frames.append(
            fetch_twcr_temperature_series(
                latitude=latitude,
                longitude=longitude,
                start_date=missing_start,
                end_date=missing_end,
                spatial_mode=spatial_mode,
                radius_km=radius_km,
                boundary_geojson=boundary_geojson,
                boundary_bbox=boundary_bbox,
                request_area_override=None
                if request_area_overrides is None
                else request_area_overrides.get((missing_start, missing_end)),
                cache_dir=request_cache_dir,
                grid_request_cache_dir=grid_request_cache_dir,
                progress_callback=_forward_progress,
            )
        )
        _emit_progress(
            progress_callback,
            "missing_range_finished",
            dataset=TWCR_SOURCE_ID,
            dataset_label=TWCR_DISPLAY_NAME,
            range_index=range_index,
            range_count=total_missing_ranges,
            range_start=missing_start.isoformat(),
            range_end=missing_end.isoformat(),
        )

    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp")
    combined = combined.reset_index(drop=True)
    if cache_path is not None:
        _store_cached_temperature_series(cache_path, combined)
    _emit_progress(
        progress_callback,
        "timeline_fetch_completed",
        dataset=TWCR_SOURCE_ID,
        dataset_label=TWCR_DISPLAY_NAME,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        missing_range_count=total_missing_ranges,
    )
    return _slice_temperature_series(combined, start_date=start_date, end_date=end_date)


def plan_twcr_downloads(
    *,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    spatial_mode: str,
    radius_km: float | None,
    boundary_geojson: dict[str, Any] | None,
    boundary_bbox: tuple[float, float, float, float] | None,
    grid_request_cache_dir: Path | None = TWCR_GRID_REQUEST_CACHE_DIR,
) -> tuple[TWCRDownloadWork, ...]:
    _validate_twcr_date_range(start_date, end_date)
    request_area = _twcr_request_area(
        latitude=latitude,
        longitude=longitude,
        spatial_mode=spatial_mode,
        radius_km=radius_km,
        boundary_geojson=boundary_geojson,
        boundary_bbox=boundary_bbox,
    )
    work = _twcr_download_work_for_window(
        start_date=start_date,
        end_date=end_date,
        request_area=request_area,
        grid_request_cache_dir=grid_request_cache_dir,
    )
    return () if work is None else (work,)


def _validate_twcr_date_range(start_date: date, end_date: date) -> None:
    if start_date > end_date:
        raise ValueError("Start date must be on or before end date.")
    if start_date < TWCR_MIN_START or end_date > TWCR_MAX_END:
        raise ValueError(
            f"20CRv3 only covers {TWCR_MIN_START.isoformat()} to {TWCR_MAX_END.isoformat()}."
        )


def _month_count_in_range(start_date: date, end_date: date) -> int:
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1


def _download_twcr_subset_window_file(
    start_date: date,
    end_date: date,
    area: tuple[float, float, float, float],
    target_path: Path,
) -> None:
    north, west, south, east = area
    query = {
        "var": "air",
        "north": north,
        "west": west,
        "south": south,
        "east": east,
        "time_start": f"{start_date.isoformat()}T00:00:00Z",
        "time_end": f"{end_date.isoformat()}T23:59:59Z",
        "horizStride": 1,
        "timeStride": 1,
        "accept": "netcdf4",
    }
    _download_twcr_file(query, target_path)


def _download_twcr_file(query: dict[str, Any], target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(TWCR_NCSS_URL, params=query, stream=True, timeout=(30, 300))
    response.raise_for_status()

    temporary_path = target_path.with_suffix(".tmp")
    with temporary_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    temporary_path.replace(target_path)
def ensure_twcr_grid_window_cached(
    start_date: date,
    end_date: date,
    area: tuple[float, float, float, float],
    grid_request_cache_dir: Path | None = TWCR_GRID_REQUEST_CACHE_DIR,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    return _ensure_twcr_grid_window_cached(
        start_date=start_date,
        end_date=end_date,
        area=area,
        grid_request_cache_dir=grid_request_cache_dir,
        progress_callback=progress_callback,
        request_index=1,
        request_count=1,
    )
def _ensure_twcr_grid_window_cached(
    *,
    start_date: date,
    end_date: date,
    area: tuple[float, float, float, float],
    grid_request_cache_dir: Path | None,
    progress_callback: ProgressCallback | None,
    request_index: int,
    request_count: int,
) -> Path:
    grid_request_path = _twcr_grid_window_request_path(
        grid_request_cache_dir,
        start_date=start_date,
        end_date=end_date,
        area=area,
    )
    if grid_request_path.exists():
        return grid_request_path

    with cache_path_lock(grid_request_path):
        if grid_request_path.exists():
            _emit_progress(
                progress_callback,
                "request_cache_hit",
                dataset=TWCR_SOURCE_ID,
                dataset_label=TWCR_DISPLAY_NAME,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                request_count=0,
                cache_scope="shared_grid_window",
                work_id=str(grid_request_path),
            )
            return grid_request_path

        try:
            _download_twcr_subset_window_file(start_date, end_date, area, grid_request_path)
        except Exception as exc:  # pragma: no cover - depends on external service.
            _emit_progress(
                progress_callback,
                "request_failed",
                dataset=TWCR_SOURCE_ID,
                dataset_label=TWCR_DISPLAY_NAME,
                request_index=request_index,
                request_count=request_count,
                request_year_start=str(start_date.year),
                request_year_end=str(end_date.year),
                request_start=start_date.isoformat(),
                request_end=end_date.isoformat(),
                month_count=_month_count_in_range(start_date, end_date),
                message=(
                    f"20CRv3 bbox download for {start_date.isoformat()} to "
                    f"{end_date.isoformat()} failed: {exc}"
                ),
                cache_scope="shared_grid_window",
                work_id=str(grid_request_path),
            )
            raise CDSRequestError(
                f"20CRv3 bbox download for {start_date.isoformat()} to {end_date.isoformat()} failed: {exc}"
            ) from exc
    return grid_request_path
def _twcr_grid_window_request_path(
    grid_request_cache_dir: Path | None,
    *,
    start_date: date,
    end_date: date,
    area: tuple[float, float, float, float],
) -> Path:
    if grid_request_cache_dir is None:
        raise ValueError("A grid-request cache directory is required for 20CRv3 subset downloads.")
    payload = {
        "cache_version": 2,
        "cache_type": "grid_window",
        "dataset": TWCR_SOURCE_ID,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "area": [round(value, 6) for value in area],
    }
    key = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return grid_request_cache_dir / f"{key}.nc"


def _twcr_timeline_cache_path(
    cache_dir: Path,
    latitude: float,
    longitude: float,
    spatial_mode: str,
    radius_km: float | None,
    boundary_geojson: dict[str, Any] | None,
    boundary_bbox: tuple[float, float, float, float] | None,
) -> Path:
    payload = {
        "cache_version": 1,
        "cache_type": "timeline",
        "dataset": TWCR_SOURCE_ID,
        "latitude": round(latitude, 6),
        "longitude": round(longitude, 6),
        "spatial_mode": spatial_mode,
        "radius_km": None if radius_km is None else round(radius_km, 6),
        "boundary_geojson": boundary_geojson,
        "boundary_bbox": list(boundary_bbox) if boundary_bbox is not None else None,
    }
    return _hashed_cache_path(cache_dir, payload)


def _twcr_window_cache_path(
    cache_dir: Path,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    spatial_mode: str,
    radius_km: float | None,
    boundary_geojson: dict[str, Any] | None,
    boundary_bbox: tuple[float, float, float, float] | None,
) -> Path:
    payload = {
        "cache_version": 1,
        "cache_type": "window",
        "dataset": TWCR_SOURCE_ID,
        "latitude": round(latitude, 6),
        "longitude": round(longitude, 6),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "spatial_mode": spatial_mode,
        "radius_km": None if radius_km is None else round(radius_km, 6),
        "boundary_geojson": boundary_geojson,
        "boundary_bbox": list(boundary_bbox) if boundary_bbox is not None else None,
    }
    return _hashed_cache_path(cache_dir, payload)


def _hashed_cache_path(cache_dir: Path, payload: dict[str, Any]) -> Path:
    key = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return cache_dir / f"{key}.csv"


def _twcr_request_area(
    *,
    latitude: float,
    longitude: float,
    spatial_mode: str,
    radius_km: float | None,
    boundary_geojson: dict[str, Any] | None,
    boundary_bbox: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float]:
    return _request_area(
        latitude=latitude,
        longitude=longitude,
        spatial_mode=spatial_mode,
        radius_km=radius_km,
        boundary_geojson=boundary_geojson,
        boundary_bbox=boundary_bbox,
        grid_step_degrees=TWCR_GRID_STEP_DEGREES,
    )


def _resolve_twcr_window_source_path(
    *,
    start_date: date,
    end_date: date,
    request_area: tuple[float, float, float, float],
    grid_request_cache_dir: Path | None,
    progress_callback: ProgressCallback | None,
    request_index: int,
    request_count: int,
) -> tuple[Path, str, str]:
    grid_request_path = _twcr_grid_window_request_path(
        grid_request_cache_dir=grid_request_cache_dir,
        start_date=start_date,
        end_date=end_date,
        area=request_area,
    )
    if grid_request_path.exists():
        return grid_request_path, "bbox_window", "local_cache"

    work = _twcr_download_work_for_window(
        start_date=start_date,
        end_date=end_date,
        request_area=request_area,
        grid_request_cache_dir=grid_request_cache_dir,
    )
    if work is None:
        return grid_request_path, "bbox_window", "local_cache"

    request_path = _ensure_twcr_grid_window_cached(
        start_date=start_date,
        end_date=end_date,
        area=request_area,
        grid_request_cache_dir=grid_request_cache_dir,
        progress_callback=progress_callback,
        request_index=request_index,
        request_count=request_count,
    )
    return request_path, "bbox_window", "download"


def _load_twcr_grid_frame_with_recovery(
    *,
    start_date: date,
    end_date: date,
    request_path: Path,
    request_scope: str,
    request_origin: str,
    request_area: tuple[float, float, float, float],
    target_latitude: float,
    target_longitude: float,
    grid_request_cache_dir: Path | None,
    progress_callback: ProgressCallback | None,
    request_index: int,
    request_count: int,
) -> tuple[pd.DataFrame, Path, str, str]:
    try:
        frame = parse_temperature_file(
            request_path,
            target_latitude=target_latitude,
            target_longitude=target_longitude,
        )
        return frame, request_path, request_scope, request_origin
    except (CDSRequestError, OSError, RuntimeError, ValueError) as exc:
        recovered_path, recovered_scope, recovered_origin = _recover_twcr_request_path(
            start_date=start_date,
            end_date=end_date,
            request_path=request_path,
            request_scope=request_scope,
            request_area=request_area,
            grid_request_cache_dir=grid_request_cache_dir,
            progress_callback=progress_callback,
            request_index=request_index,
            request_count=request_count,
            error=exc,
        )
        try:
            frame = parse_temperature_file(
                recovered_path,
                target_latitude=target_latitude,
                target_longitude=target_longitude,
            )
            return frame, recovered_path, recovered_scope, recovered_origin
        except (CDSRequestError, OSError, RuntimeError, ValueError) as recovered_exc:
            raise CDSRequestError(
                "20CRv3 data for "
                f"{start_date.isoformat()} to {end_date.isoformat()} could not be read after retry: "
                f"{recovered_exc}"
            ) from recovered_exc


def _recover_twcr_request_path(
    *,
    start_date: date,
    end_date: date,
    request_path: Path,
    request_scope: str,
    request_area: tuple[float, float, float, float],
    grid_request_cache_dir: Path | None,
    progress_callback: ProgressCallback | None,
    request_index: int,
    request_count: int,
    error: Exception,
) -> tuple[Path, str, str]:
    _drop_twcr_cached_file(request_path)

    _emit_progress(
        progress_callback,
        "request_recovery",
        dataset=TWCR_SOURCE_ID,
        dataset_label=TWCR_DISPLAY_NAME,
        request_index=request_index,
        request_count=request_count,
        request_year_start=str(start_date.year),
        request_year_end=str(end_date.year),
        request_start=start_date.isoformat(),
        request_end=end_date.isoformat(),
        month_count=_month_count_in_range(start_date, end_date),
        request_scope=request_scope,
        work_id=str(request_path),
        message=(
            "Redownloading unreadable 20CRv3 bbox request for "
            f"{start_date.isoformat()} to {end_date.isoformat()}: {error}"
        ),
    )
    grid_request_path = _ensure_twcr_grid_window_cached(
        start_date=start_date,
        end_date=end_date,
        area=request_area,
        grid_request_cache_dir=grid_request_cache_dir,
        progress_callback=progress_callback,
        request_index=request_index,
        request_count=request_count,
    )
    return grid_request_path, "bbox_window", "download"


def _twcr_download_work_for_window(
    *,
    start_date: date,
    end_date: date,
    request_area: tuple[float, float, float, float],
    grid_request_cache_dir: Path | None,
) -> TWCRDownloadWork | None:
    grid_request_path = _twcr_grid_window_request_path(
        grid_request_cache_dir,
        start_date=start_date,
        end_date=end_date,
        area=request_area,
    )
    if grid_request_path.exists():
        return None
    return TWCRDownloadWork(
        work_id=str(grid_request_path),
        task_kind="twcr_grid",
        year=None,
        start_date=start_date,
        end_date=end_date,
        area=request_area,
    )
def _drop_twcr_cached_file(path: Path) -> None:
    if not path.exists():
        return
    with cache_path_lock(path):
        path.unlink(missing_ok=True)
