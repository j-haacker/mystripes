from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any
from dataclasses import dataclass

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
TWCR_RAW_YEAR_CACHE_DIR = TWCR_CACHE_DIR / "raw-year-files"
TWCR_GRID_REQUEST_CACHE_DIR = TWCR_CACHE_DIR / "grid-year-requests"
TWCR_GRID_STEP_DEGREES = 1.0
TWCR_SUBSET_MAX_GRID_CELLS = 64


@dataclass(frozen=True)
class TWCRDownloadWork:
    work_id: str
    task_kind: str
    year: int
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
    cache_dir: Path | None = TWCR_REQUEST_CACHE_DIR,
    grid_request_cache_dir: Path | None = TWCR_GRID_REQUEST_CACHE_DIR,
    raw_year_cache_dir: Path | None = TWCR_RAW_YEAR_CACHE_DIR,
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

    years = _years_in_range(start_date, end_date)
    _emit_progress(
        progress_callback,
        "request_plan",
        dataset=TWCR_SOURCE_ID,
        dataset_label=TWCR_DISPLAY_NAME,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        request_count=len(years),
    )

    request_area = _twcr_request_area(
        latitude=latitude,
        longitude=longitude,
        spatial_mode=spatial_mode,
        radius_km=radius_km,
        boundary_geojson=boundary_geojson,
        boundary_bbox=boundary_bbox,
    )
    frames: list[pd.DataFrame] = []
    for request_index, year in enumerate(years, start=1):
        request_path, request_scope, request_origin = _resolve_twcr_year_source_path(
            year,
            request_area=request_area,
            raw_year_cache_dir=raw_year_cache_dir,
            grid_request_cache_dir=grid_request_cache_dir,
            progress_callback=progress_callback,
            request_index=request_index,
            request_count=len(years),
        )
        _emit_progress(
            progress_callback,
            "request_started",
            dataset=TWCR_SOURCE_ID,
            dataset_label=TWCR_DISPLAY_NAME,
            request_index=request_index,
            request_count=len(years),
            request_year_start=str(year),
            request_year_end=str(year),
            month_count=12,
            request_origin=request_origin,
            request_scope=request_scope,
            work_id=str(request_path),
        )

        grid_frame = parse_temperature_file(
            request_path,
            target_latitude=latitude,
            target_longitude=longitude,
        )
        frames.append(
            _aggregate_spatial_selection(
                grid_frame=grid_frame,
                latitude=latitude,
                longitude=longitude,
                spatial_mode=spatial_mode,
                radius_km=radius_km,
                boundary_geojson=boundary_geojson,
                boundary_bbox=boundary_bbox,
            )
        )
        _emit_progress(
            progress_callback,
            "request_finished",
            dataset=TWCR_SOURCE_ID,
            dataset_label=TWCR_DISPLAY_NAME,
            request_index=request_index,
            request_count=len(years),
            request_year_start=str(year),
            request_year_end=str(year),
            month_count=12,
            request_origin=request_origin,
            request_scope=request_scope,
            work_id=str(request_path),
        )

    if not frames:
        raise CDSRequestError("20CRv3 returned no monthly data.")

    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp")
    combined = combined.reset_index(drop=True)
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
        request_count=len(years),
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
    cache_dir: Path | None = TWCR_TIMELINE_CACHE_DIR,
    request_cache_dir: Path | None = TWCR_REQUEST_CACHE_DIR,
    grid_request_cache_dir: Path | None = TWCR_GRID_REQUEST_CACHE_DIR,
    raw_year_cache_dir: Path | None = TWCR_RAW_YEAR_CACHE_DIR,
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
                cache_dir=request_cache_dir,
                grid_request_cache_dir=grid_request_cache_dir,
                raw_year_cache_dir=raw_year_cache_dir,
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


def estimate_missing_twcr_years(
    start_date: date,
    end_date: date,
    raw_year_cache_dir: Path | None = TWCR_RAW_YEAR_CACHE_DIR,
) -> list[int]:
    _validate_twcr_date_range(start_date, end_date)
    if raw_year_cache_dir is None:
        return _years_in_range(start_date, end_date)
    return [
        year
        for year in _years_in_range(start_date, end_date)
        if not _twcr_raw_year_path(raw_year_cache_dir, year).exists()
    ]


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
    raw_year_cache_dir: Path | None = TWCR_RAW_YEAR_CACHE_DIR,
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
    works: list[TWCRDownloadWork] = []
    for year in _years_in_range(start_date, end_date):
        work = _twcr_download_work_for_year(
            year,
            request_area=request_area,
            raw_year_cache_dir=raw_year_cache_dir,
            grid_request_cache_dir=grid_request_cache_dir,
        )
        if work is not None:
            works.append(work)
    return tuple(works)


def _validate_twcr_date_range(start_date: date, end_date: date) -> None:
    if start_date > end_date:
        raise ValueError("Start date must be on or before end date.")
    if start_date < TWCR_MIN_START or end_date > TWCR_MAX_END:
        raise ValueError(
            f"20CRv3 only covers {TWCR_MIN_START.isoformat()} to {TWCR_MAX_END.isoformat()}."
        )


def _years_in_range(start_date: date, end_date: date) -> list[int]:
    return list(range(start_date.year, end_date.year + 1))


def _download_twcr_year_file(year: int, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    query = {
        "var": "air",
        "time_start": f"{year:04d}-01-01T00:00:00Z",
        "time_end": f"{year:04d}-12-31T23:59:59Z",
        "horizStride": 1,
        "timeStride": 1,
        "accept": "netcdf4",
    }
    _download_twcr_file(query, target_path)


def _download_twcr_subset_year_file(
    year: int,
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
        "time_start": f"{year:04d}-01-01T00:00:00Z",
        "time_end": f"{year:04d}-12-31T23:59:59Z",
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


def ensure_twcr_year_cached(
    year: int,
    raw_year_cache_dir: Path | None = TWCR_RAW_YEAR_CACHE_DIR,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    return _ensure_twcr_year_cached(
        year,
        raw_year_cache_dir=raw_year_cache_dir,
        progress_callback=progress_callback,
        request_index=1,
        request_count=1,
    )


def ensure_twcr_grid_year_cached(
    year: int,
    area: tuple[float, float, float, float],
    grid_request_cache_dir: Path | None = TWCR_GRID_REQUEST_CACHE_DIR,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    return _ensure_twcr_grid_year_cached(
        year,
        area=area,
        grid_request_cache_dir=grid_request_cache_dir,
        progress_callback=progress_callback,
        request_index=1,
        request_count=1,
    )


def _ensure_twcr_year_cached(
    year: int,
    *,
    raw_year_cache_dir: Path | None,
    progress_callback: ProgressCallback | None,
    request_index: int,
    request_count: int,
) -> Path:
    raw_year_path = _twcr_raw_year_path(raw_year_cache_dir, year)
    if raw_year_path.exists():
        return raw_year_path

    with cache_path_lock(raw_year_path):
        if raw_year_path.exists():
            _emit_progress(
                progress_callback,
                "request_cache_hit",
                dataset=TWCR_SOURCE_ID,
                dataset_label=TWCR_DISPLAY_NAME,
                start_date=f"{year:04d}-01-01",
                end_date=f"{year:04d}-12-31",
                request_count=0,
                cache_scope="shared_year",
                work_id=str(raw_year_path),
            )
            return raw_year_path

        try:
            _download_twcr_year_file(year, raw_year_path)
        except Exception as exc:  # pragma: no cover - depends on external service.
            _emit_progress(
                progress_callback,
                "request_failed",
                dataset=TWCR_SOURCE_ID,
                dataset_label=TWCR_DISPLAY_NAME,
                request_index=request_index,
                request_count=request_count,
                request_year_start=str(year),
                request_year_end=str(year),
                month_count=12,
                message=f"20CRv3 download for {year} failed: {exc}",
                cache_scope="shared_year",
                work_id=str(raw_year_path),
            )
            raise CDSRequestError(f"20CRv3 download for {year} failed: {exc}") from exc
    return raw_year_path


def _ensure_twcr_grid_year_cached(
    year: int,
    *,
    area: tuple[float, float, float, float],
    grid_request_cache_dir: Path | None,
    progress_callback: ProgressCallback | None,
    request_index: int,
    request_count: int,
) -> Path:
    grid_request_path = _twcr_grid_request_path(grid_request_cache_dir, year, area)
    if grid_request_path.exists():
        return grid_request_path

    with cache_path_lock(grid_request_path):
        if grid_request_path.exists():
            _emit_progress(
                progress_callback,
                "request_cache_hit",
                dataset=TWCR_SOURCE_ID,
                dataset_label=TWCR_DISPLAY_NAME,
                start_date=f"{year:04d}-01-01",
                end_date=f"{year:04d}-12-31",
                request_count=0,
                cache_scope="shared_grid",
                work_id=str(grid_request_path),
            )
            return grid_request_path

        try:
            _download_twcr_subset_year_file(year, area, grid_request_path)
        except Exception as exc:  # pragma: no cover - depends on external service.
            _emit_progress(
                progress_callback,
                "request_failed",
                dataset=TWCR_SOURCE_ID,
                dataset_label=TWCR_DISPLAY_NAME,
                request_index=request_index,
                request_count=request_count,
                request_year_start=str(year),
                request_year_end=str(year),
                month_count=12,
                message=f"20CRv3 subset download for {year} failed: {exc}",
                cache_scope="shared_grid",
                work_id=str(grid_request_path),
            )
            raise CDSRequestError(f"20CRv3 subset download for {year} failed: {exc}") from exc
    return grid_request_path


def _twcr_raw_year_path(raw_year_cache_dir: Path | None, year: int) -> Path:
    if raw_year_cache_dir is None:
        raise ValueError("A raw-year cache directory is required for 20CRv3 downloads.")
    return raw_year_cache_dir / f"air.2m.mon.mean.{year:04d}.nc"


def _twcr_grid_request_path(
    grid_request_cache_dir: Path | None,
    year: int,
    area: tuple[float, float, float, float],
) -> Path:
    if grid_request_cache_dir is None:
        raise ValueError("A grid-request cache directory is required for 20CRv3 subset downloads.")
    payload = {
        "cache_version": 1,
        "cache_type": "grid_year",
        "dataset": TWCR_SOURCE_ID,
        "year": year,
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


def _resolve_twcr_year_source_path(
    year: int,
    *,
    request_area: tuple[float, float, float, float],
    raw_year_cache_dir: Path | None,
    grid_request_cache_dir: Path | None,
    progress_callback: ProgressCallback | None,
    request_index: int,
    request_count: int,
) -> tuple[Path, str, str]:
    raw_year_path = _twcr_raw_year_path(raw_year_cache_dir, year)
    if raw_year_path.exists():
        return raw_year_path, "full_year", "local_cache"

    if _should_use_twcr_subset(request_area):
        grid_request_path = _twcr_grid_request_path(grid_request_cache_dir, year, request_area)
        if grid_request_path.exists():
            return grid_request_path, "year_subset", "local_cache"

    work = _twcr_download_work_for_year(
        year,
        request_area=request_area,
        raw_year_cache_dir=raw_year_cache_dir,
        grid_request_cache_dir=grid_request_cache_dir,
    )
    if work is None:
        return raw_year_path, "full_year", "local_cache"

    if work.task_kind == "twcr_grid":
        request_path = _ensure_twcr_grid_year_cached(
            year,
            area=request_area,
            grid_request_cache_dir=grid_request_cache_dir,
            progress_callback=progress_callback,
            request_index=request_index,
            request_count=request_count,
        )
        return request_path, "year_subset", "download"

    request_path = _ensure_twcr_year_cached(
        year,
        raw_year_cache_dir=raw_year_cache_dir,
        progress_callback=progress_callback,
        request_index=request_index,
        request_count=request_count,
    )
    return request_path, "full_year", "download"


def _twcr_download_work_for_year(
    year: int,
    *,
    request_area: tuple[float, float, float, float],
    raw_year_cache_dir: Path | None,
    grid_request_cache_dir: Path | None,
) -> TWCRDownloadWork | None:
    raw_year_path = _twcr_raw_year_path(raw_year_cache_dir, year)
    if raw_year_path.exists():
        return None

    if _should_use_twcr_subset(request_area):
        grid_request_path = _twcr_grid_request_path(grid_request_cache_dir, year, request_area)
        if grid_request_path.exists():
            return None
        return TWCRDownloadWork(
            work_id=str(grid_request_path),
            task_kind="twcr_grid",
            year=year,
            area=request_area,
        )

    return TWCRDownloadWork(
        work_id=str(raw_year_path),
        task_kind="twcr_year",
        year=year,
    )


def _should_use_twcr_subset(request_area: tuple[float, float, float, float]) -> bool:
    return _twcr_grid_cell_count(request_area) <= TWCR_SUBSET_MAX_GRID_CELLS


def _twcr_grid_cell_count(area: tuple[float, float, float, float]) -> int:
    north, west, south, east = area
    latitude_cells = max(1, int(round((north - south) / TWCR_GRID_STEP_DEGREES)) + 1)
    longitude_cells = max(1, int(round((east - west) / TWCR_GRID_STEP_DEGREES)) + 1)
    return latitude_cells * longitude_cells
