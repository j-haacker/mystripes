from __future__ import annotations

import calendar
import hashlib
import json
import math
import os
import tempfile
import tomllib
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from mystrips.models import CDSConfig, DatasetWindow

DATASET_NAME = "reanalysis-era5-land-monthly-means"
DATASET_URL = "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means"
DATASET_LICENCE_URL = f"{DATASET_URL}?tab=download#manage-licences"
CONSTRAINTS_URL = (
    "https://cds.climate.copernicus.eu/api/catalogue/v1/collections/"
    f"{DATASET_NAME}/constraints.json"
)
DEFAULT_CDSAPI_URL = "https://cds.climate.copernicus.eu/api"
LOCAL_CDS_CREDENTIALS_PATH = Path(".streamlit/local_cds_credentials.toml")
TEMPERATURE_CACHE_DIR = Path(".mystrips-cache")
GRID_STEP_DEGREES = 0.1


class CDSCredentialsMissingError(RuntimeError):
    """Raised when the app operator has not configured CDS credentials."""


class CDSRequestError(RuntimeError):
    """Raised when the CDS request cannot be completed."""


def get_dataset_window() -> DatasetWindow:
    response = requests.get(CONSTRAINTS_URL, timeout=30)
    response.raise_for_status()
    return _dataset_window_from_constraints(response.json())


def resolve_cds_config(
    secret_values: Mapping[str, str] | None = None,
    local_credentials_path: Path = LOCAL_CDS_CREDENTIALS_PATH,
) -> CDSConfig:
    secret_values = secret_values or {}

    secret_key = str(secret_values.get("CDSAPI_KEY", "")).strip()
    if secret_key:
        secret_url = str(secret_values.get("CDSAPI_URL", "")).strip() or DEFAULT_CDSAPI_URL
        return CDSConfig(url=secret_url, key=secret_key, source="streamlit_secrets")

    env_key = os.getenv("CDSAPI_KEY", "").strip()
    if env_key:
        env_url = os.getenv("CDSAPI_URL", "").strip() or DEFAULT_CDSAPI_URL
        return CDSConfig(url=env_url, key=env_key, source="environment")

    local_config = load_local_cds_config(local_credentials_path)
    if local_config is not None:
        return local_config

    raise CDSCredentialsMissingError(
        "Missing CDSAPI_KEY. Add it to Streamlit secrets, environment variables, or save "
        "it locally from the app sidebar."
    )


def load_local_cds_config(path: Path = LOCAL_CDS_CREDENTIALS_PATH) -> CDSConfig | None:
    if not path.exists():
        return None

    with path.open("rb") as handle:
        payload = tomllib.load(handle)

    key = str(payload.get("CDSAPI_KEY", "")).strip()
    if not key:
        return None

    url = str(payload.get("CDSAPI_URL", "")).strip() or DEFAULT_CDSAPI_URL
    return CDSConfig(url=url, key=key, source="local_file")


def save_local_cds_config(
    key: str,
    url: str = DEFAULT_CDSAPI_URL,
    path: Path = LOCAL_CDS_CREDENTIALS_PATH,
) -> None:
    normalized_key = key.strip()
    normalized_url = url.strip() or DEFAULT_CDSAPI_URL
    if not normalized_key:
        raise CDSCredentialsMissingError("A CDS API token is required before it can be saved.")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        f'CDSAPI_URL = "{_escape_toml_string(normalized_url)}"\n'
        f'CDSAPI_KEY = "{_escape_toml_string(normalized_key)}"\n'
    )
    path.write_text(payload, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except PermissionError:
        pass


def clear_local_cds_config(path: Path = LOCAL_CDS_CREDENTIALS_PATH) -> None:
    if path.exists():
        path.unlink()


def fetch_point_temperature_series(
    config: CDSConfig,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    spatial_mode: str = "single_cell",
    radius_km: float | None = None,
    boundary_geojson: Mapping[str, Any] | None = None,
    boundary_bbox: tuple[float, float, float, float] | None = None,
    cache_dir: Path | None = TEMPERATURE_CACHE_DIR,
) -> pd.DataFrame:
    if start_date > end_date:
        raise ValueError("Start date must be on or before end date.")
    if spatial_mode == "radius" and radius_km is not None and radius_km <= 0:
        raise ValueError("Radius must be greater than zero.")

    cache_path = None
    if cache_dir is not None:
        cache_path = _temperature_series_cache_path(
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
            return cached_frame

    try:
        import cdsapi
    except ModuleNotFoundError as exc:
        raise CDSRequestError(
            "cdsapi is not installed. Install the dependencies from requirements.txt."
        ) from exc

    request_area = _request_area(
        latitude=latitude,
        longitude=longitude,
        spatial_mode=spatial_mode,
        radius_km=radius_km,
        boundary_geojson=boundary_geojson,
        boundary_bbox=boundary_bbox,
    )
    requests_to_run = _build_monthly_requests(
        start_date=start_date,
        end_date=end_date,
        area=request_area,
    )
    client = cdsapi.Client(url=config.url, key=config.key, quiet=True, progress=False)
    frames: list[pd.DataFrame] = []
    last_error: Exception | None = None

    for request in requests_to_run:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "era5_land_monthly_request.nc"
            try:
                client.retrieve(DATASET_NAME, request, str(target))
            except Exception as exc:  # pragma: no cover - depends on external service.
                last_error = exc
                break

            grid_frame = parse_temperature_file(
                target,
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

    if not frames:
        if last_error is not None:
            raise CDSRequestError(_explain_cds_error(last_error))
        raise CDSRequestError("ERA5-Land CDS request returned no monthly data.")

    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp")
    combined = combined.reset_index(drop=True)
    if cache_path is not None:
        _store_cached_temperature_series(cache_path, combined)
    return combined


def parse_temperature_file(
    path: Path,
    target_latitude: float,
    target_longitude: float,
) -> pd.DataFrame:
    header = path.read_bytes()[:8]
    if header.startswith(b"\x89HDF") or header.startswith(b"CDF"):
        return parse_temperature_netcdf_grid(
            path=path,
            target_latitude=target_latitude,
            target_longitude=target_longitude,
        )

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise CDSRequestError(
            "CDS returned a binary file that the app could not parse as NetCDF. "
            "This usually means the selected data format does not match the parser."
        ) from exc

    trimmed = text.strip()
    if not trimmed:
        raise CDSRequestError("The CDS response was empty.")

    preview = trimmed.splitlines()[0][:200]
    raise CDSRequestError(f"CDS returned unexpected text instead of NetCDF: {preview}")


def parse_temperature_netcdf_grid(
    path: Path,
    target_latitude: float,
    target_longitude: float,
) -> pd.DataFrame:
    try:
        import netCDF4
    except ModuleNotFoundError as exc:
        raise CDSRequestError(
            "netCDF4 is not installed. Install the dependencies from requirements.txt."
        ) from exc

    with netCDF4.Dataset(path) as dataset:
        time_variable_name = _find_first_name(dataset.variables, ("valid_time", "time"))
        if time_variable_name is None:
            raise CDSRequestError("Could not identify the time variable in the CDS NetCDF file.")

        temperature_variable_name = _resolve_temperature_variable_name(dataset)
        temperature_variable = dataset.variables[temperature_variable_name]
        time_variable = dataset.variables[time_variable_name]

        latitude_name = _find_first_name(dataset.variables, ("latitude", "lat"))
        longitude_name = _find_first_name(dataset.variables, ("longitude", "lon"))
        if latitude_name is None or longitude_name is None:
            raise CDSRequestError("Could not identify latitude/longitude coordinates in the CDS NetCDF file.")

        latitudes = np.asarray(dataset.variables[latitude_name][:], dtype=float).reshape(-1)
        longitudes = np.asarray(dataset.variables[longitude_name][:], dtype=float).reshape(-1)
        data = _extract_temperature_cube(
            dataset=dataset,
            variable_name=temperature_variable_name,
            time_variable_name=time_variable_name,
            latitude_name=latitude_name,
            longitude_name=longitude_name,
        )
        timestamps = _extract_timestamps(netCDF4, time_variable)

        units = str(getattr(temperature_variable, "units", "") or getattr(temperature_variable, "GRIB_units", ""))
        temperature_c = data - 273.15 if units.upper().startswith("K") else data
        frame = _grid_frame_from_cube(timestamps, latitudes, longitudes, temperature_c)
        frame = frame.dropna().sort_values("timestamp").reset_index(drop=True)
        if frame.empty:
            raise CDSRequestError("The CDS NetCDF file did not contain usable temperature rows.")
        return frame


def _dataset_window_from_constraints(constraints: list[dict[str, Any]]) -> DatasetWindow:
    available_months: list[date] = []

    for entry in constraints:
        product_types = entry.get("product_type", [])
        variables = entry.get("variable", [])
        data_formats = entry.get("data_format", [])
        years = entry.get("year", [])
        months = entry.get("month", [])

        if "monthly_averaged_reanalysis" not in product_types:
            continue
        if "2m_temperature" not in variables:
            continue
        if "netcdf" not in data_formats:
            continue

        for year_text in years:
            for month_text in months:
                available_months.append(date(int(year_text), int(month_text), 1))

    if not available_months:
        raise CDSRequestError("Could not determine the ERA5-Land monthly availability window from CDS.")

    min_start = min(available_months)
    max_month = max(available_months)
    max_end = date(max_month.year, max_month.month, calendar.monthrange(max_month.year, max_month.month)[1])
    return DatasetWindow(min_start=min_start, max_end=max_end)


def _build_monthly_requests(
    start_date: date,
    end_date: date,
    area: tuple[float, float, float, float],
) -> list[dict[str, object]]:
    year_to_months: dict[str, list[str]] = {}
    current = date(start_date.year, start_date.month, 1)
    final = date(end_date.year, end_date.month, 1)

    while current <= final:
        year_key = f"{current.year:04d}"
        month_key = f"{current.month:02d}"
        year_to_months.setdefault(year_key, []).append(month_key)
        current = _next_month(current)

    requests_to_run: list[dict[str, object]] = []
    current_years: list[str] = []
    current_months: tuple[str, ...] | None = None

    for year_key in sorted(year_to_months):
        months = tuple(year_to_months[year_key])
        if current_months is None or months == current_months:
            current_years.append(year_key)
            current_months = months
            continue

        requests_to_run.append(
            _monthly_request_payload(current_years, list(current_months), area)
        )
        current_years = [year_key]
        current_months = months

    if current_years and current_months is not None:
        requests_to_run.append(
            _monthly_request_payload(current_years, list(current_months), area)
        )

    return requests_to_run


def _temperature_series_cache_path(
    cache_dir: Path,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    spatial_mode: str,
    radius_km: float | None,
    boundary_geojson: Mapping[str, Any] | None,
    boundary_bbox: tuple[float, float, float, float] | None,
) -> Path:
    payload = {
        "cache_version": 1,
        "dataset": DATASET_NAME,
        "latitude": round(latitude, 6),
        "longitude": round(longitude, 6),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "spatial_mode": spatial_mode,
        "radius_km": None if radius_km is None else round(radius_km, 6),
        "boundary_geojson": boundary_geojson,
        "boundary_bbox": list(boundary_bbox) if boundary_bbox is not None else None,
    }
    key = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return cache_dir / f"{key}.csv"


def _load_cached_temperature_series(cache_path: Path) -> pd.DataFrame | None:
    if not cache_path.exists():
        return None

    try:
        frame = pd.read_csv(cache_path)
    except Exception:
        cache_path.unlink(missing_ok=True)
        return None

    required_columns = {"timestamp", "temperature_c", "sample_days"}
    if not required_columns.issubset(frame.columns):
        cache_path.unlink(missing_ok=True)
        return None

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def _store_cached_temperature_series(cache_path: Path, frame: pd.DataFrame) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.copy()
    payload["timestamp"] = pd.to_datetime(payload["timestamp"], utc=True)

    temporary_path = cache_path.with_suffix(".tmp")
    payload.to_csv(temporary_path, index=False)
    temporary_path.replace(cache_path)


def _monthly_request_payload(
    years: list[str],
    months: list[str],
    area: tuple[float, float, float, float],
) -> dict[str, object]:
    return {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["2m_temperature"],
        "year": years,
        "month": months,
        "time": ["00:00"],
        "area": list(area),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def _snap_to_grid(value: float) -> float:
    return round(round(value / GRID_STEP_DEGREES) * GRID_STEP_DEGREES, 1)


def _next_month(current: date) -> date:
    if current.month == 12:
        return date(current.year + 1, 1, 1)
    return date(current.year, current.month + 1, 1)


def _resolve_temperature_variable_name(dataset) -> str:
    candidates = ("2m_temperature", "t2m", "temperature", "value")
    for name in candidates:
        if name in dataset.variables:
            return name

    excluded = {"latitude", "longitude", "lat", "lon", "time", "valid_time", "number", "expver"}
    for name, variable in dataset.variables.items():
        if name.lower() in excluded:
            continue
        units = str(getattr(variable, "units", "") or getattr(variable, "GRIB_units", ""))
        if units.upper().startswith("K") and getattr(variable, "dimensions", ()):
            return name

    raise CDSRequestError("Could not identify the temperature variable in the CDS NetCDF file.")


def _find_first_name(mapping: Mapping[str, Any], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in mapping:
            return candidate
    return None


def _extract_temperature_cube(
    dataset,
    variable_name: str,
    time_variable_name: str,
    latitude_name: str,
    longitude_name: str,
) -> np.ndarray:
    variable = dataset.variables[variable_name]
    indexer: list[object] = []
    kept_dimensions: list[str] = []

    for dimension_name in variable.dimensions:
        lowered = dimension_name.lower()
        if dimension_name == time_variable_name or lowered == "time" or lowered == "valid_time":
            indexer.append(slice(None))
            kept_dimensions.append("time")
            continue
        if dimension_name == latitude_name or lowered in {"latitude", "lat"}:
            indexer.append(slice(None))
            kept_dimensions.append("latitude")
            continue
        if dimension_name == longitude_name or lowered in {"longitude", "lon"}:
            indexer.append(slice(None))
            kept_dimensions.append("longitude")
            continue

        dimension_size = len(dataset.dimensions[dimension_name])
        if dimension_size == 1:
            indexer.append(0)
            continue

        raise CDSRequestError(
            f"Unexpected multi-valued dimension '{dimension_name}' in the CDS NetCDF file."
        )

    extracted = variable[tuple(indexer)]
    if np.ma.isMaskedArray(extracted):
        extracted = extracted.filled(np.nan)

    values = np.asarray(extracted, dtype=float)
    if values.ndim == 1 and kept_dimensions == ["time"]:
        return values[:, np.newaxis, np.newaxis]

    transpose_order = [kept_dimensions.index(name) for name in ("time", "latitude", "longitude")]
    return np.transpose(values, axes=transpose_order)


def _extract_timestamps(netCDF4_module, time_variable) -> pd.Series:
    units = getattr(time_variable, "units", None)
    if not units:
        raise CDSRequestError("The CDS NetCDF time variable is missing units.")

    calendar_name = getattr(time_variable, "calendar", "standard")
    raw_values = netCDF4_module.num2date(time_variable[:], units=units, calendar=calendar_name)
    timestamps = pd.to_datetime(
        [
            pd.Timestamp(
                value.year,
                value.month,
                value.day,
                getattr(value, "hour", 0),
                getattr(value, "minute", 0),
                getattr(value, "second", 0),
                tz="UTC",
            )
            for value in raw_values
        ],
        utc=True,
    )
    return pd.Series(timestamps)


def _nearest_index(values: np.ndarray, target_value: float) -> int:
    return int(np.abs(values - target_value).argmin())


def _request_area(
    latitude: float,
    longitude: float,
    spatial_mode: str,
    radius_km: float | None,
    boundary_geojson: Mapping[str, Any] | None,
    boundary_bbox: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float]:
    if spatial_mode == "single_cell":
        snapped_latitude = _snap_to_grid(latitude)
        snapped_longitude = _snap_to_grid(longitude)
        return snapped_latitude, snapped_longitude, snapped_latitude, snapped_longitude

    if spatial_mode == "radius":
        effective_radius = radius_km or 25.0
        return _radius_request_area(latitude, longitude, effective_radius)

    if spatial_mode == "boundary":
        bbox = boundary_bbox or _geometry_bounding_box(boundary_geojson)
        if bbox is None:
            raise CDSRequestError(
                "Boundary aggregation requires a place search result with a municipality or district boundary."
            )
        south, north, west, east = bbox
        return _snap_area(north, west, south, east)

    raise CDSRequestError(f"Unsupported spatial aggregation mode: {spatial_mode}")


def _radius_request_area(latitude: float, longitude: float, radius_km: float) -> tuple[float, float, float, float]:
    delta_latitude = radius_km / 111.32
    cos_latitude = max(abs(math.cos(math.radians(latitude))), 0.01)
    delta_longitude = radius_km / (111.32 * cos_latitude)
    return _snap_area(
        latitude + delta_latitude,
        longitude - delta_longitude,
        latitude - delta_latitude,
        longitude + delta_longitude,
    )


def _snap_area(north: float, west: float, south: float, east: float) -> tuple[float, float, float, float]:
    north = min(89.0, _snap_up_to_grid(north))
    south = max(-89.0, _snap_down_to_grid(south))
    west = max(-180.0, _snap_down_to_grid(west))
    east = min(180.0, _snap_up_to_grid(east))
    return north, west, south, east


def _snap_down_to_grid(value: float) -> float:
    return round(math.floor(value / GRID_STEP_DEGREES) * GRID_STEP_DEGREES, 1)


def _snap_up_to_grid(value: float) -> float:
    return round(math.ceil(value / GRID_STEP_DEGREES) * GRID_STEP_DEGREES, 1)


def _grid_frame_from_cube(
    timestamps: pd.Series,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    temperature_c: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for time_index, timestamp in enumerate(timestamps):
        sample_days = calendar.monthrange(timestamp.year, timestamp.month)[1]
        for latitude_index, grid_latitude in enumerate(latitudes):
            for longitude_index, grid_longitude in enumerate(longitudes):
                records.append(
                    {
                        "timestamp": timestamp,
                        "temperature_c": float(temperature_c[time_index, latitude_index, longitude_index]),
                        "sample_days": sample_days,
                        "grid_latitude": float(grid_latitude),
                        "grid_longitude": float(grid_longitude),
                    }
                )
    return pd.DataFrame.from_records(records)


def _aggregate_spatial_selection(
    grid_frame: pd.DataFrame,
    latitude: float,
    longitude: float,
    spatial_mode: str,
    radius_km: float | None,
    boundary_geojson: Mapping[str, Any] | None,
    boundary_bbox: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    cells = grid_frame[["grid_latitude", "grid_longitude"]].drop_duplicates().copy()

    if spatial_mode == "single_cell":
        cells["distance_km"] = cells.apply(
            lambda row: _haversine_km(latitude, longitude, row["grid_latitude"], row["grid_longitude"]),
            axis=1,
        )
        selected_cells = cells.loc[cells["distance_km"] == cells["distance_km"].min(), ["grid_latitude", "grid_longitude"]]
    elif spatial_mode == "radius":
        effective_radius = radius_km or 25.0
        cells["distance_km"] = cells.apply(
            lambda row: _haversine_km(latitude, longitude, row["grid_latitude"], row["grid_longitude"]),
            axis=1,
        )
        selected_cells = cells.loc[cells["distance_km"] <= effective_radius, ["grid_latitude", "grid_longitude"]]
        if selected_cells.empty:
            selected_cells = cells.loc[cells["distance_km"] == cells["distance_km"].min(), ["grid_latitude", "grid_longitude"]]
    elif spatial_mode == "boundary":
        selected_cells = _select_boundary_cells(cells, boundary_geojson, boundary_bbox)
        if selected_cells.empty:
            raise CDSRequestError(
                "No ERA5-Land grid-cell centers fell inside the selected municipality or district area."
            )
    else:
        raise CDSRequestError(f"Unsupported spatial aggregation mode: {spatial_mode}")

    selected = grid_frame.merge(selected_cells, on=["grid_latitude", "grid_longitude"], how="inner")
    aggregated = (
        selected.groupby("timestamp", as_index=False)
        .agg(
            temperature_c=("temperature_c", "mean"),
            sample_days=("sample_days", "first"),
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return aggregated


def _select_boundary_cells(
    cells: pd.DataFrame,
    boundary_geojson: Mapping[str, Any] | None,
    boundary_bbox: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    if boundary_geojson and str(boundary_geojson.get("type", "")).lower() in {"polygon", "multipolygon"}:
        mask = cells.apply(
            lambda row: _point_in_geometry(
                float(row["grid_longitude"]),
                float(row["grid_latitude"]),
                boundary_geojson,
            ),
            axis=1,
        )
        return cells.loc[mask, ["grid_latitude", "grid_longitude"]]

    if boundary_bbox is None:
        return cells.iloc[0:0][["grid_latitude", "grid_longitude"]]

    south, north, west, east = boundary_bbox
    mask = (
        (cells["grid_latitude"] >= south)
        & (cells["grid_latitude"] <= north)
        & (cells["grid_longitude"] >= west)
        & (cells["grid_longitude"] <= east)
    )
    return cells.loc[mask, ["grid_latitude", "grid_longitude"]]


def _geometry_bounding_box(geometry: Mapping[str, Any] | None) -> tuple[float, float, float, float] | None:
    if not geometry:
        return None

    coordinates = _flatten_geometry_points(geometry)
    if not coordinates:
        return None

    longitudes = [point[0] for point in coordinates]
    latitudes = [point[1] for point in coordinates]
    return min(latitudes), max(latitudes), min(longitudes), max(longitudes)


def _flatten_geometry_points(geometry: Mapping[str, Any]) -> list[tuple[float, float]]:
    geometry_type = str(geometry.get("type", "")).lower()
    coordinates = geometry.get("coordinates")

    if geometry_type == "point" and isinstance(coordinates, list) and len(coordinates) >= 2:
        return [(float(coordinates[0]), float(coordinates[1]))]

    if geometry_type in {"polygon", "multipolygon", "linestring", "multilinestring", "multipoint"}:
        return _flatten_coordinate_structure(coordinates)

    if geometry_type == "geometrycollection":
        points: list[tuple[float, float]] = []
        for item in geometry.get("geometries", []):
            if isinstance(item, Mapping):
                points.extend(_flatten_geometry_points(item))
        return points

    return []


def _flatten_coordinate_structure(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, list):
        if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
            return [(float(value[0]), float(value[1]))]
        points: list[tuple[float, float]] = []
        for item in value:
            points.extend(_flatten_coordinate_structure(item))
        return points
    return []


def _point_in_geometry(longitude: float, latitude: float, geometry: Mapping[str, Any]) -> bool:
    geometry_type = str(geometry.get("type", "")).lower()
    coordinates = geometry.get("coordinates")

    if geometry_type == "polygon" and isinstance(coordinates, list):
        return _point_in_polygon(longitude, latitude, coordinates)
    if geometry_type == "multipolygon" and isinstance(coordinates, list):
        return any(_point_in_polygon(longitude, latitude, polygon) for polygon in coordinates if isinstance(polygon, list))
    return False


def _point_in_polygon(longitude: float, latitude: float, polygon: list[Any]) -> bool:
    if not polygon:
        return False

    outer_ring = polygon[0]
    if not _point_in_ring(longitude, latitude, outer_ring):
        return False

    for hole in polygon[1:]:
        if _point_in_ring(longitude, latitude, hole):
            return False

    return True


def _point_in_ring(longitude: float, latitude: float, ring: list[Any]) -> bool:
    points = _flatten_coordinate_structure(ring)
    if len(points) < 3:
        return False

    inside = False
    previous_longitude, previous_latitude = points[-1]
    for current_longitude, current_latitude in points:
        intersects = ((current_latitude > latitude) != (previous_latitude > latitude)) and (
            longitude
            < (previous_longitude - current_longitude) * (latitude - current_latitude)
            / ((previous_latitude - current_latitude) or 1e-12)
            + current_longitude
        )
        if intersects:
            inside = not inside
        previous_longitude, previous_latitude = current_longitude, current_latitude
    return inside


def _haversine_km(latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float) -> float:
    earth_radius_km = 6371.0088
    latitude_a_rad = math.radians(latitude_a)
    latitude_b_rad = math.radians(latitude_b)
    delta_latitude = math.radians(latitude_b - latitude_a)
    delta_longitude = math.radians(longitude_b - longitude_a)
    haversine = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(latitude_a_rad) * math.cos(latitude_b_rad) * math.sin(delta_longitude / 2) ** 2
    )
    return 2 * earth_radius_km * math.asin(math.sqrt(haversine))


def _escape_toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _explain_cds_error(error: Exception) -> str:
    text = str(error)
    lowered = text.lower()

    if "required licences not accepted" in lowered:
        return (
            "ERA5-Land access was denied because the required CDS licence has not been "
            f"accepted for this account. Visit {DATASET_LICENCE_URL} while logged into the "
            "same CDS account, accept the licence, and try again."
        )

    if "401" in lowered or "unauthorized" in lowered:
        return (
            "CDS rejected the credentials. Current CDS authentication uses a personal access "
            "token as the bare `CDSAPI_KEY` value, not `user:token`. If you are trying "
            "older credentials, use a session-only override and provide either a private "
            "token or the legacy `user_id` plus API key."
        )

    return f"ERA5-Land CDS request failed. Last error: {text}"
