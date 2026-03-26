from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import requests

from mystripes.models import GeocodingResult

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
GEOAPIFY_SEARCH_URL = "https://api.geoapify.com/v1/geocode/search"
DEFAULT_USER_AGENT = "mystripes/0.1"
GEOCODING_CACHE_DIR = Path(".mystripes-cache/geocoding")


def search_places(
    query: str,
    limit: int = 5,
    *,
    geoapify_api_key: str | None = None,
    cache_dir: Path | None = GEOCODING_CACHE_DIR,
) -> list[GeocodingResult]:
    query = query.strip()
    if not query:
        return []

    effective_geoapify_key = (geoapify_api_key or os.getenv("GEOAPIFY_API_KEY", "")).strip()
    provider = "geoapify" if effective_geoapify_key else "nominatim"

    cache_path = None
    if cache_dir is not None:
        cache_path = _geocoding_cache_path(cache_dir, provider=provider, query=query, limit=limit)
        cached_results = _load_cached_results(cache_path)
        if cached_results is not None:
            return cached_results

    if provider == "geoapify":
        results = _search_geoapify(query, limit=limit, api_key=effective_geoapify_key)
    else:
        results = _search_nominatim(query, limit=limit)

    if cache_path is not None:
        _store_cached_results(cache_path, results)
    return results


def _search_geoapify(query: str, limit: int, api_key: str) -> list[GeocodingResult]:
    response = requests.get(
        GEOAPIFY_SEARCH_URL,
        params={
            "text": query,
            "format": "json",
            "limit": limit,
            "lang": "en",
            "bias": "countrycode:none",
            "apiKey": api_key,
        },
        headers={"User-Agent": os.getenv("GEOCODER_USER_AGENT", DEFAULT_USER_AGENT)},
        timeout=20,
    )
    response.raise_for_status()

    payload = response.json()
    raw_results = payload.get("results", [])
    return [_result_from_payload(item, provider="geoapify") for item in raw_results if isinstance(item, dict)]


def _search_nominatim(query: str, limit: int) -> list[GeocodingResult]:
    response = requests.get(
        NOMINATIM_SEARCH_URL,
        params={
            "q": query,
            "format": "jsonv2",
            "limit": limit,
            "addressdetails": 1,
            "polygon_geojson": 1,
            "polygon_threshold": 0.01,
        },
        headers={"User-Agent": os.getenv("GEOCODER_USER_AGENT", DEFAULT_USER_AGENT)},
        timeout=20,
    )
    response.raise_for_status()

    return [_result_from_payload(item, provider="nominatim") for item in response.json() if isinstance(item, dict)]


def _geocoding_cache_path(cache_dir: Path, provider: str, query: str, limit: int) -> Path:
    payload = {
        "provider": provider,
        "query": query,
        "limit": limit,
        "cache_version": 1,
    }
    key = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return cache_dir / f"{key}.json"


def _load_cached_results(cache_path: Path) -> list[GeocodingResult] | None:
    if not cache_path.exists():
        return None

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        cache_path.unlink(missing_ok=True)
        return None

    if not isinstance(payload, list):
        cache_path.unlink(missing_ok=True)
        return None

    try:
        return [_geocoding_result_from_cache(item) for item in payload]
    except Exception:
        cache_path.unlink(missing_ok=True)
        return None


def _store_cached_results(cache_path: Path, results: list[GeocodingResult]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "display_name": result.display_name,
            "latitude": result.latitude,
            "longitude": result.longitude,
            "coordinate_source": result.coordinate_source,
            "geojson": result.geojson,
            "bounding_box": list(result.bounding_box) if result.bounding_box is not None else None,
        }
        for result in results
    ]

    temporary_path = cache_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    temporary_path.replace(cache_path)


def _geocoding_result_from_cache(payload: Any) -> GeocodingResult:
    if not isinstance(payload, dict):
        raise ValueError("Cached geocoding results must be objects.")

    bounding_box = payload.get("bounding_box")
    if isinstance(bounding_box, list):
        bounding_box = tuple(float(value) for value in bounding_box)

    return GeocodingResult(
        display_name=str(payload["display_name"]),
        latitude=float(payload["latitude"]),
        longitude=float(payload["longitude"]),
        coordinate_source=str(payload["coordinate_source"]),
        geojson=payload.get("geojson") if isinstance(payload.get("geojson"), dict) else None,
        bounding_box=bounding_box if isinstance(bounding_box, tuple) and len(bounding_box) == 4 else None,
    )


def _result_from_payload(payload: dict[str, Any], provider: str = "nominatim") -> GeocodingResult:
    if provider == "geoapify":
        return _geoapify_result_from_payload(payload)
    return _nominatim_result_from_payload(payload)


def _geoapify_result_from_payload(payload: dict[str, Any]) -> GeocodingResult:
    display_name = (
        str(payload.get("formatted", "")).strip()
        or str(payload.get("address_line1", "")).strip()
        or str(payload.get("name", "")).strip()
        or "Unnamed place"
    )
    latitude, longitude, coordinate_source = _extract_geoapify_coordinates(payload)
    bounding_box = _parse_geoapify_bounding_box(payload.get("bbox"))
    return GeocodingResult(
        display_name=display_name,
        latitude=latitude,
        longitude=longitude,
        coordinate_source=coordinate_source,
        geojson=None,
        bounding_box=bounding_box,
    )


def _nominatim_result_from_payload(payload: dict[str, Any]) -> GeocodingResult:
    latitude, longitude, coordinate_source = _extract_nominatim_coordinates(payload)
    return GeocodingResult(
        display_name=str(payload["display_name"]),
        latitude=latitude,
        longitude=longitude,
        coordinate_source=coordinate_source,
        geojson=payload.get("geojson") if isinstance(payload.get("geojson"), dict) else None,
        bounding_box=_parse_nominatim_bounding_box(payload.get("boundingbox")),
    )


def _extract_geoapify_coordinates(payload: dict[str, Any]) -> tuple[float, float, str]:
    latitude = payload.get("lat")
    longitude = payload.get("lon")
    if latitude is not None and longitude is not None:
        return float(latitude), float(longitude), "result center"

    bounding_box = _parse_geoapify_bounding_box(payload.get("bbox"))
    if bounding_box is not None:
        south, north, west, east = bounding_box
        return (south + north) / 2.0, (west + east) / 2.0, "bounding box center"

    raise ValueError("The Geoapify response did not include usable coordinates.")


def _extract_nominatim_coordinates(payload: dict[str, Any]) -> tuple[float, float, str]:
    geojson = payload.get("geojson")
    if isinstance(geojson, dict):
        coordinates = _coordinates_from_geojson(geojson)
        if coordinates is not None:
            return coordinates[0], coordinates[1], coordinates[2]

    if "lat" in payload and "lon" in payload:
        return float(payload["lat"]), float(payload["lon"]), "result center"

    bounding_box = _parse_nominatim_bounding_box(payload.get("boundingbox"))
    if bounding_box is not None:
        south, north, west, east = bounding_box
        return (south + north) / 2.0, (west + east) / 2.0, "bounding box center"

    raise ValueError("The geocoder response did not include usable coordinates.")


def _parse_nominatim_bounding_box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    south, north, west, east = [float(item) for item in value]
    return south, north, west, east


def _parse_geoapify_bounding_box(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        try:
            west = float(value["lon1"])
            south = float(value["lat1"])
            east = float(value["lon2"])
            north = float(value["lat2"])
        except (KeyError, TypeError, ValueError):
            return None
        return south, north, west, east

    if isinstance(value, list) and len(value) == 4:
        west, south, east, north = [float(item) for item in value]
        return south, north, west, east

    return None


def _coordinates_from_geojson(geojson: dict[str, Any]) -> tuple[float, float, str] | None:
    geometry_type = str(geojson.get("type", "")).lower()
    coordinates = geojson.get("coordinates")

    if geometry_type == "point" and _is_coordinate_pair(coordinates):
        longitude, latitude = coordinates
        return float(latitude), float(longitude), "point geometry"

    if geometry_type == "polygon" and isinstance(coordinates, list):
        centroid = _polygon_centroid(coordinates)
        if centroid is not None:
            return centroid[0], centroid[1], "area centroid"

    if geometry_type == "multipolygon" and isinstance(coordinates, list):
        centroid = _multipolygon_centroid(coordinates)
        if centroid is not None:
            return centroid[0], centroid[1], "area centroid"

    if geometry_type == "linestring" and isinstance(coordinates, list):
        centroid = _mean_coordinate(coordinates)
        if centroid is not None:
            return centroid[0], centroid[1], "line midpoint"

    if geometry_type == "multilinestring" and isinstance(coordinates, list):
        points = [point for line in coordinates for point in line]
        centroid = _mean_coordinate(points)
        if centroid is not None:
            return centroid[0], centroid[1], "line midpoint"

    if geometry_type == "multipoint" and isinstance(coordinates, list):
        centroid = _mean_coordinate(coordinates)
        if centroid is not None:
            return centroid[0], centroid[1], "point cluster center"

    if geometry_type == "geometrycollection":
        for geometry in geojson.get("geometries", []):
            if isinstance(geometry, dict):
                centroid = _coordinates_from_geojson(geometry)
                if centroid is not None:
                    return centroid

    return None


def _polygon_centroid(rings: list[Any]) -> tuple[float, float] | None:
    if not rings:
        return None

    centroid_longitude, centroid_latitude, area = _ring_centroid(rings[0])
    if area == 0:
        return _mean_coordinate(rings[0])

    return centroid_latitude, centroid_longitude


def _multipolygon_centroid(polygons: list[Any]) -> tuple[float, float] | None:
    weighted_longitude = 0.0
    weighted_latitude = 0.0
    total_area = 0.0

    for polygon in polygons:
        if not isinstance(polygon, list) or not polygon:
            continue
        centroid_longitude, centroid_latitude, area = _ring_centroid(polygon[0])
        if area == 0:
            continue
        weighted_longitude += centroid_longitude * area
        weighted_latitude += centroid_latitude * area
        total_area += area

    if total_area > 0:
        return weighted_latitude / total_area, weighted_longitude / total_area

    points = [point for polygon in polygons if isinstance(polygon, list) for ring in polygon for point in ring]
    return _mean_coordinate(points)


def _ring_centroid(ring: list[Any]) -> tuple[float, float, float]:
    points = _normalize_ring_points(ring)
    if len(points) < 3:
        fallback = _mean_coordinate(points)
        if fallback is None:
            return 0.0, 0.0, 0.0
        latitude, longitude = fallback
        return longitude, latitude, 0.0

    area_accumulator = 0.0
    centroid_longitude = 0.0
    centroid_latitude = 0.0

    for index in range(len(points)):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % len(points)]
        cross = x1 * y2 - x2 * y1
        area_accumulator += cross
        centroid_longitude += (x1 + x2) * cross
        centroid_latitude += (y1 + y2) * cross

    signed_area = area_accumulator / 2.0
    if signed_area == 0:
        fallback = _mean_coordinate(points)
        if fallback is None:
            return 0.0, 0.0, 0.0
        latitude, longitude = fallback
        return longitude, latitude, 0.0

    centroid_longitude /= 6.0 * signed_area
    centroid_latitude /= 6.0 * signed_area
    return centroid_longitude, centroid_latitude, abs(signed_area)


def _normalize_ring_points(ring: list[Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for point in ring:
        if _is_coordinate_pair(point):
            longitude, latitude = point
            points.append((float(longitude), float(latitude)))
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    return points


def _mean_coordinate(points: list[Any]) -> tuple[float, float] | None:
    valid_points = _normalize_ring_points(points)
    if not valid_points:
        return None

    mean_longitude = sum(point[0] for point in valid_points) / len(valid_points)
    mean_latitude = sum(point[1] for point in valid_points) / len(valid_points)
    return mean_latitude, mean_longitude


def _is_coordinate_pair(value: Any) -> bool:
    return isinstance(value, list | tuple) and len(value) >= 2
