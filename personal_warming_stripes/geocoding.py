from __future__ import annotations

import os
from typing import Any

import requests

from personal_warming_stripes.models import GeocodingResult

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
DEFAULT_USER_AGENT = "personal-warming-stripes/0.1"


def search_places(query: str, limit: int = 5) -> list[GeocodingResult]:
    query = query.strip()
    if not query:
        return []

    response = requests.get(
        NOMINATIM_SEARCH_URL,
        params={
            "q": query,
            "format": "jsonv2",
            "limit": limit,
            "addressdetails": 1,
        },
        headers={"User-Agent": os.getenv("GEOCODER_USER_AGENT", DEFAULT_USER_AGENT)},
        timeout=20,
    )
    response.raise_for_status()

    results: list[GeocodingResult] = []
    for item in response.json():
        results.append(_result_from_payload(item))
    return results


def _result_from_payload(payload: dict[str, Any]) -> GeocodingResult:
    return GeocodingResult(
        display_name=str(payload["display_name"]),
        latitude=float(payload["lat"]),
        longitude=float(payload["lon"]),
    )
