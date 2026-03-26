from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class DatasetWindow:
    min_start: date
    max_end: date


@dataclass(frozen=True)
class CDSConfig:
    url: str
    key: str
    source: str = ""


@dataclass(frozen=True)
class GeocodingResult:
    display_name: str
    latitude: float
    longitude: float
    coordinate_source: str
    geojson: Mapping[str, Any] | None = None
    bounding_box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class LifePeriod:
    label: str
    place_query: str
    resolved_name: str
    start_date: date
    end_date: date
    latitude: float
    longitude: float
    boundary_geojson: Mapping[str, Any] | None = None
    bounding_box: tuple[float, float, float, float] | None = None

    @property
    def display_name(self) -> str:
        return self.resolved_name or self.place_query or self.label

    @property
    def days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    @property
    def location_key(self) -> str:
        parts = [f"{self.latitude:.4f},{self.longitude:.4f}"]
        if self.bounding_box is not None:
            parts.append(",".join(f"{value:.4f}" for value in self.bounding_box))
        return "|".join(parts)
