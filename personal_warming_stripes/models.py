from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class DatasetWindow:
    min_start: date
    max_end: date


@dataclass(frozen=True)
class CDSConfig:
    url: str
    key: str


@dataclass(frozen=True)
class GeocodingResult:
    display_name: str
    latitude: float
    longitude: float
    coordinate_source: str


@dataclass(frozen=True)
class LifePeriod:
    label: str
    place_query: str
    resolved_name: str
    start_date: date
    end_date: date
    latitude: float
    longitude: float

    @property
    def display_name(self) -> str:
        return self.resolved_name or self.place_query or self.label

    @property
    def days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    @property
    def location_key(self) -> str:
        return f"{self.latitude:.4f},{self.longitude:.4f}"
