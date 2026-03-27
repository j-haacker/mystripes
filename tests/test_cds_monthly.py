from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from mystripes.cds import (
    CDSRequestError,
    _aggregate_spatial_selection,
    _dataset_window_from_constraints,
    _request_area,
    fetch_point_temperature_series,
    fetch_saved_temperature_series,
    parse_temperature_file,
)
from mystripes.models import CDSConfig


class MonthlyCDSTests(unittest.TestCase):
    def test_dataset_window_uses_monthly_temperature_constraints(self) -> None:
        window = _dataset_window_from_constraints(
            [
                {
                    "product_type": ["monthly_averaged_reanalysis"],
                    "variable": ["2m_temperature"],
                    "data_format": ["netcdf"],
                    "year": ["2024", "2025"],
                    "month": ["01", "02"],
                },
                {
                    "product_type": ["monthly_averaged_reanalysis"],
                    "variable": ["skin_temperature"],
                    "data_format": ["netcdf"],
                    "year": ["1900"],
                    "month": ["01"],
                },
            ]
        )

        self.assertEqual(window.min_start, date(2024, 1, 1))
        self.assertEqual(window.max_end, date(2025, 2, 28))

    def test_request_area_snaps_single_cell_to_native_grid(self) -> None:
        area = _request_area(
            latitude=48.2082,
            longitude=16.3738,
            spatial_mode="single_cell",
            radius_km=None,
            boundary_geojson=None,
            boundary_bbox=None,
        )

        self.assertEqual(area, (48.2, 16.4, 48.2, 16.4))

    def test_radius_aggregation_averages_cells_inside_radius(self) -> None:
        timestamp = pd.Timestamp("2020-01-01T00:00:00Z")
        grid_frame = pd.DataFrame(
            {
                "timestamp": [timestamp, timestamp, timestamp],
                "temperature_c": [10.0, 14.0, 50.0],
                "sample_days": [31, 31, 31],
                "grid_latitude": [0.0, 0.0, 1.0],
                "grid_longitude": [0.0, 0.1, 1.0],
            }
        )

        aggregated = _aggregate_spatial_selection(
            grid_frame=grid_frame,
            latitude=0.0,
            longitude=0.0,
            spatial_mode="radius",
            radius_km=15.0,
            boundary_geojson=None,
            boundary_bbox=None,
        )

        self.assertEqual(aggregated["temperature_c"].tolist(), [12.0])
        self.assertEqual(aggregated["sample_days"].tolist(), [31])

    def test_boundary_aggregation_uses_polygon_cells(self) -> None:
        timestamp = pd.Timestamp("2020-01-01T00:00:00Z")
        grid_frame = pd.DataFrame(
            {
                "timestamp": [timestamp, timestamp, timestamp],
                "temperature_c": [10.0, 20.0, 50.0],
                "sample_days": [31, 31, 31],
                "grid_latitude": [0.0, 0.0, 0.2],
                "grid_longitude": [0.0, 0.1, 0.2],
            }
        )

        aggregated = _aggregate_spatial_selection(
            grid_frame=grid_frame,
            latitude=0.0,
            longitude=0.0,
            spatial_mode="boundary",
            radius_km=None,
            boundary_geojson={
                "type": "Polygon",
                "coordinates": [
                    [
                        [-0.05, -0.05],
                        [0.15, -0.05],
                        [0.15, 0.05],
                        [-0.05, 0.05],
                        [-0.05, -0.05],
                    ]
                ],
            },
            boundary_bbox=(-0.05, 0.05, -0.05, 0.15),
        )

        self.assertEqual(aggregated["temperature_c"].tolist(), [15.0])

    def test_parse_temperature_file_reports_unexpected_binary_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "response.bin"
            path.write_bytes(b"BINARY\xc7\x00payload")

            with self.assertRaises(CDSRequestError) as context:
                parse_temperature_file(path, target_latitude=0.0, target_longitude=0.0)

        self.assertIn("binary file", str(context.exception))

    def test_fetch_point_temperature_series_reuses_cached_response(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        class FakeClient:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            def retrieve(self, dataset_name: str, request: dict[str, object], target: str) -> None:
                calls.append((dataset_name, request))
                Path(target).write_bytes(b"CDF")

        aggregated = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2020-01-01T00:00:00Z"),
                    pd.Timestamp("2020-02-01T00:00:00Z"),
                ],
                "temperature_c": [1.5, 2.5],
                "sample_days": [31, 29],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            fake_cdsapi = SimpleNamespace(Client=FakeClient)

            with patch.dict("sys.modules", {"cdsapi": fake_cdsapi}):
                with patch("mystripes.cds.parse_temperature_file", return_value=pd.DataFrame()):
                    with patch("mystripes.cds._aggregate_spatial_selection", return_value=aggregated):
                        first = fetch_point_temperature_series(
                            config=CDSConfig(url="https://example.invalid/api", key="secret-token"),
                            latitude=48.2082,
                            longitude=16.3738,
                            start_date=date(2020, 1, 1),
                            end_date=date(2020, 2, 29),
                            cache_dir=cache_dir,
                        )
                        second = fetch_point_temperature_series(
                            config=CDSConfig(url="https://example.invalid/api", key="different-token"),
                            latitude=48.2082,
                            longitude=16.3738,
                            start_date=date(2020, 1, 1),
                            end_date=date(2020, 2, 29),
                            cache_dir=cache_dir,
                        )

        self.assertEqual(len(calls), 1)
        pd.testing.assert_frame_equal(first, aggregated)
        pd.testing.assert_frame_equal(second, aggregated)


    def test_fetch_point_temperature_series_reports_request_progress(self) -> None:
        events: list[dict[str, object]] = []

        class FakeClient:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            def retrieve(self, dataset_name: str, request: dict[str, object], target: str) -> None:
                Path(target).write_bytes(b"CDF")

        aggregated = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2020-01-01T00:00:00Z")],
                "temperature_c": [1.5],
                "sample_days": [31],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_cdsapi = SimpleNamespace(Client=FakeClient)
            with patch.dict("sys.modules", {"cdsapi": fake_cdsapi}):
                with patch("mystripes.cds.parse_temperature_file", return_value=pd.DataFrame()):
                    with patch("mystripes.cds._aggregate_spatial_selection", return_value=aggregated):
                        frame = fetch_point_temperature_series(
                            config=CDSConfig(url="https://example.invalid/api", key="secret-token"),
                            latitude=48.2082,
                            longitude=16.3738,
                            start_date=date(2020, 1, 1),
                            end_date=date(2020, 2, 29),
                            cache_dir=Path(tmpdir) / "cache",
                            progress_callback=events.append,
                        )

        self.assertEqual(
            [event["stage"] for event in events],
            ["request_plan", "request_started", "request_finished", "point_fetch_completed"],
        )
        self.assertEqual(events[1]["request_index"], 1)
        self.assertEqual(events[1]["request_count"], 1)
        pd.testing.assert_frame_equal(frame, aggregated)

    def test_fetch_saved_temperature_series_forwards_progress_events(self) -> None:
        jan_feb = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2020-01-01T00:00:00Z"),
                    pd.Timestamp("2020-02-01T00:00:00Z"),
                ],
                "temperature_c": [1.5, 2.5],
                "sample_days": [31, 29],
            }
        )
        march = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2020-03-01T00:00:00Z")],
                "temperature_c": [3.5],
                "sample_days": [31],
            }
        )
        events: list[dict[str, object]] = []

        def fake_fetch(**kwargs):
            progress_callback = kwargs.get("progress_callback")
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "request_plan",
                        "request_count": 1,
                        "start_date": kwargs["start_date"].isoformat(),
                        "end_date": kwargs["end_date"].isoformat(),
                    }
                )
                progress_callback(
                    {
                        "stage": "request_started",
                        "request_index": 1,
                        "request_count": 1,
                        "request_year_start": "2020",
                        "request_year_end": "2020",
                        "month_count": 1,
                    }
                )
                progress_callback(
                    {
                        "stage": "request_finished",
                        "request_index": 1,
                        "request_count": 1,
                        "request_year_start": "2020",
                        "request_year_end": "2020",
                        "month_count": 1,
                    }
                )
                progress_callback(
                    {
                        "stage": "point_fetch_completed",
                        "request_count": 1,
                    }
                )
            if kwargs["end_date"] == date(2020, 2, 29):
                return jan_feb
            return march

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "timeline-cache"
            with patch("mystripes.cds.fetch_point_temperature_series", side_effect=fake_fetch):
                fetch_saved_temperature_series(
                    config=CDSConfig(url="https://example.invalid/api", key="secret-token"),
                    latitude=48.2082,
                    longitude=16.3738,
                    start_date=date(2020, 1, 1),
                    end_date=date(2020, 2, 29),
                    cache_dir=cache_dir,
                    request_cache_dir=Path(tmpdir) / "request-cache",
                    progress_callback=events.append,
                )
                events.clear()
                fetch_saved_temperature_series(
                    config=CDSConfig(url="https://example.invalid/api", key="secret-token"),
                    latitude=48.2082,
                    longitude=16.3738,
                    start_date=date(2020, 1, 1),
                    end_date=date(2020, 3, 31),
                    cache_dir=cache_dir,
                    request_cache_dir=Path(tmpdir) / "request-cache",
                    progress_callback=events.append,
                )

        self.assertEqual(
            [event["stage"] for event in events],
            [
                "timeline_fetch_plan",
                "missing_range_started",
                "request_plan",
                "request_started",
                "request_finished",
                "point_fetch_completed",
                "missing_range_finished",
                "timeline_fetch_completed",
            ],
        )
        self.assertEqual(events[1]["range_start"], "2020-03-01")
        self.assertEqual(events[4]["range_end"], "2020-03-31")

    def test_fetch_saved_temperature_series_only_fetches_missing_tail_months(self) -> None:
        jan_feb = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2020-01-01T00:00:00Z"),
                    pd.Timestamp("2020-02-01T00:00:00Z"),
                ],
                "temperature_c": [1.5, 2.5],
                "sample_days": [31, 29],
            }
        )
        march = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2020-03-01T00:00:00Z")],
                "temperature_c": [3.5],
                "sample_days": [31],
            }
        )
        fetch_calls: list[tuple[date, date]] = []

        def fake_fetch(**kwargs):
            fetch_calls.append((kwargs["start_date"], kwargs["end_date"]))
            if (kwargs["start_date"], kwargs["end_date"]) == (date(2020, 1, 1), date(2020, 2, 29)):
                return jan_feb
            if (kwargs["start_date"], kwargs["end_date"]) == (date(2020, 3, 1), date(2020, 3, 31)):
                return march
            raise AssertionError(f"Unexpected fetch range: {kwargs['start_date']} to {kwargs['end_date']}")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "timeline-cache"
            with patch("mystripes.cds.fetch_point_temperature_series", side_effect=fake_fetch):
                first = fetch_saved_temperature_series(
                    config=CDSConfig(url="https://example.invalid/api", key="secret-token"),
                    latitude=48.2082,
                    longitude=16.3738,
                    start_date=date(2020, 1, 1),
                    end_date=date(2020, 2, 29),
                    cache_dir=cache_dir,
                    request_cache_dir=Path(tmpdir) / "request-cache",
                )
                second = fetch_saved_temperature_series(
                    config=CDSConfig(url="https://example.invalid/api", key="secret-token"),
                    latitude=48.2082,
                    longitude=16.3738,
                    start_date=date(2020, 1, 1),
                    end_date=date(2020, 3, 31),
                    cache_dir=cache_dir,
                    request_cache_dir=Path(tmpdir) / "request-cache",
                )
                third = fetch_saved_temperature_series(
                    config=CDSConfig(url="https://example.invalid/api", key="secret-token"),
                    latitude=48.2082,
                    longitude=16.3738,
                    start_date=date(2020, 1, 1),
                    end_date=date(2020, 3, 31),
                    cache_dir=cache_dir,
                    request_cache_dir=Path(tmpdir) / "request-cache",
                )

        self.assertEqual(
            fetch_calls,
            [
                (date(2020, 1, 1), date(2020, 2, 29)),
                (date(2020, 3, 1), date(2020, 3, 31)),
            ],
        )
        pd.testing.assert_frame_equal(first, jan_feb)
        pd.testing.assert_frame_equal(
            second,
            pd.concat([jan_feb, march], ignore_index=True),
        )
        pd.testing.assert_frame_equal(third, second)


if __name__ == "__main__":
    unittest.main()
