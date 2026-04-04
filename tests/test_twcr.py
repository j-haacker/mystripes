from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from mystripes.twcr import (
    fetch_saved_twcr_temperature_series,
    fetch_twcr_temperature_series,
    plan_twcr_downloads,
)


def _aggregated_month_frame(start: str, periods: int, value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=periods, freq="MS", tz="UTC"),
            "temperature_c": [value] * periods,
            "sample_days": [31] * periods,
        }
    )


def _parsed_grid_stub() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("1938-01-01T00:00:00Z")],
            "temperature_c": [0.0],
            "sample_days": [31],
            "grid_latitude": [48.0],
            "grid_longitude": [16.0],
        }
    )


class HistoricalTwentyCRTests(unittest.TestCase):
    def test_plan_twcr_downloads_returns_one_window_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            grid_cache_dir = Path(tmpdir) / "grid"
            works = plan_twcr_downloads(
                latitude=48.2,
                longitude=16.4,
                start_date=date(1938, 1, 1),
                end_date=date(1939, 12, 31),
                spatial_mode="single_cell",
                radius_km=None,
                boundary_geojson=None,
                boundary_bbox=None,
                grid_request_cache_dir=grid_cache_dir,
            )

        self.assertEqual(len(works), 1)
        self.assertEqual(works[0].task_kind, "twcr_grid")
        self.assertEqual(works[0].start_date, date(1938, 1, 1))
        self.assertEqual(works[0].end_date, date(1939, 12, 31))
        self.assertTrue(str(works[0].work_id).startswith(str(grid_cache_dir)))

    def test_fetch_twcr_temperature_series_uses_single_bbox_window_download(self) -> None:
        download_windows: list[tuple[date, date]] = []

        def fake_download_window(
            start_date: date,
            end_date: date,
            area,
            target_path: Path,
        ) -> None:
            download_windows.append((start_date, end_date))
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(b"CDF")

        with tempfile.TemporaryDirectory() as tmpdir:
            grid_cache_dir = Path(tmpdir) / "grid"
            window_cache_dir = Path(tmpdir) / "window"
            with patch(
                "mystripes.twcr._download_twcr_subset_window_file",
                side_effect=fake_download_window,
            ), patch(
                "mystripes.twcr.parse_temperature_file",
                return_value=_parsed_grid_stub(),
            ), patch(
                "mystripes.twcr._aggregate_spatial_selection",
                return_value=_aggregated_month_frame("1938-01-01", 24, 1938.0),
            ):
                result = fetch_twcr_temperature_series(
                    latitude=48.2,
                    longitude=16.4,
                    start_date=date(1938, 1, 1),
                    end_date=date(1939, 12, 31),
                    cache_dir=window_cache_dir,
                    grid_request_cache_dir=grid_cache_dir,
                )

        self.assertEqual(download_windows, [(date(1938, 1, 1), date(1939, 12, 31))])
        self.assertEqual(result["temperature_c"].tolist(), [1938.0] * 24)

    def test_fetch_twcr_temperature_series_reuses_shared_bbox_window_across_locations(self) -> None:
        download_windows: list[tuple[date, date]] = []

        def fake_download_window(
            start_date: date,
            end_date: date,
            area,
            target_path: Path,
        ) -> None:
            download_windows.append((start_date, end_date))
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(b"CDF")

        with tempfile.TemporaryDirectory() as tmpdir:
            grid_cache_dir = Path(tmpdir) / "grid"
            window_cache_dir = Path(tmpdir) / "window"
            shared_boundary_bbox = (47.5, 48.5, 16.0, 17.5)
            with patch(
                "mystripes.twcr._download_twcr_subset_window_file",
                side_effect=fake_download_window,
            ), patch(
                "mystripes.twcr.parse_temperature_file",
                return_value=_parsed_grid_stub(),
            ), patch(
                "mystripes.twcr._aggregate_spatial_selection",
                return_value=_aggregated_month_frame("1938-01-01", 24, 1938.0),
            ):
                first = fetch_twcr_temperature_series(
                    latitude=48.2,
                    longitude=16.4,
                    start_date=date(1938, 1, 1),
                    end_date=date(1939, 12, 31),
                    spatial_mode="boundary",
                    boundary_bbox=shared_boundary_bbox,
                    cache_dir=window_cache_dir,
                    grid_request_cache_dir=grid_cache_dir,
                )
                second = fetch_twcr_temperature_series(
                    latitude=48.1,
                    longitude=17.1,
                    start_date=date(1938, 1, 1),
                    end_date=date(1939, 12, 31),
                    spatial_mode="boundary",
                    boundary_bbox=shared_boundary_bbox,
                    cache_dir=window_cache_dir,
                    grid_request_cache_dir=grid_cache_dir,
                )

        self.assertEqual(download_windows, [(date(1938, 1, 1), date(1939, 12, 31))])
        self.assertEqual(first["temperature_c"].tolist(), [1938.0] * 24)
        self.assertEqual(second["temperature_c"].tolist(), [1938.0] * 24)

    def test_fetch_twcr_temperature_series_recovers_from_unreadable_bbox_cache(self) -> None:
        download_windows: list[tuple[date, date]] = []
        parse_calls: list[str] = []

        def fake_download_window(
            start_date: date,
            end_date: date,
            area,
            target_path: Path,
        ) -> None:
            download_windows.append((start_date, end_date))
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(b"CDF")

        def fake_parse(path: Path, target_latitude: float, target_longitude: float):
            parse_calls.append(path.name)
            if len(parse_calls) == 1:
                raise RuntimeError("NetCDF: Can't open HDF5 attribute")
            return _parsed_grid_stub()

        with tempfile.TemporaryDirectory() as tmpdir:
            grid_cache_dir = Path(tmpdir) / "grid"
            window_cache_dir = Path(tmpdir) / "window"
            broken_path = grid_cache_dir / "broken.nc"
            broken_path.parent.mkdir(parents=True, exist_ok=True)
            broken_path.write_bytes(b"\x89HDF\r\n\x1a\n")

            with patch(
                "mystripes.twcr._resolve_twcr_window_source_path",
                return_value=(broken_path, "bbox_window", "local_cache"),
            ), patch(
                "mystripes.twcr._download_twcr_subset_window_file",
                side_effect=fake_download_window,
            ), patch(
                "mystripes.twcr.parse_temperature_file",
                side_effect=fake_parse,
            ), patch(
                "mystripes.twcr._aggregate_spatial_selection",
                return_value=_aggregated_month_frame("1938-01-01", 12, 1938.0),
            ):
                result = fetch_twcr_temperature_series(
                    latitude=48.2,
                    longitude=16.4,
                    start_date=date(1938, 1, 1),
                    end_date=date(1938, 12, 31),
                    cache_dir=window_cache_dir,
                    grid_request_cache_dir=grid_cache_dir,
                )

        self.assertEqual(download_windows, [(date(1938, 1, 1), date(1938, 12, 31))])
        self.assertEqual(result["temperature_c"].tolist(), [1938.0] * 12)

    def test_fetch_saved_twcr_temperature_series_forwards_request_area_overrides(self) -> None:
        with patch(
            "mystripes.twcr.fetch_twcr_temperature_series",
            return_value=_aggregated_month_frame("1938-01-01", 12, 1938.0),
        ) as fetch_twcr_temperature_series_mock:
            result = fetch_saved_twcr_temperature_series(
                latitude=48.2,
                longitude=16.4,
                start_date=date(1938, 1, 1),
                end_date=date(1938, 12, 31),
                request_area_overrides={
                    (date(1938, 1, 1), date(1938, 12, 31)): (48.0, 16.0, 48.0, 17.0),
                },
                cache_dir=None,
                request_cache_dir=None,
                grid_request_cache_dir=None,
            )

        self.assertEqual(result["temperature_c"].tolist(), [1938.0] * 12)
        self.assertEqual(
            fetch_twcr_temperature_series_mock.call_args.kwargs["request_area_override"],
            (48.0, 16.0, 48.0, 17.0),
        )

    def test_fetch_saved_twcr_temperature_series_reuses_timeline_cache(self) -> None:
        download_windows: list[tuple[date, date]] = []
        events: list[dict[str, object]] = []

        def fake_download_window(
            start_date: date,
            end_date: date,
            area,
            target_path: Path,
        ) -> None:
            download_windows.append((start_date, end_date))
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(b"CDF")

        with tempfile.TemporaryDirectory() as tmpdir:
            grid_cache_dir = Path(tmpdir) / "grid"
            timeline_cache_dir = Path(tmpdir) / "timeline"
            window_cache_dir = Path(tmpdir) / "window"
            with patch(
                "mystripes.twcr._download_twcr_subset_window_file",
                side_effect=fake_download_window,
            ), patch(
                "mystripes.twcr.parse_temperature_file",
                return_value=_parsed_grid_stub(),
            ), patch(
                "mystripes.twcr._aggregate_spatial_selection",
                return_value=_aggregated_month_frame("1938-01-01", 12, 1938.0),
            ):
                first = fetch_saved_twcr_temperature_series(
                    latitude=48.2,
                    longitude=16.4,
                    start_date=date(1938, 1, 1),
                    end_date=date(1938, 12, 31),
                    cache_dir=timeline_cache_dir,
                    request_cache_dir=window_cache_dir,
                    grid_request_cache_dir=grid_cache_dir,
                )
                second = fetch_saved_twcr_temperature_series(
                    latitude=48.2,
                    longitude=16.4,
                    start_date=date(1938, 1, 1),
                    end_date=date(1938, 12, 31),
                    cache_dir=timeline_cache_dir,
                    request_cache_dir=window_cache_dir,
                    grid_request_cache_dir=grid_cache_dir,
                    progress_callback=events.append,
                )

        self.assertEqual(download_windows, [(date(1938, 1, 1), date(1938, 12, 31))])
        self.assertEqual(events[0]["stage"], "timeline_cache_hit")
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
