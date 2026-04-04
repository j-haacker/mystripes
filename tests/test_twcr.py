from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from mystripes.twcr import (
    estimate_missing_twcr_years,
    fetch_saved_twcr_temperature_series,
    fetch_twcr_temperature_series,
    plan_twcr_downloads,
)


def _aggregated_year_frame(year: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(f"{year}-01-01", periods=12, freq="MS", tz="UTC"),
            "temperature_c": [float(year)] * 12,
            "sample_days": [31] * 12,
        }
    )


class HistoricalTwentyCRTests(unittest.TestCase):
    def test_estimate_missing_twcr_years_skips_cached_raw_year_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_cache_dir = Path(tmpdir)
            (raw_cache_dir / "air.2m.mon.mean.1938.nc").write_bytes(b"CDF")

            missing = estimate_missing_twcr_years(
                date(1938, 1, 1),
                date(1939, 12, 31),
                raw_year_cache_dir=raw_cache_dir,
            )

        self.assertEqual(missing, [1939])

    def test_fetch_twcr_temperature_series_reuses_cached_raw_year_files_across_locations(self) -> None:
        download_years: list[int] = []

        def fake_download(year: int, target_path: Path) -> None:
            download_years.append(year)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(b"CDF")

        def fake_aggregate(*args, **kwargs):
            year = int(pd.Timestamp(kwargs["grid_frame"]["timestamp"].iloc[0]).year)
            return _aggregated_year_frame(year)

        parsed_1938 = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("1938-01-01T00:00:00Z")],
                "temperature_c": [0.0],
                "sample_days": [31],
                "grid_latitude": [0.0],
                "grid_longitude": [0.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_cache_dir = Path(tmpdir) / "raw"
            window_cache_dir = Path(tmpdir) / "window"
            with patch("mystripes.twcr._download_twcr_year_file", side_effect=fake_download), patch(
                "mystripes.twcr.parse_temperature_file",
                return_value=parsed_1938,
            ), patch(
                "mystripes.twcr._aggregate_spatial_selection",
                side_effect=fake_aggregate,
            ):
                first = fetch_twcr_temperature_series(
                    latitude=48.2,
                    longitude=16.4,
                    start_date=date(1938, 1, 1),
                    end_date=date(1938, 12, 31),
                    spatial_mode="boundary",
                    boundary_bbox=(30.0, 70.0, -20.0, 80.0),
                    cache_dir=window_cache_dir,
                    raw_year_cache_dir=raw_cache_dir,
                )
                second = fetch_twcr_temperature_series(
                    latitude=52.5,
                    longitude=13.4,
                    start_date=date(1938, 1, 1),
                    end_date=date(1938, 12, 31),
                    spatial_mode="boundary",
                    boundary_bbox=(30.0, 70.0, -20.0, 80.0),
                    cache_dir=window_cache_dir,
                    raw_year_cache_dir=raw_cache_dir,
                )

        self.assertEqual(download_years, [1938])
        self.assertEqual(first["temperature_c"].tolist(), [1938.0] * 12)
        self.assertEqual(second["temperature_c"].tolist(), [1938.0] * 12)

    def test_plan_twcr_downloads_prefers_small_subset_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_cache_dir = Path(tmpdir) / "raw"
            grid_cache_dir = Path(tmpdir) / "grid"
            works = plan_twcr_downloads(
                latitude=48.2,
                longitude=16.4,
                start_date=date(1938, 1, 1),
                end_date=date(1938, 12, 31),
                spatial_mode="single_cell",
                radius_km=None,
                boundary_geojson=None,
                boundary_bbox=None,
                raw_year_cache_dir=raw_cache_dir,
                grid_request_cache_dir=grid_cache_dir,
            )

        self.assertEqual(len(works), 1)
        self.assertEqual(works[0].task_kind, "twcr_grid")
        self.assertTrue(str(works[0].work_id).startswith(str(grid_cache_dir)))

    def test_fetch_twcr_temperature_series_uses_subset_year_downloads_for_small_requests(self) -> None:
        downloaded_subset_years: list[int] = []

        def fake_download_subset(year: int, area, target_path: Path) -> None:
            downloaded_subset_years.append(year)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(b"CDF")

        parsed_1938 = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("1938-01-01T00:00:00Z")],
                "temperature_c": [0.0],
                "sample_days": [31],
                "grid_latitude": [48.0],
                "grid_longitude": [16.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_cache_dir = Path(tmpdir) / "raw"
            grid_cache_dir = Path(tmpdir) / "grid"
            window_cache_dir = Path(tmpdir) / "window"
            with patch("mystripes.twcr._download_twcr_subset_year_file", side_effect=fake_download_subset), patch(
                "mystripes.twcr._download_twcr_year_file",
                side_effect=AssertionError("full-year download should not run"),
            ), patch(
                "mystripes.twcr.parse_temperature_file",
                return_value=parsed_1938,
            ), patch(
                "mystripes.twcr._aggregate_spatial_selection",
                return_value=_aggregated_year_frame(1938),
            ):
                result = fetch_twcr_temperature_series(
                    latitude=48.2,
                    longitude=16.4,
                    start_date=date(1938, 1, 1),
                    end_date=date(1938, 12, 31),
                    cache_dir=window_cache_dir,
                    grid_request_cache_dir=grid_cache_dir,
                    raw_year_cache_dir=raw_cache_dir,
                )

        self.assertEqual(downloaded_subset_years, [1938])
        self.assertEqual(result["temperature_c"].tolist(), [1938.0] * 12)

    def test_fetch_twcr_temperature_series_recovers_from_unreadable_subset_cache(self) -> None:
        downloaded_raw_years: list[int] = []
        parse_calls: list[str] = []

        def fake_download_raw(year: int, target_path: Path) -> None:
            downloaded_raw_years.append(year)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(b"CDF")

        def fake_parse(path: Path, target_latitude: float, target_longitude: float):
            parse_calls.append(path.name)
            if path.suffix == ".nc" and len(parse_calls) == 1:
                raise RuntimeError("NetCDF: Can't open HDF5 attribute")
            return pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("1938-01-01T00:00:00Z")],
                    "temperature_c": [0.0],
                    "sample_days": [31],
                    "grid_latitude": [48.0],
                    "grid_longitude": [16.0],
                }
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_cache_dir = Path(tmpdir) / "raw"
            grid_cache_dir = Path(tmpdir) / "grid"
            window_cache_dir = Path(tmpdir) / "window"
            broken_subset_path = grid_cache_dir / "broken.nc"
            broken_subset_path.parent.mkdir(parents=True, exist_ok=True)
            broken_subset_path.write_bytes(b"\x89HDF\r\n\x1a\n")

            with patch(
                "mystripes.twcr._resolve_twcr_year_source_path",
                return_value=(broken_subset_path, "year_subset", "local_cache"),
            ), patch(
                "mystripes.twcr._download_twcr_year_file",
                side_effect=fake_download_raw,
            ), patch(
                "mystripes.twcr.parse_temperature_file",
                side_effect=fake_parse,
            ), patch(
                "mystripes.twcr._aggregate_spatial_selection",
                return_value=_aggregated_year_frame(1938),
            ):
                result = fetch_twcr_temperature_series(
                    latitude=48.2,
                    longitude=16.4,
                    start_date=date(1938, 1, 1),
                    end_date=date(1938, 12, 31),
                    cache_dir=window_cache_dir,
                    grid_request_cache_dir=grid_cache_dir,
                    raw_year_cache_dir=raw_cache_dir,
                )

        self.assertEqual(downloaded_raw_years, [1938])
        self.assertEqual(result["temperature_c"].tolist(), [1938.0] * 12)

    def test_fetch_twcr_temperature_series_recovers_from_unreadable_full_year_cache(self) -> None:
        downloaded_raw_years: list[int] = []
        parse_calls: list[str] = []

        def fake_download_raw(year: int, target_path: Path) -> None:
            downloaded_raw_years.append(year)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(b"CDF")

        def fake_parse(path: Path, target_latitude: float, target_longitude: float):
            parse_calls.append(path.name)
            if len(parse_calls) == 1:
                raise RuntimeError("NetCDF: Can't open HDF5 attribute")
            return pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("1938-01-01T00:00:00Z")],
                    "temperature_c": [0.0],
                    "sample_days": [31],
                    "grid_latitude": [48.0],
                    "grid_longitude": [16.0],
                }
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_cache_dir = Path(tmpdir) / "raw"
            window_cache_dir = Path(tmpdir) / "window"
            raw_path = raw_cache_dir / "air.2m.mon.mean.1938.nc"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(b"\x89HDF\r\n\x1a\n")

            with patch(
                "mystripes.twcr._resolve_twcr_year_source_path",
                return_value=(raw_path, "full_year", "local_cache"),
            ), patch(
                "mystripes.twcr._download_twcr_year_file",
                side_effect=fake_download_raw,
            ), patch(
                "mystripes.twcr.parse_temperature_file",
                side_effect=fake_parse,
            ), patch(
                "mystripes.twcr._aggregate_spatial_selection",
                return_value=_aggregated_year_frame(1938),
            ):
                result = fetch_twcr_temperature_series(
                    latitude=48.2,
                    longitude=16.4,
                    start_date=date(1938, 1, 1),
                    end_date=date(1938, 12, 31),
                    cache_dir=window_cache_dir,
                    raw_year_cache_dir=raw_cache_dir,
                    spatial_mode="boundary",
                    boundary_bbox=(30.0, 70.0, -20.0, 80.0),
                )

        self.assertEqual(downloaded_raw_years, [1938])
        self.assertEqual(result["temperature_c"].tolist(), [1938.0] * 12)

    def test_fetch_saved_twcr_temperature_series_reuses_timeline_cache(self) -> None:
        download_years: list[int] = []

        def fake_download(year: int, target_path: Path) -> None:
            download_years.append(year)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(b"CDF")

        parsed_1938 = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("1938-01-01T00:00:00Z")],
                "temperature_c": [0.0],
                "sample_days": [31],
                "grid_latitude": [0.0],
                "grid_longitude": [0.0],
            }
        )
        events: list[dict[str, object]] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_cache_dir = Path(tmpdir) / "raw"
            grid_cache_dir = Path(tmpdir) / "grid"
            timeline_cache_dir = Path(tmpdir) / "timeline"
            window_cache_dir = Path(tmpdir) / "window"
            with patch("mystripes.twcr._download_twcr_year_file", side_effect=fake_download), patch(
                "mystripes.twcr._download_twcr_subset_year_file",
                side_effect=lambda year, area, target_path: fake_download(year, target_path),
            ), patch(
                "mystripes.twcr.parse_temperature_file",
                return_value=parsed_1938,
            ), patch(
                "mystripes.twcr._aggregate_spatial_selection",
                return_value=_aggregated_year_frame(1938),
            ):
                first = fetch_saved_twcr_temperature_series(
                    latitude=48.2,
                    longitude=16.4,
                    start_date=date(1938, 1, 1),
                    end_date=date(1938, 12, 31),
                    cache_dir=timeline_cache_dir,
                    request_cache_dir=window_cache_dir,
                    grid_request_cache_dir=grid_cache_dir,
                    raw_year_cache_dir=raw_cache_dir,
                )
                second = fetch_saved_twcr_temperature_series(
                    latitude=48.2,
                    longitude=16.4,
                    start_date=date(1938, 1, 1),
                    end_date=date(1938, 12, 31),
                    cache_dir=timeline_cache_dir,
                    request_cache_dir=window_cache_dir,
                    grid_request_cache_dir=grid_cache_dir,
                    raw_year_cache_dir=raw_cache_dir,
                    progress_callback=events.append,
                )

        self.assertEqual(download_years, [1938])
        self.assertEqual(events[0]["stage"], "timeline_cache_hit")
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
