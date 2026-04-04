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
                    cache_dir=window_cache_dir,
                    raw_year_cache_dir=raw_cache_dir,
                )
                second = fetch_twcr_temperature_series(
                    latitude=52.5,
                    longitude=13.4,
                    start_date=date(1938, 1, 1),
                    end_date=date(1938, 12, 31),
                    cache_dir=window_cache_dir,
                    raw_year_cache_dir=raw_cache_dir,
                )

        self.assertEqual(download_years, [1938])
        self.assertEqual(first["temperature_c"].tolist(), [1938.0] * 12)
        self.assertEqual(second["temperature_c"].tolist(), [1938.0] * 12)

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
            timeline_cache_dir = Path(tmpdir) / "timeline"
            window_cache_dir = Path(tmpdir) / "window"
            with patch("mystripes.twcr._download_twcr_year_file", side_effect=fake_download), patch(
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
                    raw_year_cache_dir=raw_cache_dir,
                )
                second = fetch_saved_twcr_temperature_series(
                    latitude=48.2,
                    longitude=16.4,
                    start_date=date(1938, 1, 1),
                    end_date=date(1938, 12, 31),
                    cache_dir=timeline_cache_dir,
                    request_cache_dir=window_cache_dir,
                    raw_year_cache_dir=raw_cache_dir,
                    progress_callback=events.append,
                )

        self.assertEqual(download_years, [1938])
        self.assertEqual(events[0]["stage"], "timeline_cache_hit")
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
