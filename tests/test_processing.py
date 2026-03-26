from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from mystrips.models import LifePeriod
from mystrips.processing import (
    build_location_baseline_stripe_frame,
    build_periods_from_entries,
    build_stripe_frame,
    calculate_life_period_baseline,
    calculate_weighted_location_baseline,
    combine_period_frames,
)


class ProcessingTests(unittest.TestCase):
    def test_build_periods_from_entries_creates_contiguous_ranges(self) -> None:
        periods, errors = build_periods_from_entries(
            entries=[
                {
                    "custom_label": "Vienna",
                    "place_query": "Vienna, Austria",
                    "resolved_name": "Vienna, Austria",
                    "latitude_text": "48.2082",
                    "longitude_text": "16.3738",
                    "end_date": date(2004, 6, 30),
                },
                {
                    "custom_label": "Berlin",
                    "place_query": "Berlin, Germany",
                    "resolved_name": "Berlin, Germany",
                    "latitude_text": "52.5200",
                    "longitude_text": "13.4050",
                    "end_date": None,
                },
            ],
            birth_date=date(2000, 1, 1),
            analysis_end=date(2005, 12, 31),
            analysis_min_start=date(1950, 1, 2),
        )

        self.assertFalse(errors)
        self.assertEqual(periods[0].start_date, date(2000, 1, 1))
        self.assertEqual(periods[0].end_date, date(2004, 6, 30))
        self.assertEqual(periods[1].start_date, date(2004, 7, 1))
        self.assertEqual(periods[1].end_date, date(2005, 12, 31))

    def test_combine_period_frames_and_build_stripes(self) -> None:
        periods = [
            LifePeriod(
                label="A",
                place_query="A",
                resolved_name="A",
                start_date=date(2000, 1, 1),
                end_date=date(2000, 12, 31),
                latitude=1.0,
                longitude=2.0,
            ),
            LifePeriod(
                label="B",
                place_query="B",
                resolved_name="B",
                start_date=date(2001, 1, 1),
                end_date=date(2001, 12, 31),
                latitude=3.0,
                longitude=4.0,
            ),
        ]

        frame_a = pd.DataFrame(
            {
                "timestamp": pd.date_range("2000-01-01", "2000-12-01", freq="MS", tz="UTC"),
                "temperature_c": [11.0] * 12,
            }
        )
        frame_b = pd.DataFrame(
            {
                "timestamp": pd.date_range("2001-01-01", "2001-12-01", freq="MS", tz="UTC"),
                "temperature_c": [14.0] * 12,
            }
        )

        _, yearly = combine_period_frames(periods, [frame_a, frame_b])
        self.assertEqual(yearly["year"].tolist(), [2000, 2001])
        self.assertEqual(yearly["mean_temp_c"].round(2).tolist(), [11.0, 14.0])
        self.assertEqual(yearly["window_start"].tolist(), [date(2000, 1, 1), date(2001, 1, 1)])
        self.assertEqual(yearly["window_end"].tolist(), [date(2000, 12, 31), date(2001, 12, 31)])

        baseline = calculate_life_period_baseline(yearly)
        stripe_frame = build_stripe_frame(yearly, baseline)
        self.assertAlmostEqual(baseline, (11.0 * 366 + 14.0 * 365) / (366 + 365))
        self.assertEqual(stripe_frame["anomaly_c"].round(2).tolist(), [-1.5, 1.5])

    def test_full_calendar_years_drop_partial_years(self) -> None:
        periods = [
            LifePeriod(
                label="A",
                place_query="A",
                resolved_name="A",
                start_date=date(2000, 6, 1),
                end_date=date(2002, 3, 31),
                latitude=1.0,
                longitude=2.0,
            )
        ]
        timestamps = pd.date_range("2000-06-01", "2002-03-01", freq="MS", tz="UTC")
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "temperature_c": [10.0] * len(timestamps),
            }
        )

        _, yearly = combine_period_frames(periods, [frame])

        self.assertEqual(yearly["year"].tolist(), [2001])
        self.assertEqual(yearly["window_start"].tolist(), [date(2001, 1, 1)])
        self.assertEqual(yearly["window_end"].tolist(), [date(2001, 12, 31)])
        self.assertEqual(yearly["days_covered"].tolist(), [365])

    def test_trailing_365_day_windows_anchor_to_latest_date(self) -> None:
        periods = [
            LifePeriod(
                label="A",
                place_query="A",
                resolved_name="A",
                start_date=date(2020, 1, 1),
                end_date=date(2022, 6, 30),
                latitude=1.0,
                longitude=2.0,
            )
        ]
        timestamps = pd.date_range("2020-01-01", "2022-06-01", freq="MS", tz="UTC")
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "temperature_c": [10.0] * len(timestamps),
            }
        )

        _, yearly = combine_period_frames(
            periods,
            [frame],
            aggregation_mode="rolling_365_day",
            rolling_window_end=date(2022, 6, 30),
        )

        self.assertEqual(yearly["window_end"].tolist(), [date(2021, 6, 30), date(2022, 6, 30)])
        self.assertEqual(yearly["window_start"].tolist(), [date(2020, 7, 1), date(2021, 7, 1)])
        self.assertEqual(yearly["days_covered"].tolist(), [365, 365])
        self.assertEqual(yearly["mean_temp_c"].round(2).tolist(), [10.0, 10.0])

    def test_weighted_location_baseline_uses_days_lived(self) -> None:
        periods = [
            LifePeriod(
                label="First",
                place_query="A",
                resolved_name="A",
                start_date=date(2000, 1, 1),
                end_date=date(2000, 1, 10),
                latitude=1.0,
                longitude=2.0,
            ),
            LifePeriod(
                label="Second",
                place_query="B",
                resolved_name="B",
                start_date=date(2000, 1, 11),
                end_date=date(2000, 1, 30),
                latitude=3.0,
                longitude=4.0,
            ),
        ]
        baseline = calculate_weighted_location_baseline(
            periods,
            baseline_by_location={
                periods[0].location_key: 10.0,
                periods[1].location_key: 20.0,
            },
        )
        self.assertAlmostEqual(baseline, (10 * 10 + 20 * 20) / 30)

    def test_location_specific_baselines_apply_per_location(self) -> None:
        periods = [
            LifePeriod(
                label="A",
                place_query="A",
                resolved_name="A",
                start_date=date(2000, 1, 1),
                end_date=date(2000, 12, 31),
                latitude=1.0,
                longitude=2.0,
            ),
            LifePeriod(
                label="B",
                place_query="B",
                resolved_name="B",
                start_date=date(2001, 1, 1),
                end_date=date(2001, 12, 31),
                latitude=3.0,
                longitude=4.0,
            ),
        ]
        frame_a = pd.DataFrame(
            {
                "timestamp": pd.date_range("2000-01-01", "2000-12-01", freq="MS", tz="UTC"),
                "temperature_c": [11.0] * 12,
            }
        )
        frame_b = pd.DataFrame(
            {
                "timestamp": pd.date_range("2001-01-01", "2001-12-01", freq="MS", tz="UTC"),
                "temperature_c": [21.0] * 12,
            }
        )

        combined, _ = combine_period_frames(periods, [frame_a, frame_b])
        stripe_frame = build_location_baseline_stripe_frame(
            combined=combined,
            baseline_by_location={
                periods[0].location_key: 10.0,
                periods[1].location_key: 20.0,
            },
        )

        self.assertEqual(stripe_frame["year"].tolist(), [2000, 2001])
        self.assertEqual(stripe_frame["baseline_c"].round(2).tolist(), [10.0, 20.0])
        self.assertEqual(stripe_frame["anomaly_c"].round(2).tolist(), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
