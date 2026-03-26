from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from personal_warming_stripes.models import LifePeriod
from personal_warming_stripes.processing import (
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
                "timestamp": pd.to_datetime(["2000-01-01T00:00:00Z", "2000-07-01T00:00:00Z"], utc=True),
                "temperature_c": [10.0, 12.0],
            }
        )
        frame_b = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2001-01-01T00:00:00Z", "2001-07-01T00:00:00Z"], utc=True),
                "temperature_c": [13.0, 15.0],
            }
        )

        _, yearly = combine_period_frames(periods, [frame_a, frame_b])
        self.assertEqual(yearly["year"].tolist(), [2000, 2001])
        self.assertEqual(yearly["mean_temp_c"].round(2).tolist(), [11.0, 14.0])

        baseline = calculate_life_period_baseline(yearly)
        stripe_frame = build_stripe_frame(yearly, baseline)
        self.assertAlmostEqual(baseline, 12.5)
        self.assertEqual(stripe_frame["anomaly_c"].round(2).tolist(), [-1.5, 1.5])

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


if __name__ == "__main__":
    unittest.main()
