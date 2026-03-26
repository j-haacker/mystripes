from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from mystripes.models import LifePeriod
from mystripes.processing import (
    aggregate_daily_series_to_stripes,
    build_merged_daily_series,
    build_periods_from_entries,
    build_period_report_tables,
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

    def test_daily_series_aggregates_to_full_year_stripes(self) -> None:
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

        daily_series = build_merged_daily_series(
            periods=periods,
            frames_by_period=[frame_a, frame_b],
            report_start=date(2000, 1, 1),
            report_end=date(2001, 12, 31),
        )
        yearly = aggregate_daily_series_to_stripes(daily_series)
        self.assertEqual(yearly["year"].tolist(), [2000, 2001])
        self.assertEqual(yearly["mean_temp_c"].round(2).tolist(), [11.0, 14.0])
        self.assertEqual(yearly["baseline_c"].round(2).tolist(), [11.0, 14.0])
        self.assertEqual(yearly["anomaly_c"].round(2).tolist(), [0.0, 0.0])
        self.assertEqual(yearly["window_start"].tolist(), [date(2000, 1, 1), date(2001, 1, 1)])
        self.assertEqual(yearly["window_end"].tolist(), [date(2000, 12, 31), date(2001, 12, 31)])

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

        daily_series = build_merged_daily_series(
            periods=periods,
            frames_by_period=[frame],
            report_start=date(2000, 6, 1),
            report_end=date(2002, 3, 31),
        )
        yearly = aggregate_daily_series_to_stripes(daily_series)

        self.assertEqual(yearly["year"].tolist(), [2001])
        self.assertEqual(yearly["window_start"].tolist(), [date(2001, 1, 1)])
        self.assertEqual(yearly["window_end"].tolist(), [date(2001, 12, 31)])
        self.assertEqual(yearly["days_covered"].tolist(), [365])

    def test_rolling_moving_average_samples_monthly_by_default(self) -> None:
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

        daily_series = build_merged_daily_series(
            periods=periods,
            frames_by_period=[frame],
            report_start=date(2020, 1, 1),
            report_end=date(2022, 6, 30),
        )
        yearly = aggregate_daily_series_to_stripes(
            daily_series,
            aggregation_mode="rolling_365_day",
            rolling_window_end=date(2022, 6, 30),
        )

        self.assertEqual(len(yearly), 19)
        self.assertEqual(yearly["sample_date"].tolist()[0], date(2020, 12, 31))
        self.assertEqual(yearly["sample_date"].tolist()[-1], date(2022, 6, 30))
        self.assertEqual(yearly["window_start"].tolist()[0], date(2020, 1, 2))
        self.assertEqual(yearly["window_end"].tolist()[-1], date(2022, 6, 30))
        self.assertTrue((yearly["days_covered"] == 365).all())
        self.assertTrue((yearly["mean_temp_c"].round(2) == 10.0).all())

    def test_rolling_moving_average_keeps_prior_year_until_after_roll(self) -> None:
        periods = [
            LifePeriod(
                label="A",
                place_query="A",
                resolved_name="A",
                start_date=date(2020, 1, 1),
                end_date=date(2020, 3, 31),
                latitude=1.0,
                longitude=2.0,
            )
        ]
        timestamps = pd.date_range("2019-01-01", "2020-03-01", freq="MS", tz="UTC")
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "temperature_c": [10.0] * len(timestamps),
            }
        )

        daily_series = build_merged_daily_series(
            periods=periods,
            frames_by_period=[frame],
            report_start=date(2019, 1, 1),
            report_end=date(2020, 3, 31),
            first_period_history_days=364,
        )
        yearly = aggregate_daily_series_to_stripes(
            daily_series,
            aggregation_mode="rolling_365_day",
            rolling_window_end=date(2020, 3, 31),
            rolling_crop_start=date(2020, 1, 1),
        )

        self.assertEqual(yearly["sample_date"].tolist(), [date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31)])
        self.assertEqual(yearly["window_start"].tolist()[0], date(2019, 2, 1))
        self.assertTrue((yearly["mean_temp_c"].round(2) == 10.0).all())

    def test_rolling_moving_average_can_use_fixed_strip_count(self) -> None:
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

        daily_series = build_merged_daily_series(
            periods=periods,
            frames_by_period=[frame],
            report_start=date(2020, 1, 1),
            report_end=date(2022, 6, 30),
        )
        yearly = aggregate_daily_series_to_stripes(
            daily_series,
            aggregation_mode="rolling_365_day",
            rolling_window_end=date(2022, 6, 30),
            rolling_sample_mode="fixed_count",
            rolling_strip_count=4,
        )

        self.assertEqual(len(yearly), 4)
        self.assertEqual(yearly["sample_date"].tolist()[0], date(2020, 12, 30))
        self.assertEqual(yearly["sample_date"].tolist()[-1], date(2022, 6, 30))
        self.assertTrue((yearly["days_covered"] == 365).all())
        self.assertTrue((yearly["mean_temp_c"].round(2) == 10.0).all())

    def test_period_report_tables_keep_one_column_per_entered_period(self) -> None:
        periods = [
            LifePeriod(
                label="Vienna childhood",
                place_query="Vienna",
                resolved_name="Vienna, Austria",
                start_date=date(2000, 1, 1),
                end_date=date(2000, 1, 31),
                latitude=48.2082,
                longitude=16.3738,
            ),
            LifePeriod(
                label="Vienna return",
                place_query="Vienna",
                resolved_name="Vienna, Austria",
                start_date=date(2000, 2, 1),
                end_date=date(2000, 2, 29),
                latitude=48.2082,
                longitude=16.3738,
            ),
        ]
        timestamps = pd.date_range("2000-01-01", "2000-02-01", freq="MS", tz="UTC")
        frame_a = pd.DataFrame({"timestamp": timestamps, "temperature_c": [1.0, 2.0]})
        frame_b = pd.DataFrame({"timestamp": timestamps, "temperature_c": [10.0, 20.0]})

        full_report, merged_report = build_period_report_tables(
            periods=periods,
            frames_by_period=[frame_a, frame_b],
            period_baselines=[0.5, 15.0],
            report_start=date(2000, 1, 1),
            report_end=date(2000, 2, 29),
        )

        self.assertEqual(
            full_report["Period 1: Vienna childhood temperature_c"].round(2).tolist(),
            [1.0, 2.0],
        )
        self.assertEqual(
            full_report["Period 2: Vienna return temperature_c"].round(2).tolist(),
            [10.0, 20.0],
        )
        self.assertEqual(
            full_report["Period 1: Vienna childhood anomaly_c"].round(2).tolist(),
            [0.5, 1.5],
        )
        self.assertEqual(
            full_report["Period 2: Vienna return anomaly_c"].round(2).tolist(),
            [-5.0, 5.0],
        )
        self.assertEqual(
            merged_report["current_period"].tolist(),
            ["Period 1: Vienna childhood", "Period 2: Vienna return"],
        )
        self.assertEqual(merged_report["temperature_c"].round(2).tolist(), [1.0, 20.0])

    def test_period_report_tables_derive_different_means_per_period(self) -> None:
        periods = [
            LifePeriod(
                label="Cold place",
                place_query="A",
                resolved_name="A",
                start_date=date(2000, 1, 1),
                end_date=date(2000, 1, 31),
                latitude=1.0,
                longitude=2.0,
            ),
            LifePeriod(
                label="Warm place",
                place_query="B",
                resolved_name="B",
                start_date=date(2000, 2, 1),
                end_date=date(2000, 2, 29),
                latitude=3.0,
                longitude=4.0,
            ),
        ]
        timestamps = pd.date_range("2000-01-01", "2000-02-01", freq="MS", tz="UTC")
        cold_frame = pd.DataFrame({"timestamp": timestamps, "temperature_c": [1.0, 3.0]})
        warm_frame = pd.DataFrame({"timestamp": timestamps, "temperature_c": [11.0, 15.0]})

        full_report, _ = build_period_report_tables(
            periods=periods,
            frames_by_period=[cold_frame, warm_frame],
            period_baselines=None,
            report_start=date(2000, 1, 1),
            report_end=date(2000, 2, 29),
        )

        self.assertEqual(
            full_report["Period 1: Cold place climatology_c"].round(2).tolist(),
            [1.0, 3.0],
        )
        self.assertEqual(
            full_report["Period 2: Warm place climatology_c"].round(2).tolist(),
            [11.0, 15.0],
        )
        self.assertEqual(
            full_report["Period 1: Cold place anomaly_c"].round(2).tolist(),
            [0.0, 0.0],
        )
        self.assertEqual(
            full_report["Period 2: Warm place anomaly_c"].round(2).tolist(),
            [0.0, 0.0],
        )

    def test_reference_window_controls_day_of_year_climatology(self) -> None:
        periods = [
            LifePeriod(
                label="Single place",
                place_query="A",
                resolved_name="A",
                start_date=date(2019, 1, 1),
                end_date=date(2020, 12, 31),
                latitude=1.0,
                longitude=2.0,
            )
        ]
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2019-01-01", "2020-12-01", freq="MS", tz="UTC"),
                "temperature_c": [1.0] * 12 + [11.0] * 12,
            }
        )

        merged_daily = build_merged_daily_series(
            periods=periods,
            frames_by_period=[frame],
            report_start=date(2019, 1, 1),
            report_end=date(2020, 12, 31),
            baseline_start=date(2020, 1, 1),
            baseline_end=date(2020, 12, 31),
        )

        stripe_frame = aggregate_daily_series_to_stripes(merged_daily)
        self.assertEqual(stripe_frame["baseline_c"].round(2).tolist(), [11.0, 11.0])
        self.assertEqual(stripe_frame["anomaly_c"].round(2).tolist(), [-10.0, 0.0])

    def test_repeated_location_uses_shared_full_timeline_climatology(self) -> None:
        periods = [
            LifePeriod(
                label="A first",
                place_query="A",
                resolved_name="A",
                start_date=date(2000, 1, 1),
                end_date=date(2000, 12, 31),
                latitude=1.0,
                longitude=2.0,
            ),
            LifePeriod(
                label="B middle",
                place_query="B",
                resolved_name="B",
                start_date=date(2001, 1, 1),
                end_date=date(2001, 12, 31),
                latitude=3.0,
                longitude=4.0,
            ),
            LifePeriod(
                label="A again",
                place_query="A",
                resolved_name="A",
                start_date=date(2002, 1, 1),
                end_date=date(2002, 12, 31),
                latitude=1.0,
                longitude=2.0,
            ),
        ]
        full_timeline = pd.date_range("2000-01-01", "2002-12-01", freq="MS", tz="UTC")
        frame_a = pd.DataFrame(
            {
                "timestamp": full_timeline,
                "temperature_c": [5.0] * 12 + [10.0] * 12 + [15.0] * 12,
            }
        )
        frame_b = pd.DataFrame(
            {
                "timestamp": full_timeline,
                "temperature_c": [20.0] * len(full_timeline),
            }
        )

        merged_daily = build_merged_daily_series(
            periods=periods,
            frames_by_period=[frame_a, frame_b, frame_a],
            report_start=date(2000, 1, 1),
            report_end=date(2002, 12, 31),
            baseline_start=date(2000, 1, 1),
            baseline_end=date(2002, 12, 31),
        )

        stripe_frame = aggregate_daily_series_to_stripes(merged_daily)
        self.assertEqual(stripe_frame["year"].tolist(), [2000, 2001, 2002])
        self.assertAlmostEqual(float(stripe_frame["baseline_c"].iloc[0]), 10.0, delta=0.02)
        self.assertAlmostEqual(float(stripe_frame["baseline_c"].iloc[1]), 20.0, delta=0.02)
        self.assertAlmostEqual(float(stripe_frame["baseline_c"].iloc[2]), 10.0, delta=0.02)
        self.assertAlmostEqual(float(stripe_frame["anomaly_c"].iloc[0]), -5.0, delta=0.02)
        self.assertAlmostEqual(float(stripe_frame["anomaly_c"].iloc[1]), 0.0, delta=0.02)
        self.assertAlmostEqual(float(stripe_frame["anomaly_c"].iloc[2]), 5.0, delta=0.02)

    def test_build_merged_daily_series_computes_anomalies_before_merge(self) -> None:
        periods = [
            LifePeriod(
                label="First",
                place_query="A",
                resolved_name="A",
                start_date=date(2000, 1, 1),
                end_date=date(2000, 6, 30),
                latitude=1.0,
                longitude=2.0,
            ),
            LifePeriod(
                label="Second",
                place_query="B",
                resolved_name="B",
                start_date=date(2000, 7, 1),
                end_date=date(2000, 12, 31),
                latitude=3.0,
                longitude=4.0,
            ),
        ]
        year_2000 = pd.date_range("2000-01-01", "2000-12-01", freq="MS", tz="UTC")
        frame_a = pd.DataFrame({"timestamp": year_2000, "temperature_c": [10.0] * len(year_2000)})
        frame_b = pd.DataFrame({"timestamp": year_2000, "temperature_c": [25.0] * len(year_2000)})

        merged_daily = build_merged_daily_series(
            periods=periods,
            frames_by_period=[frame_a, frame_b],
            report_start=date(2000, 1, 1),
            report_end=date(2000, 12, 31),
            period_baselines=[5.0, 20.0],
        )

        self.assertEqual(len(merged_daily), 366)
        self.assertTrue((merged_daily.iloc[:182]["anomaly_c"].round(2) == 5.0).all())
        self.assertTrue((merged_daily.iloc[182:]["anomaly_c"].round(2) == 5.0).all())
        self.assertEqual(merged_daily.iloc[0]["current_period"], "Period 1: First")
        self.assertEqual(merged_daily.iloc[-1]["current_period"], "Period 2: Second")

        stripe_frame = aggregate_daily_series_to_stripes(merged_daily)
        self.assertEqual(stripe_frame["baseline_c"].round(2).tolist(), [12.54])
        self.assertEqual(stripe_frame["anomaly_c"].round(2).tolist(), [5.0])

    def test_explicit_period_climatology_applies_per_period(self) -> None:
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

        merged_daily = build_merged_daily_series(
            periods=periods,
            frames_by_period=[frame_a, frame_b],
            report_start=date(2000, 1, 1),
            report_end=date(2001, 12, 31),
            period_baselines=[10.0, 20.0],
        )
        stripe_frame = aggregate_daily_series_to_stripes(merged_daily)

        self.assertEqual(stripe_frame["year"].tolist(), [2000, 2001])
        self.assertEqual(stripe_frame["baseline_c"].round(2).tolist(), [10.0, 20.0])
        self.assertEqual(stripe_frame["anomaly_c"].round(2).tolist(), [1.0, 1.0])

    def test_explicit_period_climatology_works_with_rolling_moving_average(self) -> None:
        periods = [
            LifePeriod(
                label="A",
                place_query="A",
                resolved_name="A",
                start_date=date(2020, 1, 1),
                end_date=date(2021, 6, 30),
                latitude=1.0,
                longitude=2.0,
            )
        ]
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", "2021-06-01", freq="MS", tz="UTC"),
                "temperature_c": [11.0] * 18,
            }
        )

        merged_daily = build_merged_daily_series(
            periods=periods,
            frames_by_period=[frame],
            report_start=date(2020, 1, 1),
            report_end=date(2021, 6, 30),
            period_baselines=[10.0],
        )
        stripe_frame = aggregate_daily_series_to_stripes(
            merged_daily,
            aggregation_mode="rolling_365_day",
            rolling_window_end=date(2021, 6, 30),
            rolling_sample_mode="fixed_count",
            rolling_strip_count=3,
        )

        self.assertEqual(len(stripe_frame), 3)
        self.assertEqual(stripe_frame["sample_date"].tolist()[0], date(2020, 12, 30))
        self.assertEqual(stripe_frame["sample_date"].tolist()[-1], date(2021, 6, 30))
        self.assertTrue((stripe_frame["baseline_c"].round(2) == 10.0).all())
        self.assertTrue((stripe_frame["anomaly_c"].round(2) == 1.0).all())

    def test_explicit_period_climatology_rolling_crop_happens_after_prestart_history(self) -> None:
        periods = [
            LifePeriod(
                label="A",
                place_query="A",
                resolved_name="A",
                start_date=date(2020, 1, 1),
                end_date=date(2020, 3, 31),
                latitude=1.0,
                longitude=2.0,
            )
        ]
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2019-01-01", "2020-03-01", freq="MS", tz="UTC"),
                "temperature_c": [11.0] * 15,
            }
        )

        merged_daily = build_merged_daily_series(
            periods=periods,
            frames_by_period=[frame],
            report_start=date(2019, 1, 1),
            report_end=date(2020, 3, 31),
            period_baselines=[10.0],
            first_period_history_days=364,
        )
        stripe_frame = aggregate_daily_series_to_stripes(
            merged_daily,
            aggregation_mode="rolling_365_day",
            rolling_window_end=date(2020, 3, 31),
            rolling_crop_start=date(2020, 1, 1),
        )

        self.assertEqual(stripe_frame["sample_date"].tolist(), [date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31)])
        self.assertTrue((stripe_frame["baseline_c"].round(2) == 10.0).all())
        self.assertTrue((stripe_frame["anomaly_c"].round(2) == 1.0).all())


if __name__ == "__main__":
    unittest.main()
