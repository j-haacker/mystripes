from __future__ import annotations

import unittest
from datetime import date

from mystripes.refresh import latest_data_end_that_can_change_stripes


class RefreshPolicyTests(unittest.TestCase):
    def test_rolling_mode_always_uses_latest_available_date(self) -> None:
        self.assertEqual(
            latest_data_end_that_can_change_stripes(
                report_start=date(1990, 1, 1),
                analysis_end=date(2026, 3, 27),
                aggregation_mode="rolling_365_day",
                reference_period_mode="climate_normal_1961_2010",
            ),
            date(2026, 3, 27),
        )

    def test_full_calendar_years_with_fixed_reference_uses_last_complete_year(self) -> None:
        self.assertEqual(
            latest_data_end_that_can_change_stripes(
                report_start=date(1990, 1, 1),
                analysis_end=date(2026, 3, 27),
                aggregation_mode="full_calendar_years",
                reference_period_mode="climate_normal_1961_2010",
            ),
            date(2025, 12, 31),
        )

    def test_full_calendar_years_with_storyline_reference_keeps_latest_date(self) -> None:
        self.assertEqual(
            latest_data_end_that_can_change_stripes(
                report_start=date(1990, 1, 1),
                analysis_end=date(2026, 3, 27),
                aggregation_mode="full_calendar_years",
                reference_period_mode="story_line_period",
            ),
            date(2026, 3, 27),
        )

    def test_full_calendar_years_keeps_latest_date_when_no_complete_year_exists(self) -> None:
        self.assertEqual(
            latest_data_end_that_can_change_stripes(
                report_start=date(2026, 1, 1),
                analysis_end=date(2026, 3, 27),
                aggregation_mode="full_calendar_years",
                reference_period_mode="climate_normal_1961_2010",
            ),
            date(2026, 3, 27),
        )


if __name__ == "__main__":
    unittest.main()
