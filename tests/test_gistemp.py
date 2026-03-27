from __future__ import annotations

import unittest
from datetime import date

from mystripes.gistemp import average_global_warming_for_period, parse_global_mean_estimates


class GISTEMPTests(unittest.TestCase):
    def test_parse_global_mean_estimates_extracts_annual_rows(self) -> None:
        frame = parse_global_mean_estimates(
            """
Land-Ocean Temperature Index (C)
--------------------------------
Year No_Smoothing Lowess(5)
----------------------------
1999 0.38 0.47
2000 0.39 0.50
2001 0.53 0.52
"""
        )

        self.assertEqual(frame["year"].tolist(), [1999, 2000, 2001])
        self.assertEqual(frame["anomaly_c"].round(2).tolist(), [0.38, 0.39, 0.53])
        self.assertEqual(frame["lowess_5y_c"].round(2).tolist(), [0.47, 0.50, 0.52])

    def test_average_global_warming_for_period_uses_overlapping_years(self) -> None:
        frame = parse_global_mean_estimates(
            """
1999 0.38 0.47
2000 0.39 0.50
2001 0.53 0.52
2002 0.62 0.54
"""
        )

        mean_anomaly_c, start_year, end_year = average_global_warming_for_period(
            frame,
            period_start=date(2000, 6, 1),
            period_end=date(2001, 2, 1),
        )

        self.assertAlmostEqual(mean_anomaly_c, 0.46, places=6)
        self.assertEqual(start_year, 2000)
        self.assertEqual(end_year, 2001)

    def test_average_global_warming_for_period_clips_to_available_years(self) -> None:
        frame = parse_global_mean_estimates(
            """
2000 0.39 0.50
2001 0.53 0.52
"""
        )

        mean_anomaly_c, start_year, end_year = average_global_warming_for_period(
            frame,
            period_start=date(1990, 1, 1),
            period_end=date(2020, 12, 31),
        )

        self.assertAlmostEqual(mean_anomaly_c, 0.46, places=6)
        self.assertEqual(start_year, 2000)
        self.assertEqual(end_year, 2001)


if __name__ == "__main__":
    unittest.main()
