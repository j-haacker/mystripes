from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from mystripes.api import build_stripe_data, plot_stripes


class PublicAPITests(unittest.TestCase):
    def test_build_stripe_data_accepts_period_dicts_and_frame_list(self) -> None:
        stripe_data = build_stripe_data(
            periods=[
                {
                    "label": "Vienna",
                    "start_date": date(2000, 1, 1),
                    "end_date": date(2000, 12, 31),
                    "latitude": 48.2082,
                    "longitude": 16.3738,
                }
            ],
            period_data=[
                pd.DataFrame(
                    {
                        "date": pd.date_range("2000-01-01", "2000-12-01", freq="MS", tz="UTC"),
                        "temperature": [11.0] * 12,
                    }
                )
            ],
        )

        self.assertIn("stripe_frame", stripe_data)
        self.assertEqual(stripe_data["stripe_frame"]["year"].tolist(), [2000])
        self.assertIn("anomaly_c", stripe_data["stripe_frame"].columns)

    def test_build_stripe_data_can_use_location_reference_from_input_data(self) -> None:
        stripe_data = build_stripe_data(
            periods=[
                {
                    "label": "A",
                    "start_date": date(2000, 1, 1),
                    "end_date": date(2000, 12, 31),
                    "latitude": 1.0,
                    "longitude": 2.0,
                },
                {
                    "label": "B",
                    "start_date": date(2001, 1, 1),
                    "end_date": date(2001, 12, 31),
                    "latitude": 3.0,
                    "longitude": 4.0,
                },
            ],
            period_data=[
                pd.DataFrame(
                    {
                        "timestamp": pd.date_range("2000-01-01", "2000-12-01", freq="MS", tz="UTC"),
                        "temperature_c": [11.0] * 12,
                    }
                ),
                pd.DataFrame(
                    {
                        "timestamp": pd.date_range("2001-01-01", "2001-12-01", freq="MS", tz="UTC"),
                        "temperature_c": [21.0] * 12,
                    }
                ),
            ],
            baseline_mode="location_reference",
            baseline_by_location={"A": 10.0, "B": 20.0},
        )

        self.assertEqual(stripe_data["stripe_frame"]["anomaly_c"].round(2).tolist(), [1.0, 1.0])

    def test_plot_stripes_can_save_svg(self) -> None:
        stripe_data = {
            "stripe_frame": pd.DataFrame(
                {
                    "year": [2000, 2001, 2002],
                    "anomaly_c": [-1.0, 0.0, 1.0],
                }
            )
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "mystripes.svg"
            figure = plot_stripes(stripe_data, output_path=output_path, width_px=600, height_px=100, dpi=100)

            self.assertTrue(output_path.exists())
            self.assertIn("<svg", output_path.read_text(encoding="utf-8"))
            figure.clf()


if __name__ == "__main__":
    unittest.main()
