from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd
from matplotlib.colors import to_hex

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

    def test_build_stripe_data_uses_day_of_year_climatology(self) -> None:
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
        )

        self.assertEqual(stripe_data["stripe_frame"]["baseline_c"].round(2).tolist(), [11.0, 21.0])
        self.assertEqual(stripe_data["stripe_frame"]["anomaly_c"].round(2).tolist(), [0.0, 0.0])

    def test_build_stripe_data_supports_rolling_moving_average_options(self) -> None:
        stripe_data = build_stripe_data(
            periods=[
                {
                    "label": "Vienna",
                    "start_date": date(2020, 1, 1),
                    "end_date": date(2021, 6, 30),
                    "latitude": 48.2082,
                    "longitude": 16.3738,
                }
            ],
            period_data=[
                pd.DataFrame(
                    {
                        "date": pd.date_range("2020-01-01", "2021-06-01", freq="MS", tz="UTC"),
                        "temperature": [11.0] * 18,
                    }
                )
            ],
            aggregation_mode="rolling_365_day",
            rolling_sample_mode="fixed_count",
            rolling_strip_count=3,
            rolling_window_end=date(2021, 6, 30),
        )

        self.assertEqual(len(stripe_data["stripe_frame"]), 3)
        self.assertEqual(stripe_data["stripe_frame"]["sample_date"].tolist()[-1], date(2021, 6, 30))

    def test_build_stripe_data_accepts_separate_baseline_period_data(self) -> None:
        stripe_data = build_stripe_data(
            periods=[
                {
                    "label": "A",
                    "start_date": date(2020, 1, 1),
                    "end_date": date(2020, 12, 31),
                    "latitude": 1.0,
                    "longitude": 2.0,
                }
            ],
            period_data=[
                pd.DataFrame(
                    {
                        "timestamp": pd.date_range("2020-01-01", "2020-12-01", freq="MS", tz="UTC"),
                        "temperature_c": [11.0] * 12,
                    }
                )
            ],
            baseline_period_data=[
                pd.DataFrame(
                    {
                        "timestamp": pd.date_range("2000-01-01", "2000-12-01", freq="MS", tz="UTC"),
                        "temperature_c": [10.0] * 12,
                    }
                )
            ],
            baseline_start=date(2000, 1, 1),
            baseline_end=date(2000, 12, 31),
        )

        self.assertEqual(stripe_data["baseline_start"], date(2000, 1, 1))
        self.assertEqual(stripe_data["baseline_end"], date(2000, 12, 31))
        self.assertEqual(stripe_data["stripe_frame"]["baseline_c"].round(2).tolist(), [10.0])
        self.assertEqual(stripe_data["stripe_frame"]["anomaly_c"].round(2).tolist(), [1.0])

    def test_plot_stripes_supports_fitted_watermark(self) -> None:
        stripe_data = {
            "stripe_frame": pd.DataFrame(
                {
                    "year": [2000, 2001, 2002],
                    "anomaly_c": [-1.0, 0.0, 1.0],
                }
            )
        }

        figure = plot_stripes(
            stripe_data,
            width_px=600,
            height_px=100,
            dpi=100,
            watermark_text="TEST",
            watermark_horizontal_align="right",
            watermark_vertical_align="bottom",
            watermark_color="#ff0000",
            watermark_opacity=0.25,
            watermark_max_width_ratio=0.5,
            watermark_max_height_ratio=0.6,
        )

        axis = figure.axes[0]
        self.assertEqual(len(axis.texts), 1)
        watermark = axis.texts[0]
        self.assertEqual(watermark.get_text(), "TEST")
        self.assertEqual(watermark.get_horizontalalignment(), "right")
        self.assertEqual(watermark.get_verticalalignment(), "bottom")
        self.assertEqual(to_hex(watermark.get_color()), "#ff0000")
        self.assertAlmostEqual(float(watermark.get_alpha()), 0.25, places=6)
        self.assertAlmostEqual(float(watermark.get_position()[0]), 0.98, places=6)
        self.assertAlmostEqual(float(watermark.get_position()[1]), 0.02, places=6)

        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        axis_box = axis.get_window_extent(renderer=renderer)
        text_box = watermark.get_window_extent(renderer=renderer)
        self.assertLessEqual(text_box.width, (axis_box.width * 0.5) + 1.0)
        self.assertLessEqual(text_box.height, (axis_box.height * 0.6) + 1.0)
        figure.clf()

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
