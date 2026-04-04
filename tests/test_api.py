from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd
from matplotlib.colors import to_hex

from mystripes.api import build_period_indicator_specs, build_stripe_data, plot_stripes


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

    def test_build_period_indicator_specs_maps_periods_to_stripe_spans(self) -> None:
        stripe_frame = pd.DataFrame(
            {
                "window_start": [date(2000, 1, 1), date(2001, 1, 1), date(2002, 1, 1), date(2003, 1, 1)],
                "window_end": [date(2000, 12, 31), date(2001, 12, 31), date(2002, 12, 31), date(2003, 12, 31)],
                "anomaly_c": [-1.0, -0.5, 0.5, 1.0],
            }
        )

        specs = build_period_indicator_specs(
            periods=[
                {
                    "label": "Home",
                    "start_date": date(2000, 1, 1),
                    "end_date": date(2001, 12, 31),
                    "latitude": 48.2082,
                    "longitude": 16.3738,
                },
                {
                    "label": "Abroad",
                    "start_date": date(2002, 1, 1),
                    "end_date": date(2003, 12, 31),
                    "latitude": 52.52,
                    "longitude": 13.405,
                },
            ],
            stripe_frame=stripe_frame,
            included_period_indices=[0, 1],
        )

        self.assertEqual([spec["label"] for spec in specs], ["Home", "Abroad"])
        self.assertAlmostEqual(float(specs[0]["start_fraction"]), 0.0, places=6)
        self.assertAlmostEqual(float(specs[0]["end_fraction"]), 0.5, places=6)
        self.assertAlmostEqual(float(specs[1]["start_fraction"]), 0.5, places=6)
        self.assertAlmostEqual(float(specs[1]["end_fraction"]), 1.0, places=6)

    def test_plot_stripes_supports_period_indicator_styles(self) -> None:
        stripe_data = {
            "stripe_frame": pd.DataFrame(
                {
                    "year": [2000, 2001, 2002, 2003],
                    "anomaly_c": [-1.0, -0.4, 0.3, 0.9],
                }
            )
        }
        period_indicators = [
            {"label": "Home", "start_fraction": 0.0, "end_fraction": 0.5},
            {"label": "Abroad", "start_fraction": 0.5, "end_fraction": 1.0},
        ]

        figure = plot_stripes(
            stripe_data,
            width_px=600,
            height_px=120,
            dpi=100,
            period_indicators=period_indicators,
            period_indicator_style="scale_bar",
            period_indicator_vertical_align="top",
            period_indicator_color="#ffeeaa",
        )

        axis = figure.axes[0]
        self.assertEqual([text.get_text() for text in axis.texts], ["Home", "Abroad"])
        self.assertGreaterEqual(len(axis.lines), 6)
        self.assertGreater(float(axis.lines[1].get_xdata()[0]), float(axis.lines[0].get_xdata()[0]))
        self.assertLess(float(axis.lines[2].get_xdata()[0]), float(axis.lines[0].get_xdata()[1]))
        self.assertEqual(len(axis.patches), 0)
        figure.clf()

        figure = plot_stripes(
            stripe_data,
            width_px=600,
            height_px=120,
            dpi=100,
            period_indicators=period_indicators,
            period_indicator_style="scale_bar",
            period_indicator_vertical_align="bottom",
            period_indicator_color="#ffeeaa",
        )

        axis = figure.axes[0]
        scale_bar_bottom_label_y = float(axis.texts[0].get_position()[1])
        scale_bar_bottom_line_y = float(axis.lines[0].get_ydata()[0])
        self.assertGreater(scale_bar_bottom_label_y, scale_bar_bottom_line_y)
        figure.clf()

        figure = plot_stripes(
            stripe_data,
            width_px=600,
            height_px=120,
            dpi=100,
            period_indicators=period_indicators,
            period_indicator_style="outward_arrows",
            period_indicator_vertical_align="bottom",
            period_indicator_color="#ffeeaa",
        )

        axis = figure.axes[0]
        self.assertEqual([text.get_text() for text in axis.texts], ["Home", "Abroad"])
        self.assertAlmostEqual(float(axis.texts[0].get_position()[1]), scale_bar_bottom_label_y, places=6)
        self.assertEqual(len(axis.lines), 0)
        self.assertEqual(len(axis.patches), 2)
        self.assertEqual(type(axis.patches[0]).__name__, "Polygon")
        vertices = axis.patches[0].get_xy()
        self.assertAlmostEqual(float(vertices[0][0]), 0.0, places=6)
        self.assertGreater(float(vertices[5][0]), 0.498)
        self.assertLess(float(vertices[5][0]), 0.5)
        self.assertEqual(to_hex(axis.patches[0].get_edgecolor(), keep_alpha=False), "#000000")
        self.assertAlmostEqual(float(axis.patches[0].get_linewidth()), 0.266667, places=5)
        figure.clf()

    def test_plot_stripes_supports_centered_period_indicators_and_height_scaling(self) -> None:
        stripe_data = {
            "stripe_frame": pd.DataFrame(
                {
                    "year": [2000, 2001, 2002, 2003],
                    "anomaly_c": [-1.0, -0.4, 0.3, 0.9],
                }
            )
        }
        period_indicators = [
            {"label": "Home", "start_fraction": 0.0, "end_fraction": 0.5},
        ]

        figure = plot_stripes(
            stripe_data,
            width_px=600,
            height_px=160,
            dpi=100,
            period_indicators=period_indicators,
            period_indicator_style="scale_bar",
            period_indicator_vertical_align="top",
            period_indicator_height_ratio=0.2,
        )
        axis = figure.axes[0]
        top_label_y = float(axis.texts[0].get_position()[1])
        top_line_y = float(axis.lines[0].get_ydata()[0])
        top_tick_y = [float(value) for value in axis.lines[1].get_ydata()]
        self.assertLess(top_label_y, top_line_y)
        self.assertGreater(top_label_y, min(top_tick_y))
        self.assertEqual(max(top_tick_y), top_line_y)
        self.assertLess(min(top_tick_y), top_line_y)
        figure.clf()

        figure = plot_stripes(
            stripe_data,
            width_px=600,
            height_px=160,
            dpi=100,
            period_indicators=period_indicators,
            period_indicator_style="scale_bar",
            period_indicator_vertical_align="center",
            period_indicator_height_ratio=0.2,
        )
        axis = figure.axes[0]
        center_label_y = float(axis.texts[0].get_position()[1])
        center_line_y = float(axis.lines[0].get_ydata()[0])
        center_tick_y = [float(value) for value in axis.lines[1].get_ydata()]
        self.assertGreater(center_label_y, center_line_y)
        self.assertLess(min(center_tick_y), center_line_y)
        self.assertGreater(max(center_tick_y), center_line_y)
        figure.clf()

        figure = plot_stripes(
            stripe_data,
            width_px=600,
            height_px=160,
            dpi=100,
            period_indicators=period_indicators,
            period_indicator_style="scale_bar",
            period_indicator_vertical_align="bottom",
            period_indicator_height_ratio=0.15,
        )
        axis = figure.axes[0]
        small_label_y = float(axis.texts[0].get_position()[1])
        small_line_y = float(axis.lines[0].get_ydata()[0])
        small_tick_y = [float(value) for value in axis.lines[1].get_ydata()]
        small_tick_height = abs(float(axis.lines[1].get_ydata()[1]) - float(axis.lines[1].get_ydata()[0]))
        self.assertGreater(small_label_y, small_line_y)
        self.assertLess(small_label_y, max(small_tick_y))
        figure.clf()

        figure = plot_stripes(
            stripe_data,
            width_px=600,
            height_px=160,
            dpi=100,
            period_indicators=period_indicators,
            period_indicator_style="scale_bar",
            period_indicator_vertical_align="bottom",
            period_indicator_height_ratio=0.45,
        )
        axis = figure.axes[0]
        large_tick_height = abs(float(axis.lines[1].get_ydata()[1]) - float(axis.lines[1].get_ydata()[0]))
        self.assertGreater(large_tick_height, small_tick_height)
        figure.clf()

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
