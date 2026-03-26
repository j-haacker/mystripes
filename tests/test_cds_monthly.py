from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from mystrips.cds import (
    CDSRequestError,
    _aggregate_spatial_selection,
    _dataset_window_from_constraints,
    _request_area,
    parse_temperature_file,
)


class MonthlyCDSTests(unittest.TestCase):
    def test_dataset_window_uses_monthly_temperature_constraints(self) -> None:
        window = _dataset_window_from_constraints(
            [
                {
                    "product_type": ["monthly_averaged_reanalysis"],
                    "variable": ["2m_temperature"],
                    "data_format": ["netcdf"],
                    "year": ["2024", "2025"],
                    "month": ["01", "02"],
                },
                {
                    "product_type": ["monthly_averaged_reanalysis"],
                    "variable": ["skin_temperature"],
                    "data_format": ["netcdf"],
                    "year": ["1900"],
                    "month": ["01"],
                },
            ]
        )

        self.assertEqual(window.min_start, date(2024, 1, 1))
        self.assertEqual(window.max_end, date(2025, 2, 28))

    def test_request_area_snaps_single_cell_to_native_grid(self) -> None:
        area = _request_area(
            latitude=48.2082,
            longitude=16.3738,
            spatial_mode="single_cell",
            radius_km=None,
            boundary_geojson=None,
            boundary_bbox=None,
        )

        self.assertEqual(area, (48.2, 16.4, 48.2, 16.4))

    def test_radius_aggregation_averages_cells_inside_radius(self) -> None:
        timestamp = pd.Timestamp("2020-01-01T00:00:00Z")
        grid_frame = pd.DataFrame(
            {
                "timestamp": [timestamp, timestamp, timestamp],
                "temperature_c": [10.0, 14.0, 50.0],
                "sample_days": [31, 31, 31],
                "grid_latitude": [0.0, 0.0, 1.0],
                "grid_longitude": [0.0, 0.1, 1.0],
            }
        )

        aggregated = _aggregate_spatial_selection(
            grid_frame=grid_frame,
            latitude=0.0,
            longitude=0.0,
            spatial_mode="radius",
            radius_km=15.0,
            boundary_geojson=None,
            boundary_bbox=None,
        )

        self.assertEqual(aggregated["temperature_c"].tolist(), [12.0])
        self.assertEqual(aggregated["sample_days"].tolist(), [31])

    def test_boundary_aggregation_uses_polygon_cells(self) -> None:
        timestamp = pd.Timestamp("2020-01-01T00:00:00Z")
        grid_frame = pd.DataFrame(
            {
                "timestamp": [timestamp, timestamp, timestamp],
                "temperature_c": [10.0, 20.0, 50.0],
                "sample_days": [31, 31, 31],
                "grid_latitude": [0.0, 0.0, 0.2],
                "grid_longitude": [0.0, 0.1, 0.2],
            }
        )

        aggregated = _aggregate_spatial_selection(
            grid_frame=grid_frame,
            latitude=0.0,
            longitude=0.0,
            spatial_mode="boundary",
            radius_km=None,
            boundary_geojson={
                "type": "Polygon",
                "coordinates": [
                    [
                        [-0.05, -0.05],
                        [0.15, -0.05],
                        [0.15, 0.05],
                        [-0.05, 0.05],
                        [-0.05, -0.05],
                    ]
                ],
            },
            boundary_bbox=(-0.05, 0.05, -0.05, 0.15),
        )

        self.assertEqual(aggregated["temperature_c"].tolist(), [15.0])

    def test_parse_temperature_file_reports_unexpected_binary_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "response.bin"
            path.write_bytes(b"BINARY\xc7\x00payload")

            with self.assertRaises(CDSRequestError) as context:
                parse_temperature_file(path, target_latitude=0.0, target_longitude=0.0)

        self.assertIn("binary file", str(context.exception))


if __name__ == "__main__":
    unittest.main()
