from __future__ import annotations

import unittest

from personal_warming_stripes.geocoding import _result_from_payload


class GeocodingTests(unittest.TestCase):
    def test_result_uses_point_coordinates_when_available(self) -> None:
        result = _result_from_payload(
            {
                "display_name": "Vienna, Austria",
                "lat": "48.2082",
                "lon": "16.3738",
                "geojson": {"type": "Point", "coordinates": [16.3738, 48.2082]},
            }
        )

        self.assertAlmostEqual(result.latitude, 48.2082)
        self.assertAlmostEqual(result.longitude, 16.3738)
        self.assertEqual(result.coordinate_source, "point geometry")

    def test_result_uses_area_centroid_for_polygons(self) -> None:
        result = _result_from_payload(
            {
                "display_name": "Test Region",
                "geojson": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [0.0, 0.0],
                            [4.0, 0.0],
                            [4.0, 2.0],
                            [0.0, 2.0],
                            [0.0, 0.0],
                        ]
                    ],
                },
            }
        )

        self.assertAlmostEqual(result.latitude, 1.0)
        self.assertAlmostEqual(result.longitude, 2.0)
        self.assertEqual(result.coordinate_source, "area centroid")

    def test_result_falls_back_to_bounding_box_center(self) -> None:
        result = _result_from_payload(
            {
                "display_name": "Tyrol, Austria",
                "boundingbox": ["46.6", "47.8", "10.1", "12.9"],
            }
        )

        self.assertAlmostEqual(result.latitude, 47.2)
        self.assertAlmostEqual(result.longitude, 11.5)
        self.assertEqual(result.coordinate_source, "bounding box center")


if __name__ == "__main__":
    unittest.main()
