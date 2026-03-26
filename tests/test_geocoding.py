from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mystripes.geocoding import _result_from_payload, search_places


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

    def test_geoapify_result_uses_bbox_and_formatted_name(self) -> None:
        result = _result_from_payload(
            {
                "formatted": "Sydney, New South Wales, Australia",
                "lat": -33.8698,
                "lon": 151.2083,
                "bbox": {
                    "lon1": 150.5,
                    "lat1": -34.2,
                    "lon2": 151.4,
                    "lat2": -33.4,
                },
            },
            provider="geoapify",
        )

        self.assertEqual(result.display_name, "Sydney, New South Wales, Australia")
        self.assertAlmostEqual(result.latitude, -33.8698)
        self.assertAlmostEqual(result.longitude, 151.2083)
        self.assertEqual(result.bounding_box, (-34.2, -33.4, 150.5, 151.4))
        self.assertEqual(result.coordinate_source, "result center")

    def test_search_places_prefers_geoapify_when_key_is_configured(self) -> None:
        fake_response = Mock()
        fake_response.json.return_value = {
            "results": [
                {
                    "formatted": "Sydney, New South Wales, Australia",
                    "lat": -33.8698,
                    "lon": 151.2083,
                    "bbox": {
                        "lon1": 150.5,
                        "lat1": -34.2,
                        "lon2": 151.4,
                        "lat2": -33.4,
                    },
                }
            ]
        }
        fake_response.raise_for_status.return_value = None

        with patch("mystripes.geocoding.requests.get", return_value=fake_response) as mocked_get:
            results = search_places("Sydney", geoapify_api_key="test-key", cache_dir=None)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].display_name, "Sydney, New South Wales, Australia")
        self.assertIn("api.geoapify.com", mocked_get.call_args.kwargs["url"] if "url" in mocked_get.call_args.kwargs else mocked_get.call_args.args[0])

    def test_search_places_reuses_persistent_cache(self) -> None:
        fake_response = Mock()
        fake_response.json.return_value = {
            "results": [
                {
                    "formatted": "Sydney, New South Wales, Australia",
                    "lat": -33.8698,
                    "lon": 151.2083,
                }
            ]
        }
        fake_response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "geocoding-cache"

            with patch("mystripes.geocoding.requests.get", return_value=fake_response) as mocked_get:
                first = search_places("Sydney", geoapify_api_key="test-key", cache_dir=cache_dir)
                second = search_places("Sydney", geoapify_api_key="test-key", cache_dir=cache_dir)

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(mocked_get.call_count, 1)
        self.assertEqual(first[0].display_name, second[0].display_name)


if __name__ == "__main__":
    unittest.main()
