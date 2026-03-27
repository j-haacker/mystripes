from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from mystripes.storylines import (
    STORYLINE_COOKIE_PREFIX,
    build_cookie_sync_html,
    decode_storyline_cookie_value,
    encode_storyline_cookie_value,
    load_cookie_storylines,
    load_local_storylines,
    remove_local_storyline,
    save_local_storyline,
    serialize_storyline_payload,
    storyline_cookie_name,
    storyline_storage_backend_from_host,
)


class StorylinePersistenceTests(unittest.TestCase):
    def _sample_payload(self, *, include_boundary_geojson: bool = True):
        return serialize_storyline_payload(
            name="My life",
            birth_date=date(1990, 1, 1),
            period_entries=[
                {
                    "place_query": "Vienna, Austria",
                    "resolved_name": "Vienna, Austria",
                    "latitude_text": "48.2082",
                    "longitude_text": "16.3738",
                    "coordinate_source": "geoapify",
                    "boundary_geojson": {"type": "Point", "coordinates": [16.3738, 48.2082]},
                    "bounding_box": (16.2, 48.1, 16.5, 48.3),
                    "end_date": date(2010, 12, 31),
                },
                {
                    "place_query": "Berlin, Germany",
                    "resolved_name": "Berlin, Germany",
                    "latitude_text": "52.5200",
                    "longitude_text": "13.4050",
                    "coordinate_source": "geoapify",
                    "bounding_box": (13.0, 52.3, 13.8, 52.7),
                    "end_date": None,
                },
            ],
            include_boundary_geojson=include_boundary_geojson,
        )

    def test_storage_backend_from_host_detects_streamlit_cloud(self) -> None:
        self.assertEqual(
            storyline_storage_backend_from_host("mystripes-demo.streamlit.app"),
            "cookie",
        )
        self.assertEqual(storyline_storage_backend_from_host("localhost:8501"), "file")

    def test_local_storylines_round_trip(self) -> None:
        payload = self._sample_payload()
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "storylines.json"
            save_local_storyline(payload, path=path)
            loaded = load_local_storylines(path)

        self.assertEqual(list(loaded), ["My life"])
        storyline = loaded["My life"]
        self.assertEqual(storyline["birth_date"], date(1990, 1, 1))
        self.assertEqual(storyline["period_entries"][0]["end_date"], date(2010, 12, 31))
        self.assertEqual(storyline["period_entries"][0]["bounding_box"], (16.2, 48.1, 16.5, 48.3))
        self.assertEqual(storyline["period_entries"][0]["boundary_geojson"], {"type": "Point", "coordinates": [16.3738, 48.2082]})

    def test_remove_local_storyline_deletes_last_file(self) -> None:
        payload = self._sample_payload()
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "storylines.json"
            save_local_storyline(payload, path=path)
            removed = remove_local_storyline("My life", path=path)

            self.assertTrue(removed)
            self.assertFalse(path.exists())

    def test_cookie_storyline_round_trip(self) -> None:
        payload = self._sample_payload(include_boundary_geojson=False)
        encoded = encode_storyline_cookie_value(payload)
        decoded = decode_storyline_cookie_value(encoded)

        self.assertEqual(decoded["name"], "My life")
        self.assertEqual(decoded["birth_date"], date(1990, 1, 1))
        self.assertEqual(decoded["period_entries"][0]["end_date"], date(2010, 12, 31))
        self.assertIsNone(decoded["period_entries"][0]["boundary_geojson"])

    def test_load_cookie_storylines_ignores_invalid_cookie_values(self) -> None:
        payload = self._sample_payload(include_boundary_geojson=False)
        cookie_name = storyline_cookie_name(payload["name"])
        encoded = encode_storyline_cookie_value(payload)
        loaded = load_cookie_storylines(
            {
                cookie_name: encoded,
                f"{STORYLINE_COOKIE_PREFIX}broken": "not-a-valid-cookie",
                "another_cookie": "ignored",
            }
        )

        self.assertEqual(list(loaded), ["My life"])

    def test_build_cookie_sync_html_includes_reload_and_assignment(self) -> None:
        html = build_cookie_sync_html("test_cookie", "abc123")
        self.assertIn("test_cookie=abc123", html)
        self.assertIn("window.top.location.reload()", html)

        delete_html = build_cookie_sync_html("test_cookie", None)
        self.assertIn("Expires=Thu, 01 Jan 1970 00:00:00 GMT", delete_html)


if __name__ == "__main__":
    unittest.main()
