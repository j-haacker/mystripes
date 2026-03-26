from __future__ import annotations

import unittest

try:
    from fastapi.testclient import TestClient

    from api import app
except ModuleNotFoundError:
    TestClient = None
    app = None


@unittest.skipIf(TestClient is None or app is None, "fastapi test dependencies are not installed")
class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert TestClient is not None
        assert app is not None
        cls.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_palette_endpoint(self) -> None:
        response = self.client.get("/v1/palette")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["colors"]), 16)

    def test_render_svg_endpoint(self) -> None:
        response = self.client.post(
            "/v1/render",
            json={
                "anomalies": [-1.0, 0.0, 1.0],
                "width_px": 600,
                "height_px": 100,
                "dpi": 100,
                "format": "svg",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/svg+xml")
        self.assertIn("<svg", response.text)


if __name__ == "__main__":
    unittest.main()
