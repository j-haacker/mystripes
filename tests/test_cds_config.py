from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mystrips.cds import (
    DEFAULT_CDSAPI_URL,
    CDSCredentialsMissingError,
    clear_local_cds_config,
    load_local_cds_config,
    resolve_cds_config,
    save_local_cds_config,
)


class CDSConfigTests(unittest.TestCase):
    def test_save_and_load_local_credentials_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "local_cds_credentials.toml"
            save_local_cds_config(
                key="personal-access-token",
                url="https://example.test/api",
                path=path,
            )

            config = load_local_cds_config(path)

            self.assertIsNotNone(config)
            assert config is not None
            self.assertEqual(config.key, "personal-access-token")
            self.assertEqual(config.url, "https://example.test/api")
            self.assertEqual(config.source, "local_file")

    def test_resolve_prefers_streamlit_secrets_over_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {}, clear=True):
            path = Path(tmpdir) / "local_cds_credentials.toml"
            save_local_cds_config(key="local-token", path=path)

            config = resolve_cds_config(
                secret_values={
                    "CDSAPI_KEY": "secret-token",
                    "CDSAPI_URL": "https://secrets.example/api",
                },
                local_credentials_path=path,
            )

            self.assertEqual(config.key, "secret-token")
            self.assertEqual(config.url, "https://secrets.example/api")
            self.assertEqual(config.source, "streamlit_secrets")

    def test_resolve_prefers_environment_over_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "CDSAPI_KEY": "env-token",
                "CDSAPI_URL": "https://env.example/api",
            },
            clear=True,
        ):
            path = Path(tmpdir) / "local_cds_credentials.toml"
            save_local_cds_config(key="local-token", path=path)

            config = resolve_cds_config(secret_values={}, local_credentials_path=path)

            self.assertEqual(config.key, "env-token")
            self.assertEqual(config.url, "https://env.example/api")
            self.assertEqual(config.source, "environment")

    def test_resolve_uses_local_file_when_other_sources_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {}, clear=True):
            path = Path(tmpdir) / "local_cds_credentials.toml"
            save_local_cds_config(key="local-token", path=path)

            config = resolve_cds_config(secret_values={}, local_credentials_path=path)

            self.assertEqual(config.key, "local-token")
            self.assertEqual(config.url, DEFAULT_CDSAPI_URL)
            self.assertEqual(config.source, "local_file")

    def test_clear_local_credentials_removes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "local_cds_credentials.toml"
            save_local_cds_config(key="local-token", path=path)

            clear_local_cds_config(path)

            self.assertIsNone(load_local_cds_config(path))

    def test_resolve_raises_when_no_credentials_are_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {}, clear=True):
            path = Path(tmpdir) / "local_cds_credentials.toml"

            with self.assertRaises(CDSCredentialsMissingError):
                resolve_cds_config(secret_values={}, local_credentials_path=path)


if __name__ == "__main__":
    unittest.main()
