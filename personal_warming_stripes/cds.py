from __future__ import annotations

import io
import os
import tempfile
import tomllib
from collections.abc import Mapping
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from personal_warming_stripes.models import CDSConfig, DatasetWindow

DATASET_NAME = "reanalysis-era5-land-timeseries"
FORM_URL = (
    "https://cds.climate.copernicus.eu/api/catalogue/v1/collections/"
    f"{DATASET_NAME}/form.json"
)
DEFAULT_CDSAPI_URL = "https://cds.climate.copernicus.eu/api"
LOCAL_CDS_CREDENTIALS_PATH = Path(".streamlit/local_cds_credentials.toml")


class CDSCredentialsMissingError(RuntimeError):
    """Raised when the app operator has not configured CDS credentials."""


class CDSRequestError(RuntimeError):
    """Raised when the CDS request cannot be completed."""


def get_dataset_window() -> DatasetWindow:
    response = requests.get(FORM_URL, timeout=30)
    response.raise_for_status()

    for field in response.json():
        if field.get("name") == "date":
            details = field["details"]
            return DatasetWindow(
                min_start=date.fromisoformat(details["minStart"]),
                max_end=date.fromisoformat(details["maxEnd"]),
            )

    raise CDSRequestError("Could not determine the ERA5-Land date window from CDS.")


def resolve_cds_config(
    secret_values: Mapping[str, str] | None = None,
    local_credentials_path: Path = LOCAL_CDS_CREDENTIALS_PATH,
) -> CDSConfig:
    secret_values = secret_values or {}

    secret_key = str(secret_values.get("CDSAPI_KEY", "")).strip()
    if secret_key:
        secret_url = str(secret_values.get("CDSAPI_URL", "")).strip() or DEFAULT_CDSAPI_URL
        return CDSConfig(url=secret_url, key=secret_key, source="streamlit_secrets")

    env_key = os.getenv("CDSAPI_KEY", "").strip()
    if env_key:
        env_url = os.getenv("CDSAPI_URL", "").strip() or DEFAULT_CDSAPI_URL
        return CDSConfig(url=env_url, key=env_key, source="environment")

    local_config = load_local_cds_config(local_credentials_path)
    if local_config is not None:
        return local_config

    raise CDSCredentialsMissingError(
        "Missing CDSAPI_KEY. Add it to Streamlit secrets, environment variables, or save "
        "it locally from the app sidebar."
    )


def load_local_cds_config(path: Path = LOCAL_CDS_CREDENTIALS_PATH) -> CDSConfig | None:
    if not path.exists():
        return None

    with path.open("rb") as handle:
        payload = tomllib.load(handle)

    key = str(payload.get("CDSAPI_KEY", "")).strip()
    if not key:
        return None

    url = str(payload.get("CDSAPI_URL", "")).strip() or DEFAULT_CDSAPI_URL
    return CDSConfig(url=url, key=key, source="local_file")


def save_local_cds_config(
    key: str,
    url: str = DEFAULT_CDSAPI_URL,
    path: Path = LOCAL_CDS_CREDENTIALS_PATH,
) -> None:
    normalized_key = key.strip()
    normalized_url = url.strip() or DEFAULT_CDSAPI_URL
    if not normalized_key:
        raise CDSCredentialsMissingError("A CDS API token is required before it can be saved.")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        f'CDSAPI_URL = "{_escape_toml_string(normalized_url)}"\n'
        f'CDSAPI_KEY = "{_escape_toml_string(normalized_key)}"\n'
    )
    path.write_text(payload, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except PermissionError:
        pass


def clear_local_cds_config(path: Path = LOCAL_CDS_CREDENTIALS_PATH) -> None:
    if path.exists():
        path.unlink()


def fetch_point_temperature_series(
    config: CDSConfig,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    if start_date > end_date:
        raise ValueError("Start date must be on or before end date.")

    try:
        import cdsapi
    except ModuleNotFoundError as exc:
        raise CDSRequestError(
            "cdsapi is not installed. Install the dependencies from requirements.txt."
        ) from exc

    client = cdsapi.Client(url=config.url, key=config.key, quiet=True, progress=False)
    last_error: Exception | None = None

    for location_value in _candidate_location_values(latitude, longitude):
        request = {
            "variable": ["2m_temperature"],
            "location": location_value,
            "date": f"{start_date.isoformat()}/{end_date.isoformat()}",
            "data_format": "csv",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "era5_land_timeseries.csv"
            try:
                client.retrieve(DATASET_NAME, request, str(target))
            except Exception as exc:  # pragma: no cover - depends on external service.
                last_error = exc
                continue

            return parse_temperature_csv(target.read_text(encoding="utf-8"))

    message = "ERA5-Land CDS request failed."
    if last_error is not None:
        message = f"{message} Last error: {last_error}"
    raise CDSRequestError(message)


def parse_temperature_csv(raw_csv: str) -> pd.DataFrame:
    if not raw_csv.strip():
        raise CDSRequestError("The CDS response was empty.")

    frame = pd.read_csv(io.StringIO(raw_csv), sep=None, engine="python")
    frame.columns = [str(column).strip() for column in frame.columns]

    timestamp = _extract_timestamp(frame)
    value_column = _resolve_temperature_column(frame)

    parsed = pd.DataFrame(
        {
            "timestamp": timestamp,
            "temperature_c": pd.to_numeric(frame[value_column], errors="coerce") - 273.15,
        }
    )
    parsed = parsed.dropna().sort_values("timestamp").reset_index(drop=True)
    if parsed.empty:
        raise CDSRequestError("The CDS response did not contain usable temperature rows.")
    return parsed


def _candidate_location_values(latitude: float, longitude: float) -> list[object]:
    lat = round(latitude, 4)
    lon = round(longitude, 4)
    return [
        {"latitude": lat, "longitude": lon},
        {"lat": lat, "lon": lon},
        [lat, lon],
    ]


def _extract_timestamp(frame: pd.DataFrame) -> pd.Series:
    lower_names = {column.lower(): column for column in frame.columns}

    for candidate in ("valid_time", "datetime", "timestamp", "time"):
        if candidate in lower_names:
            return pd.to_datetime(frame[lower_names[candidate]], utc=True, errors="coerce")

    if "date" in lower_names and "time" in lower_names:
        combined = frame[lower_names["date"]].astype(str) + " " + frame[lower_names["time"]].astype(str)
        return pd.to_datetime(combined, utc=True, errors="coerce")

    if "date" in lower_names:
        return pd.to_datetime(frame[lower_names["date"]], utc=True, errors="coerce")

    for column in frame.columns:
        lowered = column.lower()
        if "time" in lowered or "date" in lowered:
            parsed = pd.to_datetime(frame[column], utc=True, errors="coerce")
            if parsed.notna().any():
                return parsed

    raise CDSRequestError("Could not identify a timestamp column in the CDS CSV.")


def _resolve_temperature_column(frame: pd.DataFrame) -> str:
    preferred = ("2m_temperature", "t2m", "temperature", "value")
    lower_names = {column.lower(): column for column in frame.columns}

    for candidate in preferred:
        if candidate in lower_names:
            return lower_names[candidate]

    excluded = {
        "latitude",
        "longitude",
        "lat",
        "lon",
        "valid_time",
        "datetime",
        "timestamp",
        "time",
        "date",
    }
    numeric_candidates: list[str] = []
    for column in frame.columns:
        if column.lower() in excluded:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().any():
            numeric_candidates.append(column)

    if len(numeric_candidates) == 1:
        return numeric_candidates[0]

    for column in numeric_candidates:
        if "temp" in column.lower():
            return column

    raise CDSRequestError("Could not identify the temperature column in the CDS CSV.")


def _escape_toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
