from __future__ import annotations

import re
from datetime import date
from urllib.request import urlopen

import pandas as pd

NASA_GISTEMP_GLOBAL_MEAN_URL = (
    "https://data.giss.nasa.gov/gistemp/graphs/graph_data/"
    "Global_Mean_Estimates_based_on_Land_and_Ocean_Data/graph.txt"
)

_GISTEMP_ROW_PATTERN = re.compile(
    r"^\s*(?P<year>\d{4})\s+(?P<anomaly>-?\d+(?:\.\d+)?)\s+(?P<lowess>-?\d+(?:\.\d+)?)\s*$"
)


def fetch_global_mean_estimates(
    url: str = NASA_GISTEMP_GLOBAL_MEAN_URL,
    timeout: int = 30,
) -> str:
    with urlopen(url, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def parse_global_mean_estimates(text: str) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for line in text.splitlines():
        match = _GISTEMP_ROW_PATTERN.match(line)
        if match is None:
            continue
        rows.append(
            {
                "year": int(match.group("year")),
                "anomaly_c": float(match.group("anomaly")),
                "lowess_5y_c": float(match.group("lowess")),
            }
        )

    if not rows:
        raise ValueError("Could not parse any annual NASA GISTEMP global mean estimates.")

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def load_global_mean_estimates(
    url: str = NASA_GISTEMP_GLOBAL_MEAN_URL,
    timeout: int = 30,
) -> pd.DataFrame:
    return parse_global_mean_estimates(fetch_global_mean_estimates(url=url, timeout=timeout))


def average_global_warming_for_period(
    annual_estimates: pd.DataFrame,
    period_start: date,
    period_end: date,
) -> tuple[float, int, int]:
    if period_end < period_start:
        raise ValueError("The reference period end date must be on or after the start date.")

    if "year" not in annual_estimates.columns or "anomaly_c" not in annual_estimates.columns:
        raise ValueError("NASA GISTEMP annual estimates need `year` and `anomaly_c` columns.")

    annual = annual_estimates[["year", "anomaly_c"]].copy()
    annual["year"] = pd.to_numeric(annual["year"], errors="coerce")
    annual["anomaly_c"] = pd.to_numeric(annual["anomaly_c"], errors="coerce")
    annual = annual.dropna(subset=["year", "anomaly_c"]).sort_values("year").reset_index(drop=True)
    if annual.empty:
        raise ValueError("NASA GISTEMP annual estimates are empty.")

    available_start_year = int(annual["year"].min())
    available_end_year = int(annual["year"].max())
    effective_start_year = max(period_start.year, available_start_year)
    effective_end_year = min(period_end.year, available_end_year)
    if effective_end_year < effective_start_year:
        raise ValueError("No NASA GISTEMP annual estimates overlap the selected reference period.")

    window = annual.loc[annual["year"].between(effective_start_year, effective_end_year)].copy()
    if window.empty:
        raise ValueError("No NASA GISTEMP annual estimates overlap the selected reference period.")

    return float(window["anomaly_c"].mean()), effective_start_year, effective_end_year
