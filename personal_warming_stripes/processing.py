from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from personal_warming_stripes.models import LifePeriod


def build_periods_from_entries(
    entries: list[dict[str, Any]],
    birth_date: date,
    analysis_end: date,
    analysis_min_start: date,
) -> tuple[list[LifePeriod], list[str]]:
    errors: list[str] = []
    periods: list[LifePeriod] = []

    analysis_start = max(birth_date, analysis_min_start)
    current_start = analysis_start
    total_entries = len(entries)

    if total_entries == 0:
        return [], ["Add at least one living period."]

    for index, entry in enumerate(entries):
        is_final_period = index == total_entries - 1
        if current_start > analysis_end:
            errors.append("The move dates leave no remaining time for the current period.")
            break

        current_end = analysis_end if is_final_period else entry.get("end_date")
        if not is_final_period and current_end is None:
            errors.append(f"Period {index + 1} needs a 'living here until' date.")
            break
        if current_end is None:
            current_end = analysis_end
        if current_end < current_start:
            errors.append(f"Period {index + 1} ends before it starts.")
            break
        if not is_final_period and current_end >= analysis_end:
            errors.append(
                f"Period {index + 1} must end before the final period starts on {analysis_end}."
            )
            break

        latitude, longitude, coordinate_error = _parse_coordinates(entry, index)
        if coordinate_error:
            errors.append(coordinate_error)
            break

        label = (str(entry.get("custom_label", "")).strip() or "").strip()
        resolved_name = str(entry.get("resolved_name", "")).strip()
        place_query = str(entry.get("place_query", "")).strip()

        periods.append(
            LifePeriod(
                label=label or resolved_name or place_query or f"Period {index + 1}",
                place_query=place_query,
                resolved_name=resolved_name,
                start_date=current_start,
                end_date=current_end,
                latitude=latitude,
                longitude=longitude,
            )
        )
        current_start = current_end + timedelta(days=1)

    if periods and periods[-1].end_date != analysis_end:
        errors.append("The configured periods do not reach the latest available date.")

    return periods, errors


def combine_period_frames(
    periods: list[LifePeriod],
    frames_by_period: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined_frames: list[pd.DataFrame] = []
    for period, frame in zip(periods, frames_by_period, strict=True):
        local_frame = frame.copy()
        mask = _period_mask(local_frame["timestamp"], period.start_date, period.end_date)
        local_frame = local_frame.loc[mask].copy()
        local_frame["period_label"] = period.label
        local_frame["place_name"] = period.display_name
        combined_frames.append(local_frame)

    combined = pd.concat(combined_frames, ignore_index=True).sort_values("timestamp")
    combined["year"] = combined["timestamp"].dt.year

    yearly = (
        combined.groupby("year", as_index=False)
        .agg(mean_temp_c=("temperature_c", "mean"), hours_covered=("temperature_c", "size"))
        .sort_values("year")
        .reset_index(drop=True)
    )
    yearly["coverage_days_equivalent"] = yearly["hours_covered"] / 24.0
    return combined, yearly


def build_stripe_frame(yearly: pd.DataFrame, baseline_c: float) -> pd.DataFrame:
    stripe_frame = yearly.copy()
    stripe_frame["baseline_c"] = baseline_c
    stripe_frame["anomaly_c"] = stripe_frame["mean_temp_c"] - baseline_c
    return stripe_frame


def calculate_life_period_baseline(yearly: pd.DataFrame) -> float:
    return float(yearly["mean_temp_c"].mean())


def calculate_weighted_location_baseline(
    periods: list[LifePeriod],
    baseline_by_location: dict[str, float],
) -> float:
    total_days = sum(period.days for period in periods)
    if total_days <= 0:
        raise ValueError("At least one day of coverage is required.")

    weighted_sum = 0.0
    for period in periods:
        weighted_sum += baseline_by_location[period.location_key] * period.days
    return weighted_sum / total_days


def unique_locations(periods: list[LifePeriod]) -> dict[str, tuple[float, float]]:
    unique: dict[str, tuple[float, float]] = {}
    for period in periods:
        unique[period.location_key] = (period.latitude, period.longitude)
    return unique


def _parse_coordinates(entry: dict[str, Any], index: int) -> tuple[float, float, str | None]:
    latitude_text = str(entry.get("latitude_text", "")).strip()
    longitude_text = str(entry.get("longitude_text", "")).strip()
    if not latitude_text or not longitude_text:
        return 0.0, 0.0, f"Period {index + 1} needs either a resolved place or manual coordinates."

    try:
        latitude = float(latitude_text)
        longitude = float(longitude_text)
    except ValueError:
        return 0.0, 0.0, f"Period {index + 1} has invalid coordinate values."

    if not -89.0 <= latitude <= 89.0:
        return 0.0, 0.0, f"Period {index + 1} latitude must be between -89 and 89."
    if not -180.0 <= longitude <= 180.0:
        return 0.0, 0.0, f"Period {index + 1} longitude must be between -180 and 180."

    return latitude, longitude, None


def _period_mask(
    timestamps: pd.Series,
    start_date: date,
    end_date: date,
) -> pd.Series:
    dates = timestamps.dt.date
    return (dates >= start_date) & (dates <= end_date)
