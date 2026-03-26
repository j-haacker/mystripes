from __future__ import annotations

import calendar
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
                boundary_geojson=entry.get("boundary_geojson"),
                bounding_box=entry.get("bounding_box"),
            )
        )
        current_start = current_end + timedelta(days=1)

    if periods and periods[-1].end_date != analysis_end:
        errors.append("The configured periods do not reach the latest available date.")

    return periods, errors


def combine_period_frames(
    periods: list[LifePeriod],
    frames_by_period: list[pd.DataFrame],
    aggregation_mode: str = "full_calendar_years",
    rolling_window_end: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined_frames: list[pd.DataFrame] = []
    for period, frame in zip(periods, frames_by_period, strict=True):
        local_frame = frame.copy()
        local_frame["sample_start_date"] = local_frame["timestamp"].dt.date.map(_sample_start_date)
        local_frame["sample_end_date"] = local_frame["timestamp"].dt.date.map(_sample_end_date)
        local_frame["covered_start_date"] = local_frame["sample_start_date"].map(
            lambda sample_start: max(period.start_date, sample_start)
        )
        local_frame["covered_end_date"] = local_frame["sample_end_date"].map(
            lambda sample_end: min(period.end_date, sample_end)
        )
        local_frame["overlap_days"] = local_frame.apply(
            lambda row: _overlap_days(
                period.start_date,
                period.end_date,
                row["sample_start_date"],
                row["sample_end_date"],
            ),
            axis=1,
        )
        local_frame = local_frame.loc[local_frame["overlap_days"] > 0].copy()
        local_frame["period_label"] = period.label
        local_frame["place_name"] = period.display_name
        local_frame["weighted_temp_sum"] = local_frame["temperature_c"] * local_frame["overlap_days"]
        combined_frames.append(local_frame)

    if not combined_frames:
        raise ValueError("No monthly data was available for the selected life periods.")

    combined = pd.concat(combined_frames, ignore_index=True).sort_values("timestamp")
    if aggregation_mode == "full_calendar_years":
        yearly = _aggregate_full_calendar_years(combined)
    elif aggregation_mode == "rolling_365_day":
        effective_window_end = rolling_window_end or max(period.end_date for period in periods)
        yearly = _aggregate_rolling_365_day_windows(combined, effective_window_end)
    else:
        raise ValueError(f"Unsupported aggregation mode: {aggregation_mode}")

    return combined, yearly


def build_stripe_frame(yearly: pd.DataFrame, baseline_c: float) -> pd.DataFrame:
    stripe_frame = yearly.copy()
    stripe_frame["baseline_c"] = baseline_c
    stripe_frame["anomaly_c"] = stripe_frame["mean_temp_c"] - baseline_c
    return stripe_frame


def calculate_life_period_baseline(yearly: pd.DataFrame) -> float:
    if "days_covered" in yearly.columns and yearly["days_covered"].sum() > 0:
        weighted_sum = (yearly["mean_temp_c"] * yearly["days_covered"]).sum()
        return float(weighted_sum / yearly["days_covered"].sum())
    return float(yearly["mean_temp_c"].mean())


def calculate_series_mean_temperature(frame: pd.DataFrame) -> float:
    if "sample_days" in frame.columns and frame["sample_days"].sum() > 0:
        weighted_sum = (frame["temperature_c"] * frame["sample_days"]).sum()
        return float(weighted_sum / frame["sample_days"].sum())
    return float(frame["temperature_c"].mean())


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


def _sample_start_date(sample_date: date) -> date:
    return sample_date.replace(day=1)


def _sample_end_date(sample_date: date) -> date:
    last_day = calendar.monthrange(sample_date.year, sample_date.month)[1]
    return sample_date.replace(day=last_day)


def _aggregate_full_calendar_years(combined: pd.DataFrame) -> pd.DataFrame:
    combined = combined.copy()
    combined["year"] = combined["timestamp"].dt.year

    yearly = (
        combined.groupby("year", as_index=False)
        .agg(
            weighted_temp_sum=("weighted_temp_sum", "sum"),
            days_covered=("overlap_days", "sum"),
            months_covered=("timestamp", "size"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    yearly["window_start"] = yearly["year"].map(lambda year: date(year, 1, 1))
    yearly["window_end"] = yearly["year"].map(lambda year: date(year, 12, 31))
    yearly["expected_days"] = yearly["year"].map(_days_in_year)
    yearly = yearly.loc[yearly["days_covered"] == yearly["expected_days"]].copy()
    if yearly.empty:
        raise ValueError(
            "No complete calendar years are available for the selected period. "
            "Try the rolling 365-day window mode instead."
        )
    yearly["mean_temp_c"] = yearly["weighted_temp_sum"] / yearly["days_covered"]
    return yearly[["year", "window_start", "window_end", "mean_temp_c", "days_covered", "months_covered"]]


def _aggregate_rolling_365_day_windows(
    combined: pd.DataFrame,
    rolling_window_end: date,
) -> pd.DataFrame:
    coverage_start = min(combined["covered_start_date"])
    coverage_end = max(combined["covered_end_date"])
    effective_window_end = min(rolling_window_end, coverage_end)
    earliest_complete_end = coverage_start + timedelta(days=364)

    if effective_window_end < earliest_complete_end:
        raise ValueError(
            "No complete 365-day window is available for the selected period. "
            "Try a longer date range or use full calendar years."
        )

    records: list[dict[str, object]] = []
    current_window_end = effective_window_end
    while current_window_end >= earliest_complete_end:
        window_start = current_window_end - timedelta(days=364)
        overlapping = combined.loc[
            (combined["covered_end_date"] >= window_start)
            & (combined["covered_start_date"] <= current_window_end)
        ].copy()
        overlapping["window_overlap_days"] = overlapping.apply(
            lambda row: _overlap_days(
                window_start,
                current_window_end,
                row["covered_start_date"],
                row["covered_end_date"],
            ),
            axis=1,
        )
        overlapping = overlapping.loc[overlapping["window_overlap_days"] > 0]

        days_covered = int(overlapping["window_overlap_days"].sum())
        if days_covered == 365:
            weighted_temp_sum = float(
                (overlapping["temperature_c"] * overlapping["window_overlap_days"]).sum()
            )
            records.append(
                {
                    "year": current_window_end.year,
                    "window_start": window_start,
                    "window_end": current_window_end,
                    "mean_temp_c": weighted_temp_sum / days_covered,
                    "days_covered": days_covered,
                    "months_covered": int(overlapping["timestamp"].nunique()),
                }
            )

        next_window_end = _shift_year(current_window_end, -1)
        if next_window_end >= current_window_end:
            break
        current_window_end = next_window_end

    if not records:
        raise ValueError(
            "No complete 365-day windows are available for the selected period. "
            "Try full calendar years instead."
        )

    return pd.DataFrame.from_records(records).sort_values("window_end").reset_index(drop=True)


def _overlap_days(
    period_start: date,
    period_end: date,
    sample_start: date,
    sample_end: date,
) -> int:
    overlap_start = max(period_start, sample_start)
    overlap_end = min(period_end, sample_end)
    if overlap_start > overlap_end:
        return 0
    return (overlap_end - overlap_start).days + 1


def _days_in_year(year: int) -> int:
    return 366 if calendar.isleap(year) else 365


def _shift_year(value: date, years: int) -> date:
    target_year = value.year + years
    try:
        return value.replace(year=target_year)
    except ValueError:
        return value.replace(year=target_year, day=28)
