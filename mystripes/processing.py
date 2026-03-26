from __future__ import annotations

import calendar
from datetime import date, timedelta
from typing import Any

import pandas as pd

from mystripes.models import LifePeriod


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
    rolling_sample_mode: str = "monthly",
    rolling_strip_count: int | None = None,
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
        local_frame["location_key"] = period.location_key
        local_frame["weighted_temp_sum"] = local_frame["temperature_c"] * local_frame["overlap_days"]
        combined_frames.append(local_frame)

    if not combined_frames:
        raise ValueError("No monthly data was available for the selected periods.")

    combined = pd.concat(combined_frames, ignore_index=True).sort_values("timestamp")
    if aggregation_mode == "full_calendar_years":
        yearly = _aggregate_full_calendar_years(combined)
    elif aggregation_mode == "rolling_365_day":
        effective_window_end = rolling_window_end or max(period.end_date for period in periods)
        yearly = _aggregate_rolling_365_day_windows(
            combined,
            effective_window_end,
            rolling_sample_mode=rolling_sample_mode,
            rolling_strip_count=rolling_strip_count,
        )
    else:
        raise ValueError(f"Unsupported aggregation mode: {aggregation_mode}")

    return combined, yearly


def build_stripe_frame(yearly: pd.DataFrame, baseline_c: float) -> pd.DataFrame:
    stripe_frame = yearly.copy()
    stripe_frame["baseline_c"] = baseline_c
    stripe_frame["anomaly_c"] = stripe_frame["mean_temp_c"] - baseline_c
    return stripe_frame


def build_location_baseline_stripe_frame(
    combined: pd.DataFrame,
    baseline_by_location: dict[str, float],
    aggregation_mode: str = "full_calendar_years",
    rolling_window_end: date | None = None,
    rolling_sample_mode: str = "monthly",
    rolling_strip_count: int | None = None,
) -> pd.DataFrame:
    stripe_source = combined.copy()
    stripe_source["baseline_c"] = stripe_source["location_key"].map(baseline_by_location)

    missing_locations = sorted(
        {
            str(location_key)
            for location_key in stripe_source.loc[stripe_source["baseline_c"].isna(), "location_key"].unique()
        }
    )
    if missing_locations:
        raise ValueError(
            "Missing location baselines for: " + ", ".join(missing_locations)
        )

    stripe_source["weighted_baseline_sum"] = stripe_source["baseline_c"] * stripe_source["overlap_days"]
    stripe_source["weighted_anomaly_sum"] = (
        (stripe_source["temperature_c"] - stripe_source["baseline_c"]) * stripe_source["overlap_days"]
    )

    if aggregation_mode == "full_calendar_years":
        return _aggregate_full_calendar_years_with_location_baseline(stripe_source)
    if aggregation_mode == "rolling_365_day":
        effective_window_end = rolling_window_end or max(stripe_source["covered_end_date"])
        return _aggregate_rolling_365_day_windows_with_location_baseline(
            stripe_source,
            effective_window_end,
            rolling_sample_mode=rolling_sample_mode,
            rolling_strip_count=rolling_strip_count,
        )

    raise ValueError(f"Unsupported aggregation mode: {aggregation_mode}")


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
    yearly["sample_date"] = yearly["window_end"]
    yearly["expected_days"] = yearly["year"].map(_days_in_year)
    yearly = yearly.loc[yearly["days_covered"] == yearly["expected_days"]].copy()
    if yearly.empty:
        raise ValueError(
            "No complete calendar years are available for the selected period. "
            "Try the 365-day moving-average mode instead."
        )
    yearly["mean_temp_c"] = yearly["weighted_temp_sum"] / yearly["days_covered"]
    return yearly[["year", "sample_date", "window_start", "window_end", "mean_temp_c", "days_covered", "months_covered"]]


def _aggregate_full_calendar_years_with_location_baseline(combined: pd.DataFrame) -> pd.DataFrame:
    combined = combined.copy()
    combined["year"] = combined["timestamp"].dt.year

    yearly = (
        combined.groupby("year", as_index=False)
        .agg(
            weighted_temp_sum=("weighted_temp_sum", "sum"),
            weighted_baseline_sum=("weighted_baseline_sum", "sum"),
            weighted_anomaly_sum=("weighted_anomaly_sum", "sum"),
            days_covered=("overlap_days", "sum"),
            months_covered=("timestamp", "size"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    yearly["window_start"] = yearly["year"].map(lambda year: date(year, 1, 1))
    yearly["window_end"] = yearly["year"].map(lambda year: date(year, 12, 31))
    yearly["sample_date"] = yearly["window_end"]
    yearly["expected_days"] = yearly["year"].map(_days_in_year)
    yearly = yearly.loc[yearly["days_covered"] == yearly["expected_days"]].copy()
    if yearly.empty:
        raise ValueError(
            "No complete calendar years are available for the selected period. "
            "Try the 365-day moving-average mode instead."
        )
    yearly["mean_temp_c"] = yearly["weighted_temp_sum"] / yearly["days_covered"]
    yearly["baseline_c"] = yearly["weighted_baseline_sum"] / yearly["days_covered"]
    yearly["anomaly_c"] = yearly["weighted_anomaly_sum"] / yearly["days_covered"]
    return yearly[
        [
            "year",
            "sample_date",
            "window_start",
            "window_end",
            "mean_temp_c",
            "baseline_c",
            "anomaly_c",
            "days_covered",
            "months_covered",
        ]
    ]


def _aggregate_rolling_365_day_windows(
    combined: pd.DataFrame,
    rolling_window_end: date,
    rolling_sample_mode: str = "monthly",
    rolling_strip_count: int | None = None,
) -> pd.DataFrame:
    return _aggregate_rolling_daily_series(
        combined,
        rolling_window_end,
        rolling_sample_mode=rolling_sample_mode,
        rolling_strip_count=rolling_strip_count,
        include_location_baseline=False,
    )


def _aggregate_rolling_365_day_windows_with_location_baseline(
    combined: pd.DataFrame,
    rolling_window_end: date,
    rolling_sample_mode: str = "monthly",
    rolling_strip_count: int | None = None,
) -> pd.DataFrame:
    return _aggregate_rolling_daily_series(
        combined,
        rolling_window_end,
        rolling_sample_mode=rolling_sample_mode,
        rolling_strip_count=rolling_strip_count,
        include_location_baseline=True,
    )


def _aggregate_rolling_daily_series(
    combined: pd.DataFrame,
    rolling_window_end: date,
    rolling_sample_mode: str,
    rolling_strip_count: int | None,
    include_location_baseline: bool,
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

    daily = _expand_combined_to_daily_frame(combined, include_location_baseline=include_location_baseline)
    daily = daily.loc[daily["daily_date"] <= pd.Timestamp(effective_window_end)].copy()
    daily = daily.sort_values("daily_date").reset_index(drop=True)

    if len(daily) < 365:
        raise ValueError(
            "No complete 365-day window is available for the selected period. "
            "Try a longer date range or use full calendar years."
        )

    daily["mean_temp_c"] = daily["temperature_c"].rolling(window=365, min_periods=365).mean()
    if include_location_baseline:
        daily["baseline_c"] = daily["baseline_c"].rolling(window=365, min_periods=365).mean()
        daily["anomaly_c"] = daily["mean_temp_c"] - daily["baseline_c"]

    valid = daily.loc[daily["mean_temp_c"].notna()].copy()
    sampled = _sample_rolling_daily_frame(
        valid,
        rolling_sample_mode=rolling_sample_mode,
        rolling_strip_count=rolling_strip_count,
    )
    sampled["sample_date"] = sampled["daily_date"].dt.date
    sampled["year"] = sampled["daily_date"].dt.year
    sampled["window_end"] = sampled["sample_date"]
    sampled["window_start"] = sampled["sample_date"].map(lambda value: value - timedelta(days=364))
    sampled["days_covered"] = 365
    sampled["months_covered"] = sampled.apply(
        lambda row: _months_touched(row["window_start"], row["window_end"]),
        axis=1,
    )

    if include_location_baseline:
        return sampled[
            [
                "year",
                "sample_date",
                "window_start",
                "window_end",
                "mean_temp_c",
                "baseline_c",
                "anomaly_c",
                "days_covered",
                "months_covered",
            ]
        ].reset_index(drop=True)

    return sampled[
        ["year", "sample_date", "window_start", "window_end", "mean_temp_c", "days_covered", "months_covered"]
    ].reset_index(drop=True)


def _expand_combined_to_daily_frame(
    combined: pd.DataFrame,
    include_location_baseline: bool,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for row in combined.itertuples(index=False):
        days = pd.date_range(start=row.covered_start_date, end=row.covered_end_date, freq="D")
        payload: dict[str, object] = {
            "daily_date": days,
            "temperature_c": [float(row.temperature_c)] * len(days),
        }
        if include_location_baseline:
            payload["baseline_c"] = [float(row.baseline_c)] * len(days)
        frames.append(pd.DataFrame(payload))

    if not frames:
        return pd.DataFrame(columns=["daily_date", "temperature_c", "baseline_c"])

    daily = pd.concat(frames, ignore_index=True).sort_values("daily_date").drop_duplicates("daily_date")
    return daily.reset_index(drop=True)


def _sample_rolling_daily_frame(
    daily: pd.DataFrame,
    rolling_sample_mode: str,
    rolling_strip_count: int | None,
) -> pd.DataFrame:
    if daily.empty:
        raise ValueError(
            "No complete 365-day moving-average samples are available for the selected period. "
            "Try full calendar years instead."
        )

    if rolling_sample_mode == "monthly":
        sampled = (
            daily.set_index("daily_date")
            .resample("ME")
            .last()
            .dropna(subset=["mean_temp_c"])
            .reset_index()
        )
        if sampled.empty:
            raise ValueError(
                "No monthly rolling samples are available for the selected period."
            )
        return sampled

    if rolling_sample_mode == "fixed_count":
        count = rolling_strip_count or 60
        if count <= 0:
            raise ValueError("The rolling strip count must be greater than zero.")
        indices = _evenly_spaced_indices(len(daily), count)
        return daily.iloc[indices].copy().reset_index(drop=True)

    raise ValueError(f"Unsupported rolling sample mode: {rolling_sample_mode}")


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    if count <= 1:
        return [length - 1]
    if count >= length:
        return list(range(length))

    denominator = count - 1
    return sorted(
        {
            int(round(position * (length - 1) / denominator))
            for position in range(count)
        }
    )


def _months_touched(start_date: date, end_date: date) -> int:
    return len(pd.period_range(start=start_date, end=end_date, freq="M"))


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

