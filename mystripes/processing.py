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


def calculate_series_mean_temperature(frame: pd.DataFrame) -> float:
    if "sample_days" in frame.columns and frame["sample_days"].sum() > 0:
        weighted_sum = (frame["temperature_c"] * frame["sample_days"]).sum()
        return float(weighted_sum / frame["sample_days"].sum())
    return float(frame["temperature_c"].mean())

def build_period_report_tables(
    periods: list[LifePeriod],
    frames_by_period: list[pd.DataFrame],
    period_baselines: list[float] | None,
    report_start: date,
    report_end: date,
    baseline_start: date | None = None,
    baseline_end: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_temperature, daily_baseline, daily_anomaly = _build_period_daily_tables(
        periods=periods,
        frames_by_period=frames_by_period,
        report_start=report_start,
        report_end=report_end,
        period_baselines=period_baselines,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
    )
    full_report = _build_full_period_report(
        daily_temperature=daily_temperature,
        daily_baseline=daily_baseline,
        daily_anomaly=daily_anomaly,
    )

    merged_daily = _assemble_merged_daily_series(
        periods=periods,
        daily_temperature=daily_temperature,
        daily_baseline=daily_baseline,
        daily_anomaly=daily_anomaly,
        report_start=report_start,
        report_end=report_end,
        first_period_history_days=0,
    )

    merged = (
        merged_daily.assign(
            timestamp=merged_daily["daily_date"].dt.to_period("M").dt.to_timestamp()
        )
        .groupby("timestamp", as_index=False)
        .agg(
            temperature_c=("temperature_c", "mean"),
            climatology_c=("climatology_c", "mean"),
            anomaly_c=("anomaly_c", "mean"),
            days_covered=("daily_date", "size"),
            current_period=("current_period", _join_unique_values),
            current_place=("current_place", _join_unique_values),
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    merged["sample_date"] = merged["timestamp"].dt.date
    merged = merged[
        [
            "timestamp",
            "sample_date",
            "current_period",
            "current_place",
            "temperature_c",
            "climatology_c",
            "anomaly_c",
            "days_covered",
        ]
    ]

    return full_report, merged


def build_merged_daily_series(
    periods: list[LifePeriod],
    frames_by_period: list[pd.DataFrame],
    report_start: date,
    report_end: date,
    period_baselines: list[float] | None = None,
    baseline_start: date | None = None,
    baseline_end: date | None = None,
    first_period_history_days: int = 0,
) -> pd.DataFrame:
    daily_temperature, daily_baseline, daily_anomaly = _build_period_daily_tables(
        periods=periods,
        frames_by_period=frames_by_period,
        report_start=report_start,
        report_end=report_end,
        period_baselines=period_baselines,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
    )
    return _assemble_merged_daily_series(
        periods=periods,
        daily_temperature=daily_temperature,
        daily_baseline=daily_baseline,
        daily_anomaly=daily_anomaly,
        report_start=report_start,
        report_end=report_end,
        first_period_history_days=first_period_history_days,
    )


def aggregate_daily_series_to_stripes(
    daily_series: pd.DataFrame,
    aggregation_mode: str = "full_calendar_years",
    rolling_window_end: date | None = None,
    rolling_crop_start: date | None = None,
    rolling_sample_mode: str = "monthly",
    rolling_strip_count: int | None = None,
) -> pd.DataFrame:
    if aggregation_mode == "full_calendar_years":
        return new_yearly(daily_series)
    if aggregation_mode == "rolling_365_day":
        effective_window_end = rolling_window_end or max(daily_series["sample_date"])
        effective_crop_start = rolling_crop_start or min(daily_series["sample_date"])
        return _aggregate_rolling_daily_series_from_daily(
            daily_series=daily_series,
            rolling_window_end=effective_window_end,
            rolling_crop_start=effective_crop_start,
            rolling_sample_mode=rolling_sample_mode,
            rolling_strip_count=rolling_strip_count,
        )

    raise ValueError(f"Unsupported aggregation mode: {aggregation_mode}")


def _build_period_daily_tables(
    periods: list[LifePeriod],
    frames_by_period: list[pd.DataFrame],
    report_start: date,
    report_end: date,
    period_baselines: list[float] | None = None,
    baseline_start: date | None = None,
    baseline_end: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(frames_by_period) != len(periods):
        raise ValueError("Each period needs a matching temperature frame.")
    if period_baselines is not None and len(period_baselines) != len(periods):
        raise ValueError("Each period needs a matching baseline value.")

    effective_baseline_start = baseline_start or report_start
    effective_baseline_end = baseline_end or report_end
    if effective_baseline_end < effective_baseline_start:
        raise ValueError("The baseline date range must not end before it starts.")

    daily_temperature = _build_daily_temperature_table(
        periods=periods,
        frames_by_period=frames_by_period,
        report_start=report_start,
        report_end=report_end,
    )
    daily_baseline = _build_daily_baseline_table(
        daily_temperature=daily_temperature,
        period_baselines=period_baselines,
        baseline_start=effective_baseline_start,
        baseline_end=effective_baseline_end,
    )
    daily_anomaly = _build_daily_anomaly_table(
        daily_temperature=daily_temperature,
        daily_baseline=daily_baseline,
    )
    return daily_temperature, daily_baseline, daily_anomaly


def _build_daily_temperature_table(
    periods: list[LifePeriod],
    frames_by_period: list[pd.DataFrame],
    report_start: date,
    report_end: date,
) -> pd.DataFrame:
    monthly_table = _build_monthly_temperature_table(
        periods=periods,
        frames_by_period=frames_by_period,
        report_start=report_start,
        report_end=report_end,
    )
    daily_index = pd.date_range(start=report_start, end=report_end, freq="D")
    naive_monthly = monthly_table.copy()
    if isinstance(naive_monthly.index, pd.DatetimeIndex) and naive_monthly.index.tz is not None:
        naive_monthly.index = naive_monthly.index.tz_localize(None)
    return naive_monthly.reindex(daily_index, method="ffill")


def _build_monthly_temperature_table(
    periods: list[LifePeriod],
    frames_by_period: list[pd.DataFrame],
    report_start: date,
    report_end: date,
) -> pd.DataFrame:
    report_index = pd.date_range(
        start=_sample_start_date(report_start),
        end=_sample_start_date(report_end),
        freq="MS",
        tz="UTC",
    )
    temperature_table = pd.DataFrame(index=report_index)
    for index, (period, frame) in enumerate(zip(periods, frames_by_period, strict=True)):
        period_key = _period_report_key(index, period)
        aligned = frame.drop_duplicates("timestamp").set_index("timestamp").reindex(report_index)
        temperature_table[period_key] = pd.to_numeric(aligned["temperature_c"], errors="coerce")
    return temperature_table


def _build_daily_baseline_table(
    daily_temperature: pd.DataFrame,
    period_baselines: list[float] | None,
    baseline_start: date,
    baseline_end: date,
) -> pd.DataFrame:
    baseline_index = pd.date_range(start=baseline_start, end=baseline_end, freq="D")
    baseline_reference = daily_temperature.reindex(baseline_index)

    daily_baseline = pd.DataFrame(index=daily_temperature.index)
    if period_baselines is not None:
        if len(period_baselines) != len(daily_temperature.columns):
            raise ValueError("Each period needs a matching baseline value.")
        for index, column in enumerate(daily_temperature.columns):
            daily_baseline[column] = float(period_baselines[index])
        return daily_baseline

    for column in daily_temperature.columns:
        daily_baseline[column] = new_baseline(
            daily_temperature[column],
            reference=baseline_reference[column],
        )
    return daily_baseline


def _build_daily_anomaly_table(
    daily_temperature: pd.DataFrame,
    daily_baseline: pd.DataFrame,
) -> pd.DataFrame:
    daily_anomaly = pd.DataFrame(index=daily_temperature.index)
    for column in daily_temperature.columns:
        daily_anomaly[column] = new_anomaly(
            daily_temperature[column],
            daily_baseline[column],
        )
    return daily_anomaly


def _assemble_merged_daily_series(
    periods: list[LifePeriod],
    daily_temperature: pd.DataFrame,
    daily_baseline: pd.DataFrame,
    daily_anomaly: pd.DataFrame,
    report_start: date,
    report_end: date,
    first_period_history_days: int,
) -> pd.DataFrame:
    active_period, active_place = _build_active_metadata(
        periods=periods,
        daily_temperature=daily_temperature,
        report_start=report_start,
        report_end=report_end,
        first_period_history_days=first_period_history_days,
    )
    merged = pd.DataFrame(index=daily_temperature.index)
    merged["current_period"] = active_period
    merged["current_place"] = active_place
    merged["temperature_c"] = new_merge(
        daily_temperature,
        periods,
        report_start=report_start,
        report_end=report_end,
        first_period_history_days=first_period_history_days,
    )
    merged["climatology_c"] = new_merge(
        daily_baseline,
        periods,
        report_start=report_start,
        report_end=report_end,
        first_period_history_days=first_period_history_days,
    )
    merged["anomaly_c"] = new_merge(
        daily_anomaly,
        periods,
        report_start=report_start,
        report_end=report_end,
        first_period_history_days=first_period_history_days,
    )

    merged = merged.reset_index(names="daily_date")
    merged["sample_date"] = merged["daily_date"].dt.date
    return merged[
        [
            "daily_date",
            "sample_date",
            "current_period",
            "current_place",
            "temperature_c",
            "climatology_c",
            "anomaly_c",
        ]
    ]


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


def _sample_start_date(sample_date: date) -> date:
    return sample_date.replace(day=1)

def _aggregate_rolling_daily_series_from_daily(
    daily_series: pd.DataFrame,
    rolling_window_end: date,
    rolling_crop_start: date,
    rolling_sample_mode: str,
    rolling_strip_count: int | None,
) -> pd.DataFrame:
    daily = daily_series.copy().sort_values("daily_date").reset_index(drop=True)
    coverage_start = min(daily["sample_date"])
    coverage_end = max(daily["sample_date"])
    effective_window_end = min(rolling_window_end, coverage_end)
    earliest_complete_end = coverage_start + timedelta(days=364)

    if effective_window_end < earliest_complete_end:
        raise ValueError(
            "No complete 365-day window is available for the selected period. "
            "Try a longer date range or use full calendar years."
        )

    daily = daily.loc[daily["sample_date"] <= effective_window_end].copy()
    if len(daily) < 365:
        raise ValueError(
            "No complete 365-day window is available for the selected period. "
            "Try a longer date range or use full calendar years."
        )

    daily["mean_temp_c"] = daily["temperature_c"].rolling(window=365, min_periods=365).mean()
    daily["baseline_c"] = daily["climatology_c"].rolling(window=365, min_periods=365).mean()
    daily["anomaly_c"] = daily["anomaly_c"].rolling(window=365, min_periods=365).mean()

    valid = daily.loc[daily["mean_temp_c"].notna()].copy()
    valid = valid.loc[valid["sample_date"] >= rolling_crop_start].copy()
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

    columns = ["year", "sample_date", "window_start", "window_end", "mean_temp_c"]
    if "baseline_c" in sampled.columns:
        columns.append("baseline_c")
    if "anomaly_c" in sampled.columns:
        columns.append("anomaly_c")
    columns.extend(["days_covered", "months_covered"])
    return sampled[columns].reset_index(drop=True)


def new_baseline(
    data: pd.Series,
    reference: pd.Series | None = None,
) -> pd.Series:
    return _new_baseline_default(data, reference=reference)


def new_anomaly(data, baseline):
    temperature = pd.to_numeric(data, errors="coerce")
    reference = pd.to_numeric(baseline, errors="coerce")
    return (temperature - reference).astype("float64")


def new_merge(
    anom_df: pd.DataFrame,
    periods: list[LifePeriod],
    report_start: date,
    report_end: date,
    first_period_history_days: int = 0,
) -> pd.Series:
    active_ranges, required_start = _active_ranges(
        periods=periods,
        daily_temperature=anom_df,
        report_start=report_start,
        report_end=report_end,
        first_period_history_days=first_period_history_days,
    )
    merged = pd.Series(index=anom_df.index, dtype="float64")
    for _, period_key, active_days in active_ranges:
        if merged.loc[active_days].notna().any():
            raise ValueError("The configured periods overlap in the selected timeline.")
        values = pd.to_numeric(anom_df.loc[active_days, period_key], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Missing values for {period_key} in the selected timeline.")
        merged.loc[active_days] = values.to_numpy()
    if merged.loc[pd.Timestamp(required_start):].isna().any():
        raise ValueError("The configured periods do not cover every day in the selected timeline.")
    return merged.astype("float64")


def new_yearly(merged: pd.DataFrame) -> pd.DataFrame:
    daily = merged.copy()
    daily["year"] = daily["daily_date"].dt.year

    yearly = (
        daily.groupby("year", as_index=False)
        .agg(
            mean_temp_c=("temperature_c", "mean"),
            baseline_c=("climatology_c", "mean"),
            anomaly_c=("anomaly_c", "mean"),
            days_covered=("daily_date", "size"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    yearly["window_start"] = yearly["year"].map(lambda year: date(year, 1, 1))
    yearly["window_end"] = yearly["year"].map(lambda year: date(year, 12, 31))
    yearly["sample_date"] = yearly["window_end"]
    yearly["months_covered"] = 12
    yearly["expected_days"] = yearly["year"].map(_days_in_year)
    yearly = yearly.loc[yearly["days_covered"] == yearly["expected_days"]].copy()
    if yearly.empty:
        raise ValueError(
            "No complete calendar years are available for the selected period. "
            "Try the 365-day moving-average mode instead."
        )

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


def _new_baseline_default(
    data: pd.Series,
    reference: pd.Series | None = None,
) -> pd.Series:
    # This is the single place to change how non-location-specific baselines are computed.
    source = pd.to_numeric(reference if reference is not None else data, errors="coerce").dropna()
    if source.empty:
        name = data.name or "period"
        raise ValueError(f"Missing baseline temperatures for: {name}")

    doy_index = data.index.dayofyear
    doy_means = data.groupby(doy_index).mean()
    baseline_value = doy_means.loc[doy_index]
    baseline_value.index = data.index
    return pd.Series(baseline_value, index=data.index, dtype="float64")


def _build_full_period_report(
    daily_temperature: pd.DataFrame,
    daily_baseline: pd.DataFrame,
    daily_anomaly: pd.DataFrame,
) -> pd.DataFrame:
    monthly_temperature = daily_temperature.resample("MS").mean()
    monthly_baseline = daily_baseline.resample("MS").mean()
    monthly_anomaly = daily_anomaly.resample("MS").mean()

    full_report = pd.DataFrame({"timestamp": monthly_temperature.index})
    full_report["sample_date"] = full_report["timestamp"].dt.date
    for period_key in monthly_temperature.columns:
        full_report[f"{period_key} temperature_c"] = monthly_temperature[period_key].to_numpy()
        full_report[f"{period_key} climatology_c"] = monthly_baseline[period_key].to_numpy()
        full_report[f"{period_key} anomaly_c"] = monthly_anomaly[period_key].to_numpy()
    return full_report


def _build_active_metadata(
    periods: list[LifePeriod],
    daily_temperature: pd.DataFrame,
    report_start: date,
    report_end: date,
    first_period_history_days: int,
) -> tuple[pd.Series, pd.Series]:
    active_ranges, required_start = _active_ranges(
        periods=periods,
        daily_temperature=daily_temperature,
        report_start=report_start,
        report_end=report_end,
        first_period_history_days=first_period_history_days,
    )
    current_period = pd.Series(index=daily_temperature.index, dtype="string")
    current_place = pd.Series(index=daily_temperature.index, dtype="string")

    for period, period_key, active_days in active_ranges:
        if current_period.loc[active_days].notna().any():
            raise ValueError("The configured periods overlap in the selected timeline.")
        current_period.loc[active_days] = period_key
        current_place.loc[active_days] = period.display_name

    required_slice = slice(pd.Timestamp(required_start), None)
    if current_period.loc[required_slice].isna().any() or current_place.loc[required_slice].isna().any():
        raise ValueError("The configured periods do not cover every day in the selected timeline.")
    return current_period, current_place


def _active_ranges(
    periods: list[LifePeriod],
    daily_temperature: pd.DataFrame,
    report_start: date,
    report_end: date,
    first_period_history_days: int,
) -> tuple[list[tuple[LifePeriod, str, pd.DatetimeIndex]], date]:
    active_ranges: list[tuple[LifePeriod, str, pd.DatetimeIndex]] = []
    required_start_dates: list[date] = []

    for index, period in enumerate(periods):
        period_key = _period_report_key(index, period)
        if period_key not in daily_temperature.columns:
            raise ValueError(f"Missing period column for {period_key}.")

        available = pd.to_numeric(daily_temperature[period_key], errors="coerce").dropna()
        if available.empty:
            raise ValueError(f"No temperature values are available for {period_key}.")

        available_start = available.index.min().date()
        available_end = available.index.max().date()
        effective_start = _effective_period_start(
            period=period,
            index=index,
            first_period_history_days=first_period_history_days,
        )
        active_start = max(report_start, effective_start, available_start)
        active_end = min(report_end, period.end_date, available_end)
        if active_start > active_end:
            continue

        required_start_dates.append(active_start)
        active_ranges.append(
            (
                period,
                period_key,
                pd.date_range(start=active_start, end=active_end, freq="D"),
            )
        )

    if not required_start_dates:
        raise ValueError("The configured periods do not cover every day in the selected timeline.")
    return active_ranges, min(required_start_dates)


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


def _period_report_key(index: int, period: LifePeriod) -> str:
    return f"Period {index + 1}: {period.label}"


def _effective_period_start(
    period: LifePeriod,
    index: int,
    first_period_history_days: int,
) -> date:
    if index == 0 and first_period_history_days > 0:
        return period.start_date - timedelta(days=first_period_history_days)
    return period.start_date
def _join_unique_values(values: pd.Series) -> str:
    ordered_unique = list(dict.fromkeys(str(value) for value in values if str(value).strip()))
    return " + ".join(ordered_unique)


def _days_in_year(year: int) -> int:
    return 366 if calendar.isleap(year) else 365
