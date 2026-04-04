from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from matplotlib.figure import Figure

from mystripes.models import LifePeriod
from mystripes.plotting import export_figure_bytes, render_stripes_figure
from mystripes.processing import (
    aggregate_daily_series_to_stripes,
    build_merged_daily_series,
    build_period_report_tables,
)

AggregationMode = Literal["full_calendar_years", "rolling_365_day"]
RollingSampleMode = Literal["monthly", "fixed_count"]
WatermarkHorizontalAlign = Literal["left", "center", "right"]
WatermarkVerticalAlign = Literal["bottom", "center", "top"]
PeriodIndicatorStyle = Literal["scale_bar", "outward_arrows"]
PeriodIndicatorVerticalAlign = Literal["bottom", "center", "top"]


def build_stripe_data(
    periods: pd.DataFrame | Sequence[LifePeriod | Mapping[str, Any]] | Mapping[str, Any],
    period_data: pd.DataFrame | Sequence[pd.DataFrame | Sequence[Mapping[str, Any]] | Mapping[str, Any]] | Mapping[object, Any],
    *,
    aggregation_mode: AggregationMode = "full_calendar_years",
    rolling_window_end: date | datetime | str | pd.Timestamp | None = None,
    rolling_sample_mode: RollingSampleMode = "monthly",
    rolling_strip_count: int | None = None,
    baseline_start: date | datetime | str | pd.Timestamp | None = None,
    baseline_end: date | datetime | str | pd.Timestamp | None = None,
    baseline_period_data: pd.DataFrame
    | Sequence[pd.DataFrame | Sequence[Mapping[str, Any]] | Mapping[str, Any]]
    | Mapping[object, Any]
    | None = None,
) -> dict[str, Any]:
    """Build stripe-ready data from periods plus monthly temperature frames.

    `periods` accepts a list of dicts, a pandas DataFrame, or LifePeriod objects.
    `period_data` accepts monthly frames aligned to the periods as:
    - a list of pandas DataFrames
    - a dict keyed by period label or index
    - a pandas DataFrame with `period_label` or `period_index`

    Each monthly frame must contain a timestamp-like column and a temperature column.
    Supported column names include `timestamp`/`date`/`time` and
    `temperature_c`/`temperature`/`value`.

    Climatology is always computed per period from day-of-year means before the
    periods are merged into one timeline. Pass `baseline_period_data` together with
    `baseline_start` and `baseline_end` when the climatology reference window lies
    outside the displayed timeline.
    """

    normalized_periods = _coerce_periods(periods)
    normalized_frames = _coerce_period_frames(period_data, normalized_periods)
    normalized_baseline_frames = (
        _coerce_period_frames(baseline_period_data, normalized_periods)
        if baseline_period_data is not None
        else None
    )
    effective_window_end = _coerce_optional_date(rolling_window_end) or max(
        period.end_date for period in normalized_periods
    )
    effective_rolling_crop_start = min(period.start_date for period in normalized_periods)
    first_period_history_days = 364 if aggregation_mode == "rolling_365_day" else 0
    timeline_start = min(period.start_date for period in normalized_periods)
    effective_baseline_start = _coerce_optional_date(baseline_start) or timeline_start
    effective_baseline_end = _coerce_optional_date(baseline_end) or effective_window_end

    result: dict[str, Any] = {
        "periods": normalized_periods,
        "aggregation_mode": aggregation_mode,
        "rolling_window_end": effective_window_end,
        "rolling_sample_mode": rolling_sample_mode,
        "rolling_strip_count": rolling_strip_count,
        "baseline_start": effective_baseline_start,
        "baseline_end": effective_baseline_end,
    }

    daily_series = build_merged_daily_series(
        periods=normalized_periods,
        frames_by_period=normalized_frames,
        report_start=timeline_start - timedelta(days=first_period_history_days),
        report_end=effective_window_end,
        baseline_start=effective_baseline_start,
        baseline_end=effective_baseline_end,
        first_period_history_days=first_period_history_days,
        baseline_frames_by_period=normalized_baseline_frames,
    )
    stripe_frame = aggregate_daily_series_to_stripes(
        daily_series=daily_series,
        aggregation_mode=aggregation_mode,
        rolling_window_end=effective_window_end,
        rolling_crop_start=effective_rolling_crop_start,
        rolling_sample_mode=rolling_sample_mode,
        rolling_strip_count=rolling_strip_count,
    )
    full_report, merged_report = build_period_report_tables(
        periods=normalized_periods,
        frames_by_period=normalized_frames,
        report_start=timeline_start,
        report_end=effective_window_end,
        baseline_start=effective_baseline_start,
        baseline_end=effective_baseline_end,
        baseline_frames_by_period=normalized_baseline_frames,
    )
    result["daily_temperature"] = daily_series[
        ["daily_date", "sample_date", "current_period", "current_place", "temperature_c"]
    ].copy()
    result["daily_series"] = daily_series
    result["full_report"] = full_report
    result["merged_report"] = merged_report
    result["stripe_frame"] = stripe_frame
    return result


def build_period_indicator_specs(
    periods: Sequence[LifePeriod] | pd.DataFrame | Sequence[Mapping[str, Any]] | Mapping[str, Any],
    stripe_frame: pd.DataFrame,
    *,
    included_period_indices: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    normalized_periods = _coerce_periods(periods)
    required_columns = {"window_start", "window_end"}
    if not required_columns.issubset(stripe_frame.columns):
        raise ValueError("stripe_frame must include window_start and window_end columns.")

    included = set(range(len(normalized_periods))) if included_period_indices is None else {
        int(index) for index in included_period_indices if 0 <= int(index) < len(normalized_periods)
    }
    if not included:
        return []

    windows: list[tuple[date, date]] = []
    for row in stripe_frame[["window_start", "window_end"]].itertuples(index=False):
        windows.append((_coerce_required_date(row.window_start, "window_start"), _coerce_required_date(row.window_end, "window_end")))
    if not windows:
        return []

    dominant_period_by_window: list[int | None] = []
    overlapping_periods_by_window: list[set[int]] = []
    for window_start, window_end in windows:
        best_period_index: int | None = None
        best_overlap_days = 0
        overlapping_indices: set[int] = set()
        for period_index, period in enumerate(normalized_periods):
            overlap_start = max(window_start, period.start_date)
            overlap_end = min(window_end, period.end_date)
            if overlap_start > overlap_end:
                continue
            overlap_days = (overlap_end - overlap_start).days + 1
            overlapping_indices.add(period_index)
            if overlap_days > best_overlap_days:
                best_overlap_days = overlap_days
                best_period_index = period_index
        dominant_period_by_window.append(best_period_index)
        overlapping_periods_by_window.append(overlapping_indices)

    indicator_specs: list[dict[str, Any]] = []
    window_count = len(windows)
    for period_index, period in enumerate(normalized_periods):
        if period_index not in included:
            continue
        assigned_indices = [
            window_index
            for window_index, dominant_period_index in enumerate(dominant_period_by_window)
            if dominant_period_index == period_index
        ]
        if assigned_indices:
            start_index = assigned_indices[0]
            end_index = assigned_indices[-1]
        else:
            overlapping_indices = [
                window_index
                for window_index, overlapping_period_indices in enumerate(overlapping_periods_by_window)
                if period_index in overlapping_period_indices
            ]
            if not overlapping_indices:
                continue
            start_index = overlapping_indices[0]
            end_index = overlapping_indices[-1]

        indicator_specs.append(
            {
                "period_index": period_index,
                "label": period.label,
                "start_fraction": start_index / window_count,
                "end_fraction": (end_index + 1) / window_count,
            }
        )

    return indicator_specs


def plot_stripes(
    stripe_data: Mapping[str, Any] | pd.DataFrame,
    *,
    width_px: int = 1800,
    height_px: int = 260,
    dpi: int = 200,
    output_path: str | Path | None = None,
    fmt: Literal["png", "svg", "pdf"] | None = None,
    watermark_text: str | None = None,
    watermark_horizontal_align: WatermarkHorizontalAlign = "center",
    watermark_vertical_align: WatermarkVerticalAlign = "center",
    watermark_color: str = "#ffffff",
    watermark_opacity: float = 0.35,
    watermark_shadow: bool = False,
    watermark_max_width_ratio: float = 0.8,
    watermark_max_height_ratio: float = 0.8,
    period_indicators: Sequence[Mapping[str, Any]] | None = None,
    period_indicator_style: PeriodIndicatorStyle = "scale_bar",
    period_indicator_vertical_align: PeriodIndicatorVerticalAlign = "bottom",
    period_indicator_color: str = "#ffffff",
    period_indicator_height_ratio: float = 0.2,
) -> Figure:
    """Plot stripes from the bundle returned by `build_stripe_data`.

    If `output_path` is provided, the figure is also saved there. Provide
    `watermark_text` and the other watermark arguments to overlay a fitted
    text watermark over the stripes graphic. Provide `period_indicators`
    together with the period-indicator style arguments to overlay approximate
    period-range labels on top of the stripes. Use
    `period_indicator_height_ratio` to scale how much of the stripe height
    the indicator overlay should occupy.
    """

    stripe_frame = _extract_stripe_frame(stripe_data)
    figure = render_stripes_figure(
        anomalies=stripe_frame["anomaly_c"].astype(float).tolist(),
        width_inches=width_px / dpi,
        height_inches=height_px / dpi,
        watermark_text=watermark_text,
        watermark_horizontal_align=watermark_horizontal_align,
        watermark_vertical_align=watermark_vertical_align,
        watermark_color=watermark_color,
        watermark_opacity=watermark_opacity,
        watermark_shadow=watermark_shadow,
        watermark_max_width_ratio=watermark_max_width_ratio,
        watermark_max_height_ratio=watermark_max_height_ratio,
        period_indicators=period_indicators,
        period_indicator_style=period_indicator_style,
        period_indicator_vertical_align=period_indicator_vertical_align,
        period_indicator_color=period_indicator_color,
        period_indicator_height_ratio=period_indicator_height_ratio,
    )

    if output_path is not None:
        path = Path(output_path)
        output_format = (fmt or path.suffix.lstrip(".") or "png").lower()
        if output_format not in {"png", "svg", "pdf"}:
            raise ValueError(f"Unsupported output format: {output_format}")
        payload = export_figure_bytes(
            figure=figure,
            fmt=output_format,
            png_dpi=dpi,
        )
        path.write_bytes(payload)

    return figure


def _coerce_periods(
    periods: pd.DataFrame | Sequence[LifePeriod | Mapping[str, Any]] | Mapping[str, Any],
) -> list[LifePeriod]:
    if isinstance(periods, pd.DataFrame):
        raw_periods: Sequence[LifePeriod | Mapping[str, Any]] = periods.to_dict(orient="records")
    elif isinstance(periods, Mapping):
        raw_periods = [periods]
    else:
        raw_periods = periods

    normalized: list[LifePeriod] = []
    for index, item in enumerate(raw_periods):
        if isinstance(item, LifePeriod):
            normalized.append(item)
            continue
        if not isinstance(item, Mapping):
            raise TypeError("Periods must be LifePeriod objects, mappings, or a pandas DataFrame.")
        normalized.append(_life_period_from_mapping(item, index))
    if not normalized:
        raise ValueError("At least one period is required.")
    return normalized


def _life_period_from_mapping(item: Mapping[str, Any], index: int) -> LifePeriod:
    start_date = _coerce_required_date(_first_value(item, "start_date", "start"), "start_date")
    end_date = _coerce_required_date(_first_value(item, "end_date", "end"), "end_date")
    latitude = _coerce_required_float(
        _first_value(item, "latitude", "lat", "latitude_text"),
        "latitude",
    )
    longitude = _coerce_required_float(
        _first_value(item, "longitude", "lon", "longitude_text"),
        "longitude",
    )
    label = _first_value(item, "label", "custom_label")
    place_query = _first_value(item, "place_query", "place", "location", "name")
    resolved_name = _first_value(item, "resolved_name", "display_name", "place_name")

    return LifePeriod(
        label=str(label or resolved_name or place_query or f"Period {index + 1}"),
        place_query=str(place_query or ""),
        resolved_name=str(resolved_name or ""),
        start_date=start_date,
        end_date=end_date,
        latitude=latitude,
        longitude=longitude,
        boundary_geojson=item.get("boundary_geojson") or item.get("geojson"),
        bounding_box=_coerce_bounding_box(item.get("bounding_box")),
    )


def _coerce_period_frames(
    period_data: pd.DataFrame | Sequence[pd.DataFrame | Sequence[Mapping[str, Any]] | Mapping[str, Any]] | Mapping[object, Any],
    periods: list[LifePeriod],
) -> list[pd.DataFrame]:
    if isinstance(period_data, pd.DataFrame):
        return _frames_from_dataframe(period_data, periods)

    if isinstance(period_data, Mapping):
        return _frames_from_mapping(period_data, periods)

    if not isinstance(period_data, Sequence) or isinstance(period_data, str | bytes):
        raise TypeError("Period data must be a pandas DataFrame, sequence, or mapping.")
    if len(period_data) != len(periods):
        raise ValueError("When period data is a sequence, it must match the number of periods.")
    return [_coerce_temperature_frame(frame_like) for frame_like in period_data]


def _frames_from_dataframe(frame: pd.DataFrame, periods: list[LifePeriod]) -> list[pd.DataFrame]:
    if len(periods) == 1 and "period_label" not in frame.columns and "period_index" not in frame.columns:
        return [_coerce_temperature_frame(frame)]

    if "period_index" in frame.columns:
        return [
            _coerce_temperature_frame(frame.loc[frame["period_index"] == index].drop(columns=["period_index"]))
            for index in range(len(periods))
        ]

    label_column = "period_label" if "period_label" in frame.columns else "label" if "label" in frame.columns else None
    if label_column is None:
        raise ValueError(
            "A combined period DataFrame needs a `period_index`, `period_label`, or `label` column."
        )

    labels = [period.label for period in periods]
    if len(set(labels)) != len(labels):
        raise ValueError("Period labels must be unique when using a combined period DataFrame.")
    return [
        _coerce_temperature_frame(
            frame.loc[frame[label_column] == period.label].drop(columns=[label_column])
        )
        for period in periods
    ]


def _frames_from_mapping(mapping: Mapping[object, Any], periods: list[LifePeriod]) -> list[pd.DataFrame]:
    labels = [period.label for period in periods]
    unique_labels = len(set(labels)) == len(labels)
    frames: list[pd.DataFrame] = []

    for index, period in enumerate(periods):
        candidates: list[object] = [index, str(index), f"period_{index + 1}"]
        if unique_labels:
            candidates.insert(0, period.label)

        frame_like = None
        for candidate in candidates:
            if candidate in mapping:
                frame_like = mapping[candidate]
                break
        if frame_like is None:
            raise KeyError(
                f"Could not find monthly data for period '{period.label}'. "
                "Use period labels or 0-based indices as mapping keys."
            )
        frames.append(_coerce_temperature_frame(frame_like))

    return frames


def _coerce_temperature_frame(frame_like: pd.DataFrame | Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> pd.DataFrame:
    if isinstance(frame_like, pd.DataFrame):
        frame = frame_like.copy()
    elif isinstance(frame_like, Mapping):
        frame = pd.DataFrame(frame_like)
    else:
        frame = pd.DataFrame(frame_like)

    timestamp_column = _first_present_column(frame, "timestamp", "date", "time")
    if timestamp_column is None:
        raise ValueError("Each monthly data frame needs a `timestamp`, `date`, or `time` column.")
    temperature_column = _first_present_column(frame, "temperature_c", "temperature", "value")
    if temperature_column is None:
        raise ValueError(
            "Each monthly data frame needs a `temperature_c`, `temperature`, or `value` column."
        )

    normalized = frame.rename(
        columns={
            timestamp_column: "timestamp",
            temperature_column: "temperature_c",
        }
    ).copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
    normalized["temperature_c"] = pd.to_numeric(normalized["temperature_c"])
    if "sample_days" in normalized.columns:
        normalized["sample_days"] = pd.to_numeric(normalized["sample_days"])
    return normalized.sort_values("timestamp").reset_index(drop=True)


def _extract_stripe_frame(stripe_data: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(stripe_data, pd.DataFrame):
        stripe_frame = stripe_data.copy()
    else:
        stripe_frame = stripe_data.get("stripe_frame")
        if not isinstance(stripe_frame, pd.DataFrame):
            raise TypeError("Stripe data must be a DataFrame or a mapping containing `stripe_frame`.")

    if "anomaly_c" not in stripe_frame.columns:
        raise ValueError("Stripe data needs an `anomaly_c` column.")
    return stripe_frame


def _first_value(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _first_present_column(frame: pd.DataFrame, *columns: str) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _coerce_required_date(value: Any, field_name: str) -> date:
    parsed = _coerce_optional_date(value)
    if parsed is None:
        raise ValueError(f"`{field_name}` is required for each period.")
    return parsed


def _coerce_optional_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    return pd.Timestamp(value).date()


def _coerce_required_float(value: Any, field_name: str) -> float:
    if value is None or value == "":
        raise ValueError(f"`{field_name}` is required for each period.")
    return float(value)


def _coerce_bounding_box(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 4:
        return tuple(float(item) for item in value)
    if isinstance(value, list) and len(value) == 4:
        return tuple(float(item) for item in value)
    raise ValueError("`bounding_box` must contain four numeric values when provided.")

__all__ = ["build_stripe_data", "plot_stripes"]
