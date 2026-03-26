from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from matplotlib.figure import Figure

from mystrips.models import LifePeriod
from mystrips.plotting import export_figure_bytes, render_stripes_figure
from mystrips.processing import (
    build_location_baseline_stripe_frame,
    build_stripe_frame,
    calculate_life_period_baseline,
    calculate_series_mean_temperature,
    combine_period_frames,
)

BaselineMode = Literal["timeline_mean", "location_reference"]
AggregationMode = Literal["full_calendar_years", "rolling_365_day"]


def build_stripe_data(
    periods: pd.DataFrame | Sequence[LifePeriod | Mapping[str, Any]] | Mapping[str, Any],
    period_data: pd.DataFrame | Sequence[pd.DataFrame | Sequence[Mapping[str, Any]] | Mapping[str, Any]] | Mapping[object, Any],
    *,
    baseline_mode: BaselineMode = "timeline_mean",
    aggregation_mode: AggregationMode = "full_calendar_years",
    rolling_window_end: date | datetime | str | pd.Timestamp | None = None,
    reference_data: pd.DataFrame | Sequence[pd.DataFrame | Sequence[Mapping[str, Any]] | Mapping[str, Any]] | Mapping[object, Any] | None = None,
    baseline_by_location: Mapping[object, float] | None = None,
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
    """

    normalized_periods = _coerce_periods(periods)
    normalized_frames = _coerce_period_frames(period_data, normalized_periods)
    effective_window_end = _coerce_optional_date(rolling_window_end) or max(
        period.end_date for period in normalized_periods
    )

    combined, yearly = combine_period_frames(
        normalized_periods,
        normalized_frames,
        aggregation_mode=aggregation_mode,
        rolling_window_end=effective_window_end,
    )

    result: dict[str, Any] = {
        "periods": normalized_periods,
        "combined": combined,
        "yearly": yearly,
        "aggregation_mode": aggregation_mode,
        "rolling_window_end": effective_window_end,
        "baseline_mode": baseline_mode,
    }

    if baseline_mode == "timeline_mean":
        baseline_c = calculate_life_period_baseline(yearly)
        stripe_frame = build_stripe_frame(yearly, baseline_c)
        result["baseline_c"] = baseline_c
        result["stripe_frame"] = stripe_frame
        return result

    if baseline_mode != "location_reference":
        raise ValueError(f"Unsupported baseline mode: {baseline_mode}")

    normalized_baseline_by_location = _normalize_baseline_by_location(
        baseline_by_location,
        normalized_periods,
    )
    if normalized_baseline_by_location is None:
        reference_frames = _coerce_period_frames(
            reference_data if reference_data is not None else period_data,
            normalized_periods,
        )
        normalized_baseline_by_location = _calculate_baseline_by_location(
            normalized_periods,
            reference_frames,
        )

    stripe_frame = build_location_baseline_stripe_frame(
        combined=combined,
        baseline_by_location=normalized_baseline_by_location,
        aggregation_mode=aggregation_mode,
        rolling_window_end=effective_window_end,
    )
    result["baseline_by_location"] = normalized_baseline_by_location
    result["stripe_frame"] = stripe_frame
    return result


def plot_stripes(
    stripe_data: Mapping[str, Any] | pd.DataFrame,
    *,
    width_px: int = 1800,
    height_px: int = 260,
    dpi: int = 200,
    transparent_background: bool = False,
    output_path: str | Path | None = None,
    fmt: Literal["png", "svg", "pdf"] | None = None,
) -> Figure:
    """Plot stripes from the bundle returned by `build_stripe_data`.

    If `output_path` is provided, the figure is also saved there.
    """

    stripe_frame = _extract_stripe_frame(stripe_data)
    figure = render_stripes_figure(
        anomalies=stripe_frame["anomaly_c"].astype(float).tolist(),
        width_inches=width_px / dpi,
        height_inches=height_px / dpi,
        transparent_background=transparent_background,
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
            transparent_background=transparent_background,
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


def _normalize_baseline_by_location(
    baseline_by_location: Mapping[object, float] | None,
    periods: list[LifePeriod],
) -> dict[str, float] | None:
    if baseline_by_location is None:
        return None

    location_keys = {period.location_key for period in periods}
    labels_to_location = _unique_period_label_mapping(periods)
    normalized: dict[str, float] = {}

    for raw_key, raw_value in baseline_by_location.items():
        if raw_key in location_keys:
            normalized[str(raw_key)] = float(raw_value)
            continue
        if raw_key in labels_to_location:
            normalized[labels_to_location[str(raw_key)]] = float(raw_value)
            continue
        if isinstance(raw_key, int) and 0 <= raw_key < len(periods):
            normalized[periods[raw_key].location_key] = float(raw_value)
            continue
        raise KeyError(
            f"Unknown baseline key {raw_key!r}. Use a period label, period index, or location_key."
        )

    return normalized


def _calculate_baseline_by_location(
    periods: list[LifePeriod],
    reference_frames: list[pd.DataFrame],
) -> dict[str, float]:
    grouped_frames: dict[str, list[pd.DataFrame]] = {}
    for period, frame in zip(periods, reference_frames, strict=True):
        grouped_frames.setdefault(period.location_key, []).append(frame)

    baselines: dict[str, float] = {}
    for location_key, frames in grouped_frames.items():
        baselines[location_key] = calculate_series_mean_temperature(
            pd.concat(frames, ignore_index=True)
        )
    return baselines


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


def _unique_period_label_mapping(periods: list[LifePeriod]) -> dict[str, str]:
    labels_to_location: dict[str, str] = {}
    for period in periods:
        if period.label in labels_to_location and labels_to_location[period.label] != period.location_key:
            raise ValueError(
                "Period labels must be unique when using label-keyed baseline mappings."
            )
        labels_to_location[period.label] = period.location_key
    return labels_to_location


__all__ = ["build_stripe_data", "plot_stripes"]
