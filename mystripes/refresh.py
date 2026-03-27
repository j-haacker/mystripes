from __future__ import annotations

from datetime import date


def latest_data_end_that_can_change_stripes(
    report_start: date,
    analysis_end: date,
    aggregation_mode: str,
    reference_period_mode: str,
) -> date:
    if aggregation_mode == "rolling_365_day":
        return analysis_end
    if aggregation_mode != "full_calendar_years":
        raise ValueError(f"Unsupported aggregation mode: {aggregation_mode}")
    if reference_period_mode == "story_line_period":
        return analysis_end
    if reference_period_mode != "climate_normal_1961_2010":
        raise ValueError(f"Unsupported climatology reference period: {reference_period_mode}")

    candidate = _last_complete_calendar_year_end(analysis_end)
    if candidate < report_start:
        return analysis_end
    return candidate


def _last_complete_calendar_year_end(value: date) -> date:
    if value.month == 12 and value.day == 31:
        return value
    return date(value.year - 1, 12, 31)
