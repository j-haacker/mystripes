from __future__ import annotations

from datetime import date, timedelta
from html import escape
from time import perf_counter
from uuid import uuid4
from pprint import pformat

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from mystripes.api import build_period_indicator_specs
from mystripes.cds import (
    CDSCredentialsMissingError,
    CDSRequestError,
    DEFAULT_CDSAPI_URL,
    clear_local_cds_config,
    load_local_cds_config,
    resolve_cds_config,
    save_local_cds_config,
)
from mystripes.climate_stack import (
    build_climate_batch_plan,
    build_location_climate_requests,
    climate_stack_note,
    estimate_climate_downloads,
    fetch_saved_climate_series_batch,
    get_climate_dataset_window,
    preflight_download_message,
)
from mystripes.cookie_consent import (
    COOKIE_CONSENT_COOKIE_NAME,
    COOKIE_CONSENT_MAX_AGE_SECONDS,
    build_cookie_consent_payload,
    cookie_consent_choice,
    decode_cookie_consent_value,
    encode_cookie_consent_value,
    optional_cookie_consent_granted,
)
from mystripes.geocoding import search_places
from mystripes.gistemp import (
    NASA_GISTEMP_GLOBAL_MEAN_URL,
    average_global_warming_for_period,
    load_global_mean_estimates,
)
from mystripes.models import CDSConfig
from mystripes.notices import (
    CONTRIBUTING_GUIDE_URL,
    PROJECT_ISSUES_URL,
    PROJECT_REPOSITORY_URL,
    ERA5_MONTHLY_DATASET_DOI,
    ERA5_MONTHLY_DATASET_NAME,
    ERA5_REFERENCE_CITATION,
    ERA5_MONTHLY_DATASET_URL,
    ERA5_LAND_MONTHLY_DATASET_DOI,
    ERA5_LAND_REFERENCE_CITATION,
    ERA5_LAND_MONTHLY_DATASET_NAME,
    ERA5_LAND_MONTHLY_DATASET_URL,
    GENERATED_GRAPHICS_CC0_NOTICE,
    NASA_GISTEMP_REFERENCE_CITATION,
    NASA_GISTEMP_SHORT_SOURCE_CREDIT,
    NASA_GISTEMP_SOURCE_CREDIT,
    NASA_GISTEMP_WEB_CITATION_GUIDANCE,
    NASA_GISTEMP_WEB_CITATION_TEMPLATE,
    SHOW_YOUR_STRIPES_CREDIT,
    SHOW_YOUR_STRIPES_URL,
    SOFTWARE_MIT_NOTICE,
    TWCR_ACKNOWLEDGEMENT_TEXT,
    TWCR_FOUNDATIONAL_REFERENCE_CITATION,
    TWCR_MONTHLY_DATASET_NAME,
    TWCR_MONTHLY_DATASET_URL,
    TWCR_REFERENCE_CITATION,
    copernicus_credit_notice,
)
from mystripes.plotting import export_figure_bytes, render_stripes_figure
from mystripes.refresh import latest_data_end_that_can_change_stripes
from mystripes.processing import (
    aggregate_daily_series_to_stripes,
    build_merged_daily_series,
    build_period_report_tables,
    build_periods_from_entries,
)
from mystripes.storyline_cookie_component import sync_storyline_cookie_store
from mystripes.storylines import (
    LOCAL_STORYLINES_PATH,
    STORYLINE_COOKIE_MAX_AGE_SECONDS,
    STORYLINE_COOKIE_PREFIX,
    encode_storyline_cookie_value,
    load_cookie_storylines,
    load_local_storylines,
    remove_local_storyline,
    save_local_storyline,
    serialize_storyline_payload,
    storyline_cookie_name,
    storyline_storage_backend_from_host,
)

st.set_page_config(
    page_title="MyStripes",
    page_icon="||",
    layout="wide",
)


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def cached_dataset_window():
    return get_climate_dataset_window()


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def cached_search_places(query: str, geoapify_api_key: str):
    return search_places(query, geoapify_api_key=geoapify_api_key or None)


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def cached_gistemp_global_mean_estimates() -> pd.DataFrame:
    return load_global_mean_estimates()


def _storyline_storage_backend() -> str:
    headers = getattr(st.context, "headers", None)
    host = ""
    if headers is not None:
        host = str(headers.get("host") or headers.get("Host") or "")
    return storyline_storage_backend_from_host(host)


def _cookie_storyline_values() -> dict[str, str]:
    raw_values = st.session_state.get("storyline_cookie_values", {})
    if not isinstance(raw_values, dict):
        return {}
    return {str(name): str(value) for name, value in raw_values.items()}


def _cookie_consent_payload_from_session() -> dict[str, object] | None:
    raw_payload = st.session_state.get("cookie_consent_payload")
    return dict(raw_payload) if isinstance(raw_payload, dict) else None


def _cloud_cookie_consent_choice() -> str | None:
    return cookie_consent_choice(_cookie_consent_payload_from_session())


def _cloud_storyline_storage_enabled() -> bool:
    return optional_cookie_consent_granted(_cookie_consent_payload_from_session())


def _load_saved_storylines(storage_backend: str) -> dict[str, dict[str, object]]:
    if storage_backend == "cookie":
        if not _cloud_storyline_storage_enabled():
            return {}
        return load_cookie_storylines(_cookie_storyline_values())
    return load_local_storylines()


def _set_storyline_feedback(kind: str, message: str) -> None:
    st.session_state.storyline_feedback = {"kind": kind, "message": message}


def _render_storyline_feedback(sidebar) -> None:
    feedback = st.session_state.pop("storyline_feedback", None)
    if not isinstance(feedback, dict):
        return
    message = str(feedback.get("message", "")).strip()
    if not message:
        return
    kind = str(feedback.get("kind", "info"))
    if kind == "success":
        sidebar.success(message)
    elif kind == "error":
        sidebar.error(message)
    else:
        sidebar.info(message)


def _queue_cookie_storyline_operation(
    *,
    action: str,
    storyline_name: str,
    cookie_name: str,
    cookie_value: str | None = None,
) -> None:
    if action == "save":
        if cookie_value is None:
            raise ValueError("Missing cookie value for save operation.")
        cookie_operations = [
            {
                "action": "set",
                "cookie_name": cookie_name,
                "cookie_value": cookie_value,
                "max_age_seconds": STORYLINE_COOKIE_MAX_AGE_SECONDS,
            }
        ]
    elif action == "remove":
        cookie_operations = [{"action": "remove", "cookie_name": cookie_name}]
    else:
        raise ValueError(f"Unsupported storyline cookie action: {action}")

    st.session_state.pending_cookie_store_operation = {
        "id": uuid4().hex,
        "kind": "storyline",
        "action": action,
        "storyline_name": storyline_name,
        "cookie_operations": cookie_operations,
    }


def _queue_cookie_consent_operation(choice: str) -> None:
    payload = build_cookie_consent_payload(choice)
    cookie_operations: list[dict[str, object]] = [
        {
            "action": "set",
            "cookie_name": COOKIE_CONSENT_COOKIE_NAME,
            "cookie_value": encode_cookie_consent_value(payload),
            "max_age_seconds": COOKIE_CONSENT_MAX_AGE_SECONDS,
        }
    ]
    if payload["choice"] == "rejected":
        cookie_operations.append(
            {
                "action": "remove_prefix",
                "cookie_prefix": STORYLINE_COOKIE_PREFIX,
            }
        )

    st.session_state.pending_cookie_store_operation = {
        "id": uuid4().hex,
        "kind": "consent",
        "choice": payload["choice"],
        "cookie_operations": cookie_operations,
    }


def _queue_storyline_widget_update(key: str, value: object) -> None:
    st.session_state[f"_pending_{key}"] = "" if value is None else str(value)


def _apply_pending_storyline_widget_updates() -> None:
    for key in ("storyline_name", "saved_storyline_name"):
        pending_key = f"_pending_{key}"
        if pending_key in st.session_state:
            st.session_state[key] = st.session_state.pop(pending_key)


def _sync_cookie_store() -> None:
    previous_choice = _cloud_cookie_consent_choice()
    monitor_storyline_cookies = previous_choice == "accepted"
    result = sync_storyline_cookie_store(
        monitored_cookie_names=[COOKIE_CONSENT_COOKIE_NAME],
        monitored_cookie_prefixes=[STORYLINE_COOKIE_PREFIX] if monitor_storyline_cookies else [],
        operation=st.session_state.get("pending_cookie_store_operation"),
        key="storyline_cookie_store",
    )
    if not isinstance(result, dict):
        return

    raw_cookies = result.get("cookies", {})
    normalized_cookies = {}
    if isinstance(raw_cookies, dict):
        normalized_cookies = {
            str(name): str(value)
            for name, value in raw_cookies.items()
        }

    st.session_state.storyline_cookie_values = {
        name: value
        for name, value in normalized_cookies.items()
        if name.startswith(STORYLINE_COOKIE_PREFIX)
    }

    consent_cookie_value = normalized_cookies.get(COOKIE_CONSENT_COOKIE_NAME, "")
    if consent_cookie_value:
        try:
            st.session_state.cookie_consent_payload = decode_cookie_consent_value(consent_cookie_value)
        except ValueError:
            st.session_state.cookie_consent_payload = None
    else:
        st.session_state.cookie_consent_payload = None

    pending_operation = st.session_state.get("pending_cookie_store_operation")
    current_choice = _cloud_cookie_consent_choice()
    if not isinstance(pending_operation, dict):
        if current_choice == "accepted" and not monitor_storyline_cookies:
            st.rerun()
        return

    completed_operation_id = str(result.get("completed_operation_id") or "")
    if completed_operation_id != str(pending_operation.get("id") or ""):
        return

    st.session_state.pending_cookie_store_operation = None
    error_message = str(result.get("error") or "").strip()
    if error_message:
        _set_storyline_feedback("error", error_message)
        st.rerun()

    operation_kind = str(pending_operation.get("kind") or "")
    if operation_kind == "consent":
        if current_choice == "accepted":
            _set_storyline_feedback("success", "Optional convenience cookies enabled in this browser.")
        else:
            st.session_state.storyline_cookie_values = {}
            _queue_storyline_widget_update("saved_storyline_name", "")
            _set_storyline_feedback(
                "info",
                "Optional convenience cookies disabled. Any saved cloud story lines were removed from this browser.",
            )
        st.rerun()

    storyline_name = str(pending_operation.get("storyline_name") or "").strip()
    action = str(pending_operation.get("action") or "")
    if action == "save":
        _queue_storyline_widget_update("storyline_name", storyline_name)
        _queue_storyline_widget_update("saved_storyline_name", storyline_name)
        _set_storyline_feedback("success", f"Saved story line `{storyline_name}`.")
    elif action == "remove":
        if st.session_state.storyline_name == storyline_name:
            _queue_storyline_widget_update("storyline_name", "")
        _queue_storyline_widget_update("saved_storyline_name", "")
        _set_storyline_feedback("success", f"Removed story line `{storyline_name}`.")
    st.rerun()


def _current_storyline_period_entries() -> list[dict[str, object]]:
    synchronized_entries: list[dict[str, object]] = []
    entries = st.session_state.period_entries
    for index, base_entry in enumerate(entries):
        entry = dict(base_entry)
        place_query_key = f"place_query_{index}"
        latitude_key = f"latitude_{index}"
        longitude_key = f"longitude_{index}"
        end_date_key = f"end_date_{index}"
        indicator_label_key = f"indicator_label_{index}"
        show_indicator_key = f"show_indicator_{index}"
        if place_query_key in st.session_state:
            entry["place_query"] = str(st.session_state[place_query_key])
        if latitude_key in st.session_state:
            entry["latitude_text"] = str(st.session_state[latitude_key])
        if longitude_key in st.session_state:
            entry["longitude_text"] = str(st.session_state[longitude_key])
        if indicator_label_key in st.session_state:
            entry["indicator_label"] = str(st.session_state[indicator_label_key])
        if show_indicator_key in st.session_state:
            entry["show_indicator"] = bool(st.session_state[show_indicator_key])
        if index < len(entries) - 1 and end_date_key in st.session_state:
            entry["end_date"] = st.session_state[end_date_key]
        elif index == len(entries) - 1:
            entry["end_date"] = None
        synchronized_entries.append(entry)
    return synchronized_entries


def _clear_period_widget_state() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith((
            "_pending_geocode_search_",
            "custom_label_",
            "end_date_",
            "geocode_choice_",
            "geocode_results_",
            "indicator_label_",
            "latitude_",
            "longitude_",
            "place_query_",
            "show_indicator_",
        )):
            del st.session_state[key]


def _apply_storyline_to_session(payload: dict[str, object], analysis_end: date) -> None:
    _clear_period_widget_state()
    loaded_birth_date = payload["birth_date"]
    if not isinstance(loaded_birth_date, date):
        loaded_birth_date = date.fromisoformat(str(loaded_birth_date))
    clamped_birth_date = min(max(loaded_birth_date, date(1900, 1, 1)), analysis_end)

    raw_entries = payload["period_entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raw_entries = [_blank_entry()]

    entries: list[dict[str, object]] = []
    for raw_entry in raw_entries:
        entry = _blank_entry()
        entry.update(dict(raw_entry))
        indicator_label = str(entry.get("indicator_label", entry.get("custom_label", "")) or "")
        entry["indicator_label"] = indicator_label
        entry["show_indicator"] = bool(entry.get("show_indicator", False) or indicator_label)
        entries.append(entry)

    st.session_state.birth_date = clamped_birth_date
    st.session_state.period_entries = entries
    for index, entry in enumerate(entries):
        st.session_state[f"place_query_{index}"] = str(entry.get("place_query", "") or "")
        st.session_state[f"latitude_{index}"] = str(entry.get("latitude_text", "") or "")
        st.session_state[f"longitude_{index}"] = str(entry.get("longitude_text", "") or "")
        st.session_state[f"indicator_label_{index}"] = str(entry.get("indicator_label", "") or "")
        st.session_state[f"show_indicator_{index}"] = bool(entry.get("show_indicator", False))
        if index < len(entries) - 1 and entry.get("end_date") is not None:
            st.session_state[f"end_date_{index}"] = entry["end_date"]


def _format_progress_duration(total_seconds: float) -> str:
    rounded_seconds = max(0, int(round(total_seconds)))
    minutes, seconds = divmod(rounded_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _format_progress_rate(rate_per_minute: float, unit_label: str) -> str:
    if rate_per_minute >= 10:
        value_text = f"{rate_per_minute:.0f}"
    elif rate_per_minute >= 1:
        value_text = f"{rate_per_minute:.1f}"
    else:
        value_text = f"{rate_per_minute:.2f}"
    return f"{value_text} {unit_label}/min"


def _request_batch_label(event: dict[str, object]) -> str:
    parts: list[str] = []
    request_index = int(event.get("request_index") or 0)
    request_count = int(event.get("request_count") or 0)
    if request_index and request_count:
        parts.append(f"batch {request_index}/{request_count}")

    request_year_start = str(event.get("request_year_start") or "")
    request_year_end = str(event.get("request_year_end") or "")
    if request_year_start and request_year_end:
        if request_year_start == request_year_end:
            parts.append(request_year_start)
        else:
            parts.append(f"{request_year_start}-{request_year_end}")

    month_count = int(event.get("month_count") or 0)
    if month_count:
        noun = "month" if month_count == 1 else "months"
        parts.append(f"{month_count} {noun}")

    return ", ".join(parts)


def _describe_temperature_fetch_event(event: dict[str, object]) -> str:
    stage = str(event.get("stage") or "")
    purpose = str(event.get("purpose") or "source")
    range_index = int(event.get("range_index") or 0)
    range_count = int(event.get("range_count") or 0)
    range_start = str(event.get("range_start") or "")
    range_end = str(event.get("range_end") or "")
    dataset_label = str(event.get("dataset_label") or event.get("source_label") or "climate data")
    request_origin = str(event.get("request_origin") or "")
    location_label = str(event.get("location_label") or "")
    cache_scope = str(event.get("cache_scope") or "")

    if stage == "batch_plan":
        return (
            f"Planned {int(event.get('total_shared_tasks') or 0)} shared source tasks for "
            f"{int(event.get('total_locations') or 0)} locations."
        )
    if stage == "shared_task_started":
        source_label = str(event.get("source_label") or dataset_label)
        return f"Starting shared {source_label} preparation."
    if stage == "shared_task_finished":
        source_label = str(event.get("source_label") or dataset_label)
        return f"Finished shared {source_label} preparation."
    if stage == "location_started":
        return f"Building the combined climate timeline for {location_label or 'one location'}."
    if stage == "location_finished":
        return f"{location_label or 'One location'} is ready."

    if stage == "timeline_cache_hit":
        return f"Using the saved {dataset_label} timeline from local cache."
    if stage == "timeline_fetch_plan":
        missing_range_count = int(event.get("missing_range_count") or 0)
        noun = "segment" if missing_range_count == 1 else "segments"
        if bool(event.get("has_cached_data")):
            return f"Refreshing {missing_range_count} missing {dataset_label} timeline {noun}."
        return f"Preparing {missing_range_count} {dataset_label} timeline {noun}."
    if stage == "missing_range_started":
        return f"Timeline segment {range_index}/{range_count}: {range_start} to {range_end}."
    if stage == "request_cache_hit":
        if cache_scope == "shared_grid":
            return f"Using a saved shared {dataset_label} grid request."
        if cache_scope == "shared_grid_window":
            return f"Using a saved shared {dataset_label} bbox window."
        if purpose == "calibration":
            return f"Using a saved {dataset_label} calibration window."
        return f"Using a saved {dataset_label} request chunk for {range_start} to {range_end}."
    if stage == "request_plan":
        request_count = int(event.get("request_count") or 0)
        if str(event.get("dataset") or "").startswith("reanalysis-era5"):
            noun = "batch" if request_count == 1 else "batches"
        else:
            noun = "bbox request" if request_count == 1 else "bbox requests"
        if purpose == "calibration":
            return f"Preparing {request_count} {dataset_label} calibration {noun}."
        return f"Need {request_count} {dataset_label} request {noun} for this timeline segment."
    if stage == "request_started":
        batch_label = _request_batch_label(event)
        request_scope = str(event.get("request_scope") or "")
        if request_scope == "bbox_window":
            batch_label = batch_label or "bbox window"
        if request_origin == "local_cache":
            return (
                f"Reading saved {dataset_label} {batch_label}."
                if batch_label
                else f"Reading saved {dataset_label} data."
            )
        if purpose == "calibration":
            return (
                f"Downloading {dataset_label} calibration {batch_label}."
                if batch_label
                else f"Downloading {dataset_label} calibration data."
            )
        return (
            f"Downloading {dataset_label} {batch_label}."
            if batch_label
            else f"Downloading {dataset_label} data."
        )
    if stage == "request_finished":
        batch_label = _request_batch_label(event)
        request_scope = str(event.get("request_scope") or "")
        if request_scope == "bbox_window":
            batch_label = batch_label or "bbox window"
        if request_origin == "local_cache":
            return (
                f"Finished reading saved {dataset_label} {batch_label}."
                if batch_label
                else f"Finished reading saved {dataset_label} data."
            )
        if purpose == "calibration":
            return (
                f"Finished {dataset_label} calibration {batch_label}."
                if batch_label
                else f"Finished a {dataset_label} calibration step."
            )
        return (
            f"Finished {dataset_label} {batch_label}."
            if batch_label
            else f"Finished a {dataset_label} request step."
        )
    if stage == "request_failed":
        message = str(event.get("message") or "")
        return message or "A CDS request failed."
    if stage == "request_recovery":
        message = str(event.get("message") or "")
        return message or f"Retrying {dataset_label} after a cache read problem."
    if stage == "point_fetch_completed":
        if purpose == "calibration":
            return f"Prepared the {dataset_label} calibration monthly series."
        return f"Prepared monthly {dataset_label} data for the current timeline segment."
    if stage == "missing_range_finished":
        return f"Finished timeline segment {range_index}/{range_count}: {range_start} to {range_end}."
    if stage == "timeline_fetch_completed":
        return "Saved the refreshed timeline to the local cache."
    if stage == "source_started":
        source_label = str(event.get("source_label") or "Climate source")
        return f"Switching to {source_label} for {event.get('source_start')} to {event.get('source_end')}."
    if stage == "source_finished":
        source_label = str(event.get("source_label") or "Climate source")
        return f"Finished {source_label} for this time slice."
    return "Loading climate data..."


def _render_temperature_fetch_progress(
    placeholder,
    *,
    total_locations: int,
    completed_locations: int,
    total_shared_tasks: int,
    completed_shared_tasks: int,
    active_locations: int,
    stage_label: str,
    current_detail: str,
    recent_updates: list[str],
    started_at: float,
    active: bool,
) -> None:
    normalized_total_locations = max(1, total_locations)
    location_fraction = min(1.0, completed_locations / normalized_total_locations)
    shared_fraction = 1.0 if total_shared_tasks == 0 else min(1.0, completed_shared_tasks / total_shared_tasks)
    overall_fraction = location_fraction if total_shared_tasks == 0 else min(
        1.0,
        (0.65 * shared_fraction) + (0.35 * location_fraction),
    )
    display_fraction = overall_fraction
    if active:
        display_fraction = max(display_fraction, min(0.04, 1.0 / max(1, normalized_total_locations + total_shared_tasks)))

    elapsed_seconds = perf_counter() - started_at
    rate_parts: list[str] = []
    if elapsed_seconds > 0 and completed_locations > 0:
        rate_parts.append(_format_progress_rate(completed_locations * 60.0 / elapsed_seconds, "locations"))
    if elapsed_seconds > 0 and completed_shared_tasks > 0:
        rate_parts.append(_format_progress_rate(completed_shared_tasks * 60.0 / elapsed_seconds, "shared tasks"))

    title = "Preparing climate-data timelines"
    if stage_label:
        title = stage_label
    if not active and completed_locations >= total_locations:
        title = f"Loaded {completed_locations}/{normalized_total_locations} locations"

    status_line = current_detail or "Loading climate data..."
    summary_left = f"{completed_locations}/{normalized_total_locations} places ready"
    summary_right = " | ".join(rate_parts) if rate_parts else "Waiting for the first completed step..."
    elapsed_text = f"Elapsed {_format_progress_duration(elapsed_seconds)}"
    width_percent = max(0.0, min(100.0, display_fraction * 100.0))
    active_class = " mystripes-progress-fill-active" if active else ""
    shared_summary = (
        f"{completed_shared_tasks}/{total_shared_tasks} shared source tasks ready"
        if total_shared_tasks
        else "No shared source downloads needed"
    )
    active_summary = (
        f"{active_locations} location timeline{'s' if active_locations != 1 else ''} active"
        if active
        else "No active location tasks"
    )
    updates_html = ""
    if recent_updates:
        items = "".join(f"<li>{escape(update)}</li>" for update in recent_updates[-4:])
        updates_html = f'<ul class="mystripes-progress-updates">{items}</ul>'

    html = f'''
<style>
@keyframes mystripes-progress-shimmer {{
  from {{ background-position: 0 0; }}
  to {{ background-position: 1.6rem 0; }}
}}
.mystripes-progress-card {{
  border: 1px solid rgba(148, 163, 184, 0.45);
  border-radius: 0.9rem;
  padding: 0.9rem 1rem;
  margin: 0.35rem 0 1rem 0;
  background: linear-gradient(180deg, rgba(248, 250, 252, 0.96), rgba(241, 245, 249, 0.96));
}}
.mystripes-progress-kicker {{
  font-size: 0.78rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #475569;
}}
.mystripes-progress-title {{
  font-size: 1rem;
  font-weight: 600;
  color: #0f172a;
  margin-top: 0.15rem;
}}
.mystripes-progress-detail {{
  color: #334155;
  margin-top: 0.2rem;
}}
.mystripes-progress-track {{
  position: relative;
  overflow: hidden;
  height: 0.8rem;
  margin: 0.75rem 0 0.55rem 0;
  border-radius: 999px;
  background: rgba(148, 163, 184, 0.22);
}}
.mystripes-progress-fill {{
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #0f766e, #38bdf8);
}}
.mystripes-progress-fill-active {{
  background-image:
    linear-gradient(135deg,
      rgba(15, 118, 110, 0.95) 0%,
      rgba(15, 118, 110, 0.95) 25%,
      rgba(56, 189, 248, 0.95) 25%,
      rgba(56, 189, 248, 0.95) 50%,
      rgba(15, 118, 110, 0.95) 50%,
      rgba(15, 118, 110, 0.95) 75%,
      rgba(56, 189, 248, 0.95) 75%,
      rgba(56, 189, 248, 0.95) 100%);
  background-size: 1.6rem 1.6rem;
  animation: mystripes-progress-shimmer 0.9s linear infinite;
}}
.mystripes-progress-meta {{
  display: flex;
  justify-content: space-between;
  gap: 0.75rem;
  flex-wrap: wrap;
  font-size: 0.84rem;
  color: #475569;
}}
.mystripes-progress-updates {{
  margin: 0.7rem 0 0 1rem;
  color: #475569;
  font-size: 0.84rem;
}}
.mystripes-progress-updates li + li {{
  margin-top: 0.2rem;
}}
</style>
<div class="mystripes-progress-card">
  <div class="mystripes-progress-kicker">Climate-data load progress</div>
  <div class="mystripes-progress-title">{escape(title)}</div>
  <div class="mystripes-progress-detail">{escape(status_line)}</div>
  <div class="mystripes-progress-track">
    <div class="mystripes-progress-fill{active_class}" style="width: {width_percent:.1f}%;"></div>
  </div>
  <div class="mystripes-progress-meta">
    <span>{escape(summary_left)}</span>
    <span>{escape(shared_summary)}</span>
    <span>{escape(active_summary)}</span>
    <span>{escape(summary_right)}</span>
    <span>{escape(elapsed_text)}</span>
  </div>
  {updates_html}
</div>
'''
    placeholder.markdown(html, unsafe_allow_html=True)


def _cookie_consent_copy() -> str:
    return (
        "On Streamlit Community Cloud, MyStripes can optionally store your story line name "
        "and place-period timeline entries in browser cookies so you can load them later. "
        "The app does not collect or retain user data for itself. Any cookie-stored data "
        "stays on your machine and only aids convenience. MyStripes is open-source on "
        f"[GitHub]({PROJECT_REPOSITORY_URL})."
    )


def _cookie_storage_note() -> str:
    return (
        "Saved cloud story lines use optional browser cookies that stay on this machine. "
        "The app does not collect or retain user data for itself, and the cookies only aid "
        "convenience."
    )


def _render_cookie_consent_banner(storage_backend: str) -> None:
    if storage_backend != "cookie" or _cloud_cookie_consent_choice() is not None:
        return

    with st.container(border=True):
        st.markdown("**Cookie choice for saved story lines**")
        st.markdown(_cookie_consent_copy())
        st.caption(
            "These optional convenience cookies are off until you choose below. Rejecting "
            "keeps save-load-remove hidden on Streamlit Community Cloud. You can change this "
            "later in Cookie settings in the sidebar."
        )
        if st.session_state.get("pending_cookie_store_operation"):
            st.caption("Updating the cookie choice in this browser...")
        accept_column, reject_column = st.columns(2)
        if accept_column.button("Allow convenience cookies", key="cookie_banner_accept"):
            _queue_cookie_consent_operation("accepted")
            st.rerun()
        if reject_column.button("Reject optional cookies", key="cookie_banner_reject"):
            _queue_cookie_consent_operation("rejected")
            st.rerun()


def _render_cookie_settings(sidebar, storage_backend: str) -> None:
    if storage_backend != "cookie":
        return

    consent_choice = _cloud_cookie_consent_choice()
    with sidebar.expander("Cookie settings", expanded=consent_choice is None):
        st.markdown(_cookie_consent_copy())
        st.caption(
            "You can change this choice at any time here. Rejecting optional convenience "
            "cookies removes saved cloud story lines from this browser."
        )
        if consent_choice == "accepted":
            st.success("Optional convenience cookies are enabled in this browser.")
        elif consent_choice == "rejected":
            st.info("Optional convenience cookies are disabled in this browser.")
        else:
            st.warning("Optional convenience cookies are not enabled yet.")
        if st.session_state.get("pending_cookie_store_operation"):
            st.caption("Updating the cookie choice in this browser...")
        accept_column, reject_column = st.columns(2)
        if accept_column.button("Allow convenience cookies", key="cookie_settings_accept"):
            _queue_cookie_consent_operation("accepted")
            st.rerun()
        if reject_column.button("Reject optional cookies", key="cookie_settings_reject"):
            _queue_cookie_consent_operation("rejected")
            st.rerun()


def _render_storyline_panel(sidebar, analysis_end: date, storage_backend: str) -> None:
    _render_storyline_feedback(sidebar)
    _apply_pending_storyline_widget_updates()

    try:
        saved_storylines = _load_saved_storylines(storage_backend)
    except Exception as exc:
        sidebar.error(f"Could not load saved story lines: {exc}")
        saved_storylines = {}

    cloud_storage_enabled = storage_backend != "cookie" or _cloud_storyline_storage_enabled()
    expander_expanded = bool(saved_storylines) or not cloud_storage_enabled

    with sidebar.expander("Story lines", expanded=expander_expanded):
        st.caption(
            "Save, reload, and remove place-based story lines so you can revisit the same "
            "timeline later."
        )

        if storage_backend == "cookie" and not cloud_storage_enabled:
            if _cloud_cookie_consent_choice() == "rejected":
                st.info(
                    "Optional convenience cookies are currently disabled in this browser. Use "
                    "Cookie settings above if you want to save or reload story lines here."
                )
            else:
                st.info(
                    "On Streamlit Community Cloud, save-load-remove uses optional convenience "
                    "cookies. Choose in Cookie settings above if you want this feature in this "
                    "browser."
                )
            st.caption(_cookie_storage_note())
            return

        st.text_input(
            "Story line name",
            key="storyline_name",
            placeholder="Life in Vienna, Berlin, Amsterdam...",
            help="Used to identify the saved story line when loading it later.",
        )

        selected_storyline_name = ""
        if saved_storylines:
            saved_names = list(saved_storylines)
            if st.session_state.saved_storyline_name not in saved_names:
                st.session_state.saved_storyline_name = saved_names[0]
            selected_storyline_name = st.selectbox(
                "Saved story lines",
                options=saved_names,
                key="saved_storyline_name",
            )
        else:
            st.caption("No saved story lines yet.")

        action_columns = st.columns(3)
        save_requested = action_columns[0].button("Save", key="save_storyline")
        load_requested = action_columns[1].button(
            "Load",
            key="load_storyline",
            disabled=not selected_storyline_name,
        )
        remove_requested = action_columns[2].button(
            "Remove",
            key="remove_storyline",
            disabled=not selected_storyline_name,
        )

        if storage_backend == "file":
            st.caption(f"Local story lines are written to `{LOCAL_STORYLINES_PATH}` on this machine.")
        else:
            st.caption(_cookie_storage_note())
            if st.session_state.get("pending_cookie_store_operation"):
                st.caption("Syncing saved story lines in this browser...")

        if save_requested:
            try:
                payload = serialize_storyline_payload(
                    name=st.session_state.storyline_name,
                    birth_date=st.session_state.birth_date,
                    period_entries=_current_storyline_period_entries(),
                    include_boundary_geojson=storage_backend == "file",
                )
                if storage_backend == "file":
                    _queue_storyline_widget_update("storyline_name", payload["name"])
                    _queue_storyline_widget_update("saved_storyline_name", payload["name"])
                    save_local_storyline(payload)
                    _set_storyline_feedback("success", f"Saved story line `{payload['name']}`.")
                    st.rerun()

                _queue_cookie_storyline_operation(
                    action="save",
                    storyline_name=str(payload["name"]),
                    cookie_name=storyline_cookie_name(payload["name"]),
                    cookie_value=encode_storyline_cookie_value(payload),
                )
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        if load_requested and selected_storyline_name:
            payload = saved_storylines[selected_storyline_name]
            _apply_storyline_to_session(payload, analysis_end)
            _queue_storyline_widget_update("storyline_name", selected_storyline_name)
            _queue_storyline_widget_update("saved_storyline_name", selected_storyline_name)
            _set_storyline_feedback("success", f"Loaded story line `{selected_storyline_name}`.")
            st.rerun()

        if remove_requested and selected_storyline_name:
            try:
                if storage_backend == "file":
                    removed = remove_local_storyline(selected_storyline_name)
                    if removed:
                        if st.session_state.storyline_name == selected_storyline_name:
                            _queue_storyline_widget_update("storyline_name", "")
                        _queue_storyline_widget_update("saved_storyline_name", "")
                        _set_storyline_feedback("success", f"Removed story line `{selected_storyline_name}`.")
                        st.rerun()
                    st.error(f"No saved story line named `{selected_storyline_name}` was found.")
                else:
                    _queue_cookie_storyline_operation(
                        action="remove",
                        storyline_name=selected_storyline_name,
                        cookie_name=storyline_cookie_name(selected_storyline_name),
                    )
                    st.rerun()
            except Exception as exc:
                st.error(str(exc))


def main() -> None:
    dataset_window = cached_dataset_window()
    today = date.today()
    analysis_end = min(today, dataset_window.max_end)

    st.title("MyStripes")
    st.write(
        "Build climate strips from places and periods using ERA5-Land monthly temperature "
        "data with automatic historical fallback for earlier years, and export them as "
        "minimal graphic in PNG, SVG, or PDF. Use it to communicate how you or some entity "
        "experienced climate change."
    )
    st.markdown(
        f"**[{SHOW_YOUR_STRIPES_CREDIT}]({SHOW_YOUR_STRIPES_URL})** "
        "The original warming stripes are a powerful tool to communicate climate change. "
        "Also, visit [#BiodiversityStripes](https://biodiversitystripes.info/) "
        "to communicate the intensifying biodiversity crisis."
    )
    st.info(
        f"MyStripes is open-source on [GitHub]({PROJECT_REPOSITORY_URL}). Direct contributions "
        f"and ideas are welcome: [contributing guide]({CONTRIBUTING_GUIDE_URL}) or "
        f"[open a GitHub issue]({PROJECT_ISSUES_URL})."
    )

    _initialize_state(analysis_end)
    storage_backend = _storyline_storage_backend()
    if storage_backend == "cookie":
        _sync_cookie_store()

    _render_cookie_consent_banner(storage_backend)

    sidebar = st.sidebar
    _render_cookie_settings(sidebar, storage_backend)
    _render_storyline_panel(sidebar, analysis_end, storage_backend)
    active_cds_config = _render_cds_access_panel(sidebar)
    sidebar.header("Output")
    birth_date = sidebar.date_input(
        "Timeline start",
        min_value=dataset_window.min_start,
        max_value=analysis_end,
        key="birth_date",
        help="For personal timelines, set this to your birth date.",
    )
    spatial_mode = sidebar.selectbox(
        "Spatial aggregation",
        options=("single_cell", "radius", "boundary"),
        format_func=lambda value: {
            "single_cell": "Nearest single grid cell",
            "radius": "Average grid cells in a radius",
            "boundary": "Average grid cells inside place boundary",
        }[value],
        help=(
            "MyStripes uses ERA5-Land where available and automatically falls back to "
            "historical datasets for earlier years. Radius mode averages grid cells inside "
            "the chosen radius. Boundary mode uses the municipality, district, region, or "
            "other place polygon returned by the geocoder when available, with a bounding-box "
            "fallback when only an area extent is available."
        ),
    )
    radius_km = None
    if spatial_mode == "radius":
        radius_km = float(
            sidebar.number_input("Radius (km)", min_value=5.0, max_value=250.0, value=25.0, step=5.0)
        )
    aggregation_mode = sidebar.selectbox(
        "Stripe period",
        options=("full_calendar_years", "rolling_365_day"),
        format_func=lambda value: {
            "full_calendar_years": "Full calendar years only",
            "rolling_365_day": "365-day moving average",
        }[value],
        help=(
            "Full calendar years only drops partial birth and current years. The 365-day "
            "moving average mode expands the monthly series to daily values, applies a "
            "365-day rolling mean, then samples the smoothed series."
        ),
    )
    reference_period_mode = sidebar.selectbox(
        "Climatology reference period",
        options=("climate_normal_1961_2010", "story_line_period"),
        format_func=lambda value: {
            "climate_normal_1961_2010": "1961-2010",
            "story_line_period": "Story line period",
        }[value],
        help=(
            "Choose which date window defines the per-location day-of-year climatology. "
            "The app reuses a shared climate-data timeline for each unique place and derives "
            "the selected reference period from that cached data."
        ),
    )
    rolling_sample_mode = "monthly"
    rolling_strip_count = None
    if aggregation_mode == "rolling_365_day":
        rolling_sample_mode = sidebar.selectbox(
            "Rolling sample spacing",
            options=("monthly", "fixed_count"),
            format_func=lambda value: {
                "monthly": "Monthly",
                "fixed_count": "Fixed number of strips",
            }[value],
            help=(
                "Monthly keeps one rolling-average stripe per month using the latest "
                "available day in that month. Fixed count distributes the chosen number of "
                "stripes evenly across the available rolling-average dates."
            ),
        )
        if rolling_sample_mode == "fixed_count":
            rolling_strip_count = int(
                sidebar.number_input(
                    "Rolling strips",
                    min_value=2,
                    max_value=400,
                    value=60,
                    step=1,
                )
            )
        sidebar.warning(
            "Rolling mode can be misread because warming stripes are conventionally "
            "interpreted as one stripe per year. Here, each stripe represents a sampled "
            "365-day moving average instead."
        )
    width_px = int(sidebar.number_input("Width (px)", min_value=100, max_value=6000, value=1800, step=100))
    height_px = int(sidebar.number_input("Height (px)", min_value=20, max_value=2400, value=260, step=20))
    png_dpi = int(sidebar.number_input("PNG DPI", min_value=10, max_value=600, value=200, step=10))
    watermark_text = sidebar.text_input(
        "Watermark text",
        value="",
        help="Optional text rendered over the stripes graphic.",
    )
    watermark_horizontal_align = "center"
    watermark_vertical_align = "center"
    watermark_color = "#FFFFFF"
    watermark_opacity = 0.35
    watermark_shadow = False
    watermark_max_width_ratio = 0.8
    watermark_max_height_ratio = 0.8
    if watermark_text.strip():
        watermark_horizontal_align = sidebar.selectbox(
            "Watermark horizontal align",
            options=("left", "center", "right"),
            format_func=lambda value: value.title(),
        )
        watermark_vertical_align = sidebar.selectbox(
            "Watermark vertical align",
            options=("top", "center", "bottom"),
            format_func=lambda value: value.title(),
        )
        watermark_color = sidebar.color_picker("Watermark color", value=watermark_color)
        watermark_opacity = float(
            sidebar.slider(
                "Watermark opacity",
                min_value=0.0,
                max_value=1.0,
                value=watermark_opacity,
                step=0.05,
                format="%.2f",
            )
        )
        watermark_shadow = sidebar.checkbox(
            "Watermark shadow",
            value=watermark_shadow,
            help="Adds a dark outline behind the watermark text for stronger contrast.",
        )
        watermark_max_width_ratio = float(
            sidebar.slider(
                "Watermark max width",
                min_value=0.1,
                max_value=1.0,
                value=watermark_max_width_ratio,
                step=0.05,
                format="%.2f",
            )
        )
        watermark_max_height_ratio = float(
            sidebar.slider(
                "Watermark max height",
                min_value=0.1,
                max_value=1.0,
                value=watermark_max_height_ratio,
                step=0.05,
                format="%.2f",
            )
        )
    file_stem = sidebar.text_input("Download name", value="mystripes")
    debug_mode = sidebar.checkbox(
        "Debug mode",
        key="debug_mode",
        help="Print diagnostic information to stdout / server logs.",
    )

    _render_credit_and_license_panel(today.year)
    _render_time_series_summary_note()

    if birth_date < dataset_window.min_start:
        st.caption(
            f"The currently supported climate-data stack begins on {dataset_window.min_start.isoformat()}, "
            "so the stripes start there even if your timeline start or birth date is earlier."
        )

    for index, entry in enumerate(st.session_state.period_entries):
        start_label, end_label = _period_range_label(
            index=index,
            entries=st.session_state.period_entries,
            birth_date=max(birth_date, dataset_window.min_start),
            analysis_end=analysis_end,
        )
        expander_label = f"Period {index + 1}: {start_label} to {end_label}"
        with st.expander(expander_label, expanded=True):
            entry["place_query"] = st.text_input(
                "City or region",
                value=entry["place_query"],
                key=f"place_query_{index}",
                on_change=_queue_geocoding_search,
                args=(index,),
                placeholder="Vienna, Austria or Tyrol, Austria",
                help=(
                    "Press Enter or use Find place. Search by city, state, province, region, "
                    "or country. Area results use a centroid to auto-fill coordinates and can "
                    "also supply an area boundary for boundary aggregation."
                ),
            )

            action_columns = st.columns((1, 2))
            search_requested = action_columns[0].button("Find place", key=f"find_place_{index}")
            search_requested = bool(st.session_state.pop(f"_pending_geocode_search_{index}", False)) or search_requested
            if search_requested:
                try:
                    _run_geocoding_search(index)
                except ValueError as exc:
                    st.warning(str(exc))
                except Exception as exc:
                    st.error(f"Geocoding failed: {exc}")

            geocode_results = st.session_state.get(f"geocode_results_{index}", [])
            if geocode_results:
                labels = [result["display_name"] for result in geocode_results]
                default_index = _current_choice_index(geocode_results, entry["resolved_name"])
                selected_label = action_columns[1].selectbox(
                    "Matches",
                    options=labels,
                    index=default_index,
                    key=f"geocode_choice_{index}",
                )
                selected_result = next(
                    result for result in geocode_results if result["display_name"] == selected_label
                )
                _apply_geocoding_choice(index, selected_result)

            resolved_name = entry["resolved_name"] or "No geocoded place selected yet"
            coordinate_source = str(entry.get("coordinate_source", "")).strip()
            if coordinate_source:
                st.caption(f"Using: {resolved_name} ({coordinate_source})")
            else:
                st.caption(f"Using: {resolved_name}")
            if spatial_mode == "boundary":
                st.caption(_boundary_mode_caption(entry))

            coordinate_columns = st.columns(2)
            entry["latitude_text"] = coordinate_columns[0].text_input(
                "Latitude",
                value=entry["latitude_text"],
                key=f"latitude_{index}",
                placeholder="48.2082",
            )
            entry["longitude_text"] = coordinate_columns[1].text_input(
                "Longitude",
                value=entry["longitude_text"],
                key=f"longitude_{index}",
                placeholder="16.3738",
            )

            if index < len(st.session_state.period_entries) - 1:
                entry["end_date"] = st.date_input(
                    "Period ends",
                    value=entry["end_date"] or (analysis_end - timedelta(days=365)),
                    min_value=max(birth_date, dataset_window.min_start),
                    max_value=analysis_end - timedelta(days=1),
                    key=f"end_date_{index}",
                )
            else:
                st.caption(f"Final period currently ends on `{analysis_end.isoformat()}`.")

            entry["show_indicator"] = st.checkbox(
                "Indicate this period in graphic",
                value=bool(entry.get("show_indicator", False)),
                key=f"show_indicator_{index}",
                help="Adds an approximate range guide for this period on top of the stripes preview.",
            )
            if entry["show_indicator"]:
                entry["indicator_label"] = st.text_input(
                    "Graphic label",
                    value=str(entry.get("indicator_label", "") or ""),
                    key=f"indicator_label_{index}",
                    placeholder=_default_indicator_label(entry, index),
                    help="Optional shorter label for the graphic. Leave blank to use the place name.",
                )

    controls = st.columns((1, 1, 2))
    if controls[0].button("Add period"):
        _add_period_entry(analysis_end)
        st.rerun()
    if controls[1].button("Remove last period", disabled=len(st.session_state.period_entries) == 1):
        _remove_period_entry()
        st.rerun()
    controls[2].caption(
        "Enter one place per period. This works for personal life stations, study or work "
        "phases, projects, tours, or any other place-based sequence. The app derives each "
        "next period start date from the previous period end date."
    )

    indicated_period_indices = [
        index
        for index, entry in enumerate(st.session_state.period_entries)
        if bool(entry.get("show_indicator", False))
    ]
    period_indicator_style = "scale_bar"
    period_indicator_vertical_align = "bottom"
    period_indicator_color = "#ffffff"
    period_indicator_height_ratio = 0.2
    if indicated_period_indices:
        with st.expander("Period indicators", expanded=True):
            st.caption(
                "Show approximate guides for the selected periods on top of the stripes. "
                "Scale bars keep a tiny gap between neighboring indicators so they do not touch."
            )
            indicator_columns = st.columns(4)
            period_indicator_style = indicator_columns[0].selectbox(
                "Design",
                options=("scale_bar", "outward_arrows"),
                format_func=lambda option: {
                    "scale_bar": "Scale bar",
                    "outward_arrows": "Outward arrows",
                }[option],
                key="period_indicator_style",
            )
            period_indicator_vertical_align = indicator_columns[1].selectbox(
                "Placement",
                options=("bottom", "center", "top"),
                format_func=lambda option: option.capitalize(),
                key="period_indicator_vertical_align",
            )
            period_indicator_color = indicator_columns[2].color_picker(
                "Indicator color",
                value="#ffffff",
                key="period_indicator_color",
            )
            period_indicator_height_ratio = (
                indicator_columns[3].slider(
                    "Indicator height",
                    min_value=5,
                    max_value=100,
                    value=20,
                    step=5,
                    format="%d%%",
                    key="period_indicator_height_percent",
                    help="Approximate share of the stripe graphic height that the indicator overlay may use.",
                )
                / 100.0
            )

    st.session_state.period_entries = _current_storyline_period_entries()
    periods_preview, preview_errors = build_periods_from_entries(
        entries=st.session_state.period_entries,
        birth_date=birth_date,
        analysis_end=analysis_end,
        analysis_min_start=dataset_window.min_start,
    )
    debug_period_aliases = _build_debug_period_aliases(periods_preview)
    debug_period_identifications = _build_debug_period_identifications(periods_preview)

    for error in preview_errors:
        st.error(error)
    if debug_period_identifications:
        _debug_print(
            debug_mode,
            "period_aliases",
            debug_period_identifications,
            period_aliases=debug_period_aliases,
        )
    _debug_print(
        debug_mode,
        "preview_summary",
        {"period_count": len(periods_preview), "errors": preview_errors},
        period_aliases=debug_period_aliases,
    )

    report_start = max(birth_date, dataset_window.min_start)
    first_period_history_days = 364 if aggregation_mode == "rolling_365_day" else 0
    effective_report_end = analysis_end
    reference_start = report_start
    reference_end = analysis_end
    baseline_label = ""
    baseline_metric_value = ""
    shared_climatology_label = ""
    fetch_window_start = report_start
    fetch_window_end = analysis_end
    location_requests = []
    climate_batch_plan = None
    historical_note = None
    historical_download_estimate = None
    reference_resolution_error = None
    try:
        effective_report_end = latest_data_end_that_can_change_stripes(
            report_start=report_start,
            analysis_end=analysis_end,
            aggregation_mode=aggregation_mode,
            reference_period_mode=reference_period_mode,
        )
        (
            reference_start,
            reference_end,
            baseline_label,
            baseline_metric_value,
            shared_climatology_label,
        ) = _resolve_climatology_reference_period(
            reference_mode=reference_period_mode,
            story_line_start=report_start,
            story_line_end=effective_report_end,
            dataset_start=dataset_window.min_start,
            dataset_end=dataset_window.max_end,
        )
        fetch_window_start = min(
            max(dataset_window.min_start, report_start - timedelta(days=first_period_history_days)),
            reference_start,
        )
        fetch_window_end = max(effective_report_end, reference_end)
        location_requests = build_location_climate_requests(
            periods_preview,
            reference_start=reference_start,
            reference_end=reference_end,
            first_period_history_days=first_period_history_days,
        )
        historical_note = climate_stack_note(fetch_window_start, fetch_window_end)
        historical_download_estimate = estimate_climate_downloads(
            location_requests,
            spatial_mode=spatial_mode,
            radius_km=radius_km,
        )
        climate_batch_plan = build_climate_batch_plan(
            location_requests,
            spatial_mode=spatial_mode,
            radius_km=radius_km,
        )
    except ValueError as exc:
        reference_resolution_error = str(exc)
        _debug_print(debug_mode, "reference_period_error", str(exc), period_aliases=debug_period_aliases)
        st.error(str(exc))

    if historical_note and not preview_errors:
        st.caption(historical_note)
    if historical_download_estimate is not None and not preview_errors:
        preflight_message = preflight_download_message(historical_download_estimate)
        if preflight_message is not None:
            level, message = preflight_message
            getattr(st, level)(message)

    generate = st.button(
        "Generate stripes",
        type="primary",
        disabled=bool(preview_errors or reference_resolution_error),
    )
    if not generate:
        return

    if active_cds_config is None:
        st.error(
            "Missing CDS credentials. Add them to Streamlit secrets, environment variables, "
            "enter a session-only override, or save a local token from the app sidebar."
        )
        _debug_print(debug_mode, "missing_cds_credentials", period_aliases=debug_period_aliases)
        return

    _debug_print(
        debug_mode,
        "generate_request",
        {
            "report_start": report_start.isoformat(),
            "analysis_end": analysis_end.isoformat(),
            "effective_report_end": effective_report_end.isoformat(),
            "spatial_mode": spatial_mode,
            "radius_km": radius_km,
            "climatology": {
                "mode": reference_period_mode,
                "start": reference_start.isoformat(),
                "end": reference_end.isoformat(),
            },
            "fetch_window": {
                "start": fetch_window_start.isoformat(),
                "end": fetch_window_end.isoformat(),
            },
            "aggregation_mode": aggregation_mode,
            "rolling_sample_mode": rolling_sample_mode,
            "rolling_strip_count": rolling_strip_count,
            "cds_source": active_cds_config.source,
            "historical_download_estimate": (
                None
                if historical_download_estimate is None
                else {
                    "uses_historical_fallback": historical_download_estimate.uses_historical_fallback,
                    "uncached_era5_bridge_fetches": historical_download_estimate.uncached_era5_bridge_fetches,
                    "uncached_twcr_fetches": historical_download_estimate.uncached_twcr_fetches,
                    "uncached_twcr_years": list(historical_download_estimate.uncached_twcr_years),
                }
            ),
            "shared_climate_tasks": 0 if climate_batch_plan is None else len(climate_batch_plan.shared_tasks),
            "watermark": {
                "text": watermark_text,
                "horizontal_align": watermark_horizontal_align,
                "vertical_align": watermark_vertical_align,
                "color": watermark_color,
                "opacity": watermark_opacity,
                "shadow": watermark_shadow,
                "max_width_ratio": watermark_max_width_ratio,
                "max_height_ratio": watermark_max_height_ratio,
            },
            "periods": [
                {
                    "label": period.label,
                    "place": period.display_name,
                    "start": period.start_date.isoformat(),
                    "end": period.end_date.isoformat(),
                    "latitude": round(period.latitude, 4),
                    "longitude": round(period.longitude, 4),
                }
                for period in periods_preview
            ],
        },
        period_aliases=debug_period_aliases,
    )

    rolling_crop_start = min(period.start_date for period in periods_preview)
    unique_periods_by_location = {request.location_key: request for request in location_requests}

    download_progress_placeholder = st.empty()
    download_started_at = perf_counter()
    total_locations = len(unique_periods_by_location)
    total_shared_tasks = 0 if climate_batch_plan is None else len(climate_batch_plan.shared_tasks)
    completed_locations = 0
    completed_shared_task_ids: set[str] = set()
    active_location_keys: set[str] = set()
    stage_label = "Planning shared climate-data fetches"
    current_detail = "Checking saved timelines and determining whether climate-data updates are needed."
    recent_updates: list[str] = []

    def refresh_download_progress(*, active: bool) -> None:
        _render_temperature_fetch_progress(
            download_progress_placeholder,
            total_locations=total_locations,
            completed_locations=completed_locations,
            total_shared_tasks=total_shared_tasks,
            completed_shared_tasks=len(completed_shared_task_ids),
            active_locations=len(active_location_keys),
            stage_label=stage_label,
            current_detail=current_detail,
            recent_updates=recent_updates,
            started_at=download_started_at,
            active=active,
        )

    refresh_download_progress(active=True)

    try:
        def _on_fetch_progress(event: dict[str, object]) -> None:
            nonlocal completed_locations
            nonlocal current_detail
            nonlocal stage_label

            stage = str(event.get("stage") or "")
            work_id = str(event.get("work_id") or "").strip()
            location_key = str(event.get("location_key") or "").strip()

            if stage == "batch_plan":
                stage_label = "Planning shared climate-data fetches"
            elif stage == "shared_task_started":
                stage_label = "Downloading shared source files"
            elif stage == "shared_task_finished" and work_id:
                completed_shared_task_ids.add(work_id)
                if len(completed_shared_task_ids) >= total_shared_tasks and total_locations:
                    stage_label = "Assembling location timelines"
            elif stage == "location_started":
                if location_key:
                    active_location_keys.add(location_key)
                stage_label = "Assembling location timelines"
            elif stage == "location_finished":
                if location_key:
                    active_location_keys.discard(location_key)
                completed_locations += 1
                if completed_locations >= total_locations:
                    stage_label = "Finalizing climate-data timelines"

            detail = _describe_temperature_fetch_event(event)
            if detail:
                current_detail = detail
                if not recent_updates or recent_updates[-1] != detail:
                    recent_updates.append(detail)
                    del recent_updates[:-4]

            still_active = (
                len(completed_shared_task_ids) < total_shared_tasks
                or completed_locations < total_locations
                or bool(active_location_keys)
            )
            refresh_download_progress(active=still_active)

        location_frames = fetch_saved_climate_series_batch(
            config=active_cds_config,
            location_requests=list(unique_periods_by_location.values()),
            spatial_mode=spatial_mode,
            radius_km=radius_km,
            progress_callback=_on_fetch_progress,
        )

        period_frames = [location_frames[period.location_key] for period in periods_preview]
    except (CDSRequestError, ValueError) as exc:
        _debug_print(debug_mode, "period_fetch_error", str(exc), period_aliases=debug_period_aliases)
        current_detail = f"Stopped while loading climate data: {exc}"
        stage_label = "Climate-data loading stopped"
        refresh_download_progress(active=False)
        st.error(str(exc))
        return

    stage_label = "Climate-data timelines ready"
    current_detail = "All requested climate-data timelines are ready."
    refresh_download_progress(active=False)
    _debug_print(debug_mode, "fetched_period_frames", period_frames, period_aliases=debug_period_aliases)

    _debug_print(
        debug_mode,
        "period_baseline_mode",
        {
            "label": baseline_label,
            "start": reference_start.isoformat(),
            "end": reference_end.isoformat(),
        },
        period_aliases=debug_period_aliases,
    )

    try:
        merged_daily = build_merged_daily_series(
            periods=periods_preview,
            frames_by_period=period_frames,
            report_start=report_start - timedelta(days=first_period_history_days),
            report_end=effective_report_end,
            baseline_start=reference_start,
            baseline_end=reference_end,
            first_period_history_days=first_period_history_days,
        )
        stripe_frame = aggregate_daily_series_to_stripes(
            daily_series=merged_daily,
            aggregation_mode=aggregation_mode,
            rolling_window_end=effective_report_end,
            rolling_crop_start=rolling_crop_start,
            rolling_sample_mode=rolling_sample_mode,
            rolling_strip_count=rolling_strip_count,
        )
    except ValueError as exc:
        _debug_print(debug_mode, "stripe_aggregation_error", str(exc), period_aliases=debug_period_aliases)
        st.error(str(exc))
        return
    _debug_print(debug_mode, "merged_daily", merged_daily, period_aliases=debug_period_aliases)
    _debug_print(debug_mode, "stripe_frame", stripe_frame, period_aliases=debug_period_aliases)

    try:
        all_periods_report, merged_report = build_period_report_tables(
            periods=periods_preview,
            frames_by_period=period_frames,
            report_start=report_start,
            report_end=effective_report_end,
            baseline_start=reference_start,
            baseline_end=reference_end,
        )
    except ValueError as exc:
        _debug_print(debug_mode, "report_build_error", str(exc), period_aliases=debug_period_aliases)
        st.error(str(exc))
        return
    _debug_print(debug_mode, "all_periods_report", all_periods_report, period_aliases=debug_period_aliases)
    _debug_print(debug_mode, "merged_report", merged_report, period_aliases=debug_period_aliases)

    period_indicator_specs = None
    if indicated_period_indices:
        period_indicator_specs = build_period_indicator_specs(
            periods_preview,
            stripe_frame,
            included_period_indices=indicated_period_indices,
        )
        for spec in period_indicator_specs:
            period_index = int(spec["period_index"])
            fallback_label = periods_preview[period_index].label
            spec["label"] = (
                str(st.session_state.period_entries[period_index].get("indicator_label", "") or "").strip()
                or fallback_label
            )

    width_inches = width_px / png_dpi
    height_inches = height_px / png_dpi
    figure = render_stripes_figure(
        anomalies=stripe_frame["anomaly_c"].tolist(),
        width_inches=width_inches,
        height_inches=height_inches,
        watermark_text=watermark_text or None,
        watermark_horizontal_align=watermark_horizontal_align,
        watermark_vertical_align=watermark_vertical_align,
        watermark_color=watermark_color,
        watermark_opacity=watermark_opacity,
        watermark_shadow=watermark_shadow,
        watermark_max_width_ratio=watermark_max_width_ratio,
        watermark_max_height_ratio=watermark_max_height_ratio,
        period_indicators=period_indicator_specs,
        period_indicator_style=period_indicator_style,
        period_indicator_vertical_align=period_indicator_vertical_align,
        period_indicator_color=period_indicator_color,
        period_indicator_height_ratio=period_indicator_height_ratio,
    )

    png_bytes = export_figure_bytes(figure, "png", png_dpi)
    svg_bytes = export_figure_bytes(figure, "svg", png_dpi)
    pdf_bytes = export_figure_bytes(figure, "pdf", png_dpi)

    st.subheader("Preview")
    st.image(png_bytes, width="stretch")
    _render_reference_period_global_warming_warning(
        reference_start=reference_start,
        reference_end=reference_end,
    )
    if effective_report_end < analysis_end:
        st.caption(
            "Temperature data is currently reused through "
            f"{effective_report_end.isoformat()} because newer monthly values cannot yet change "
            "full-calendar-year stripes with the selected fixed climatology reference period."
        )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Stripes shown", int(len(stripe_frame)))
    metric_columns[1].metric("Climatology", baseline_metric_value)
    metric_columns[2].metric("Warmest anomaly", f"{stripe_frame['anomaly_c'].max():+.2f} C")
    metric_columns[3].metric("Coolest anomaly", f"{stripe_frame['anomaly_c'].min():+.2f} C")
    st.caption(
        _aggregation_mode_caption(
            baseline_label,
            aggregation_mode,
            effective_report_end,
            rolling_sample_mode,
            rolling_strip_count,
        )
    )
    if spatial_mode == "radius" and radius_km is not None:
        st.caption(f"Spatial aggregation: mean across grid cells within {radius_km:.0f} km.")
    elif spatial_mode == "boundary":
        st.caption("Spatial aggregation: mean across grid cells inside the selected place boundary.")
    else:
        st.caption("Spatial aggregation: nearest single ERA5-Land grid cell.")

    download_columns = st.columns(3)
    download_columns[0].download_button(
        "Download PNG",
        data=png_bytes,
        file_name=f"{file_stem}.png",
        mime="image/png",
    )
    download_columns[1].download_button(
        "Download SVG",
        data=svg_bytes,
        file_name=f"{file_stem}.svg",
        mime="image/svg+xml",
    )
    download_columns[2].download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name=f"{file_stem}.pdf",
        mime="application/pdf",
    )
    st.caption("Exports work well in email signatures, presentation decks, reports, posters, and profile pages.")
    st.caption(GENERATED_GRAPHICS_CC0_NOTICE)

    details_tab, report_tab, merged_tab, yearly_tab = st.tabs(
        ("Periods", "All period series", "Merged series", "Stripe values")
    )
    with details_tab:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Label": period.label,
                        "Place": period.display_name,
                        "Start": period.start_date,
                        "End": period.end_date,
                        "Latitude": round(period.latitude, 4),
                        "Longitude": round(period.longitude, 4),
                    }
                    for period in periods_preview
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    with report_tab:
        st.caption(
            "Each entered period keeps its own monthly temperature, climatology, and "
            "anomaly columns. When two periods point to the same place, they reuse the "
            f"same {shared_climatology_label}."
        )
        if aggregation_mode == "rolling_365_day":
            st.caption(
                "These reports stay monthly. The 365-day moving average only affects the "
                "stripe preview above."
            )
        st.dataframe(
            _format_all_periods_report(all_periods_report),
            width="stretch",
            hide_index=True,
        )
    with merged_tab:
        st.caption(
            "This merged monthly view follows only the active period schedule. Months "
            "split across moves reflect the active daily series in that month."
        )
        st.dataframe(
            _format_merged_report(merged_report),
            width="stretch",
            hide_index=True,
        )
    with yearly_tab:
        yearly_display = stripe_frame.copy()
        if "sample_date" in yearly_display.columns:
            yearly_display["sample_date"] = yearly_display["sample_date"].astype(str)
        yearly_display["window_start"] = yearly_display["window_start"].astype(str)
        yearly_display["window_end"] = yearly_display["window_end"].astype(str)
        yearly_display["baseline_c"] = yearly_display["baseline_c"].round(2)
        yearly_display["mean_temp_c"] = yearly_display["mean_temp_c"].round(2)
        yearly_display["anomaly_c"] = yearly_display["anomaly_c"].round(2)
        yearly_display["days_covered"] = yearly_display["days_covered"].round(0).astype(int)
        yearly_display["months_covered"] = yearly_display["months_covered"].round(0).astype(int)
        st.dataframe(yearly_display, width="stretch", hide_index=True)

    # Free the figure after export.
    plt.close(figure)


def _format_all_periods_report(report: pd.DataFrame) -> pd.DataFrame:
    display = report.copy()
    display["sample_date"] = display["sample_date"].astype(str)
    display = display.drop(columns=["timestamp"])
    for column in display.columns:
        if column.endswith("_c"):
            display[column] = pd.to_numeric(display[column], errors="coerce").round(2)
    return display


def _format_merged_report(report: pd.DataFrame) -> pd.DataFrame:
    display = report.copy()
    display["sample_date"] = display["sample_date"].astype(str)
    display = display.drop(columns=["timestamp"])
    for column in ("temperature_c", "climatology_c", "anomaly_c"):
        display[column] = pd.to_numeric(display[column], errors="coerce").round(2)
    display["days_covered"] = pd.to_numeric(display["days_covered"], errors="coerce").round(0).astype(int)
    return display


def _initialize_state(analysis_end: date) -> None:
    if "period_entries" not in st.session_state:
        st.session_state.period_entries = [_blank_entry()]
    if "birth_date" not in st.session_state:
        st.session_state.birth_date = min(date(1990, 1, 1), analysis_end)
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    local_config = load_local_cds_config()
    if "local_cds_url" not in st.session_state:
        st.session_state.local_cds_url = local_config.url if local_config else DEFAULT_CDSAPI_URL
    if "local_cds_key" not in st.session_state:
        st.session_state.local_cds_key = local_config.key if local_config else ""
    if "session_cds_url" not in st.session_state:
        st.session_state.session_cds_url = DEFAULT_CDSAPI_URL
    if "session_cds_token" not in st.session_state:
        st.session_state.session_cds_token = ""
    if "session_cds_user_id" not in st.session_state:
        st.session_state.session_cds_user_id = ""
    if "session_cds_api_key" not in st.session_state:
        st.session_state.session_cds_api_key = ""
    if "storyline_name" not in st.session_state:
        st.session_state.storyline_name = ""
    if "saved_storyline_name" not in st.session_state:
        st.session_state.saved_storyline_name = ""
    if "storyline_cookie_values" not in st.session_state:
        st.session_state.storyline_cookie_values = {}
    if "cookie_consent_payload" not in st.session_state:
        st.session_state.cookie_consent_payload = None
    if "pending_cookie_store_operation" not in st.session_state:
        st.session_state.pending_cookie_store_operation = st.session_state.pop(
            "pending_storyline_cookie_operation",
            None,
        )


def _configured_geoapify_api_key() -> str:
    return str(st.secrets.get("GEOAPIFY_API_KEY", "")).strip()


def _blank_entry() -> dict[str, object]:
    return {
        "place_query": "",
        "resolved_name": "",
        "latitude_text": "",
        "longitude_text": "",
        "coordinate_source": "",
        "indicator_label": "",
        "show_indicator": False,
        "boundary_geojson": None,
        "bounding_box": None,
        "end_date": None,
    }


def _render_cds_access_panel(sidebar) -> CDSConfig | None:
    try:
        persisted_config = resolve_cds_config(st.secrets)
    except CDSCredentialsMissingError:
        persisted_config = None

    session_override = _session_override_config()
    active_cds_config = session_override or persisted_config

    with sidebar.expander("CDS access", expanded=active_cds_config is None):
        st.caption(
            "The CDS licence for the dataset must be accepted first for the account you use. "
            "Session-only credentials entered below stay only in this Streamlit session and "
            "are never written to disk by the app."
        )

        if active_cds_config is None:
            st.warning("No CDS credentials are configured yet.")
        else:
            st.success(f"Using `{active_cds_config.source}` for CDS access.")

        st.text_input(
            "Session CDS API URL",
            key="session_cds_url",
            help="Used only for the current Streamlit session.",
        )
        st.text_input(
            "Session private token",
            key="session_cds_token",
            type="password",
            help="Personal access token, session only, never written to disk.",
        )
        session_columns = st.columns(2)
        session_columns[0].text_input(
            "Session legacy user ID",
            key="session_cds_user_id",
            help="Optional legacy fallback. Session only.",
        )
        session_columns[1].text_input(
            "Session legacy API key",
            key="session_cds_api_key",
            type="password",
            help="Optional legacy fallback. Session only.",
        )
        if st.button("Clear session override", key="clear_session_cds"):
            st.session_state.session_cds_url = DEFAULT_CDSAPI_URL
            st.session_state.session_cds_token = ""
            st.session_state.session_cds_user_id = ""
            st.session_state.session_cds_api_key = ""
            st.success("Removed session-only CDS credentials.")
            st.rerun()

        if (
            not str(st.session_state.session_cds_token).strip()
            and (
                str(st.session_state.session_cds_user_id).strip()
                or str(st.session_state.session_cds_api_key).strip()
            )
            and not _has_complete_legacy_session_credentials()
        ):
            st.warning("To use legacy session credentials, enter both the user ID and the API key.")

        st.text_input(
            "Local CDS API URL",
            key="local_cds_url",
            help="Saved only in .streamlit/local_cds_credentials.toml.",
        )
        st.text_input(
            "Local CDS API token",
            key="local_cds_key",
            type="password",
            help="Saved locally only and excluded from git.",
        )

        action_columns = st.columns(2)
        if action_columns[0].button("Save locally", key="save_local_cds"):
            try:
                save_local_cds_config(
                    key=str(st.session_state.local_cds_key),
                    url=str(st.session_state.local_cds_url),
                )
            except CDSCredentialsMissingError as exc:
                st.error(str(exc))
            else:
                st.success("Saved local CDS credentials.")
                st.rerun()

        if action_columns[1].button("Clear local", key="clear_local_cds"):
            clear_local_cds_config()
            st.session_state.local_cds_url = DEFAULT_CDSAPI_URL
            st.session_state.local_cds_key = ""
            st.success("Removed locally saved CDS credentials.")
            st.rerun()

        st.caption(
            "Local credentials are written to `.streamlit/local_cds_credentials.toml`, which "
            "is gitignored."
        )
        if persisted_config is not None and persisted_config.source == "streamlit_secrets" and session_override is None:
            st.caption(
                "This deployment is using Streamlit secrets. Session-only overrides take "
                "precedence if you enter them here."
            )

    return active_cds_config


def _render_credit_and_license_panel(current_year: int) -> None:
    with st.expander("Credits and licenses", expanded=False):
        st.markdown(
            f"- Copernicus climate stack: `{ERA5_LAND_MONTHLY_DATASET_NAME}` "
            f"{ERA5_LAND_MONTHLY_DATASET_URL} (DOI `{ERA5_LAND_MONTHLY_DATASET_DOI}`) and "
            f"`{ERA5_MONTHLY_DATASET_NAME}` {ERA5_MONTHLY_DATASET_URL} "
            f"(DOI `{ERA5_MONTHLY_DATASET_DOI}`), both distributed through the Climate Data Store "
            "under the CDS `CC-BY` licence.\n"
            f"- Inspiration: {SHOW_YOUR_STRIPES_CREDIT} {SHOW_YOUR_STRIPES_URL}\n"
            f"- Copernicus credit notice: {copernicus_credit_notice(current_year)}\n"
            f"- Underlying ERA5-Land reference: {ERA5_LAND_REFERENCE_CITATION}\n"
            f"- Underlying ERA5 reference: {ERA5_REFERENCE_CITATION}\n"
            f"- Historical fallback dataset: `{TWCR_MONTHLY_DATASET_NAME}` {TWCR_MONTHLY_DATASET_URL}\n"
            f"- NOAA PSL requested 20CRv3 acknowledgment: {TWCR_ACKNOWLEDGEMENT_TEXT}\n"
            f"- 20CRv3 references: {TWCR_REFERENCE_CITATION} {TWCR_FOUNDATIONAL_REFERENCE_CITATION}\n"
            f"- Global-mean reference context: NASA GISTEMP {NASA_GISTEMP_GLOBAL_MEAN_URL}\n"
            f"- NASA GISTEMP credit guidance: {NASA_GISTEMP_WEB_CITATION_GUIDANCE} "
            f"Source credit: `{NASA_GISTEMP_SOURCE_CREDIT}` or `{NASA_GISTEMP_SHORT_SOURCE_CREDIT}`. "
            f"Template: {NASA_GISTEMP_WEB_CITATION_TEMPLATE}\n"
            f"- NASA GISTEMP scholarly reference: {NASA_GISTEMP_REFERENCE_CITATION}\n"
            f"- Generated graphics: {GENERATED_GRAPHICS_CC0_NOTICE}\n"
            f"- Software: {SOFTWARE_MIT_NOTICE}"
        )


def _render_time_series_summary_note() -> None:
    with st.expander("How it works", expanded=False):
        st.caption(
            "MyStripes loads monthly near-surface air temperatures for each place, "
            "using ERA5-Land from 1950 onward, ERA5 for 1940-1949, and 20CRv3 before 1940 when needed. "
            "Historical slices are anomaly-aligned to ERA5-Land, monthly values are expanded to daily "
            "values within each period, merged along the active period schedule, and then compared with "
            "the selected reference period to compute the climatology and anomalies behind the stripes."
        )


def _add_period_entry(analysis_end: date) -> None:
    entries = st.session_state.period_entries
    if entries and entries[-1]["end_date"] is None:
        entries[-1]["end_date"] = analysis_end - timedelta(days=365)
    entries.append(_blank_entry())


def _remove_period_entry() -> None:
    st.session_state.period_entries.pop()
    st.session_state.period_entries[-1]["end_date"] = None


def _apply_geocoding_choice(index: int, result: dict[str, object]) -> None:
    entry = st.session_state.period_entries[index]
    entry["resolved_name"] = str(result["display_name"])
    entry["latitude_text"] = f"{float(result['latitude']):.4f}"
    entry["longitude_text"] = f"{float(result['longitude']):.4f}"
    entry["coordinate_source"] = str(result.get("coordinate_source", "")).strip()
    entry["boundary_geojson"] = result.get("geojson")
    entry["bounding_box"] = result.get("bounding_box")
    st.session_state[f"latitude_{index}"] = entry["latitude_text"]
    st.session_state[f"longitude_{index}"] = entry["longitude_text"]


def _queue_geocoding_search(index: int) -> None:
    st.session_state[f"_pending_geocode_search_{index}"] = True


def _run_geocoding_search(index: int) -> None:
    query = str(st.session_state.get(f"place_query_{index}", "")).strip()
    if not query:
        st.session_state[f"geocode_results_{index}"] = []
        st.session_state.pop(f"geocode_choice_{index}", None)
        raise ValueError("Enter a place name before searching.")

    results = cached_search_places(query, _configured_geoapify_api_key())
    normalized_results = [
        {
            "display_name": result.display_name,
            "latitude": result.latitude,
            "longitude": result.longitude,
            "coordinate_source": result.coordinate_source,
            "geojson": result.geojson,
            "bounding_box": result.bounding_box,
        }
        for result in results
    ]
    st.session_state[f"geocode_results_{index}"] = normalized_results
    if normalized_results:
        st.session_state[f"geocode_choice_{index}"] = normalized_results[0]["display_name"]
        _apply_geocoding_choice(index, normalized_results[0])
    else:
        st.session_state.pop(f"geocode_choice_{index}", None)


def _current_choice_index(results: list[dict[str, object]], resolved_name: str) -> int:
    if not resolved_name:
        return 0
    for index, result in enumerate(results):
        if result["display_name"] == resolved_name:
            return index
    return 0


def _session_override_config() -> CDSConfig | None:
    session_url = str(st.session_state.get("session_cds_url", "")).strip() or DEFAULT_CDSAPI_URL
    session_token = str(st.session_state.get("session_cds_token", "")).strip()
    if session_token:
        return CDSConfig(url=session_url, key=session_token, source="session_private_token")

    if _has_complete_legacy_session_credentials():
        user_id = str(st.session_state.get("session_cds_user_id", "")).strip()
        api_key = str(st.session_state.get("session_cds_api_key", "")).strip()
        return CDSConfig(url=session_url, key=f"{user_id}:{api_key}", source="session_user_credentials")

    return None


def _has_complete_legacy_session_credentials() -> bool:
    user_id = str(st.session_state.get("session_cds_user_id", "")).strip()
    api_key = str(st.session_state.get("session_cds_api_key", "")).strip()
    return bool(user_id and api_key)


def _period_range_label(
    index: int,
    entries: list[dict[str, object]],
    birth_date: date,
    analysis_end: date,
) -> tuple[str, str]:
    current_start = birth_date
    for entry_index, entry in enumerate(entries):
        current_end = analysis_end if entry_index == len(entries) - 1 else entry.get("end_date")
        if entry_index == index:
            start_label = current_start.isoformat()
            end_label = "current" if current_end is None or entry_index == len(entries) - 1 else current_end.isoformat()
            return start_label, end_label
        if isinstance(current_end, date):
            current_start = current_end + timedelta(days=1)
    return birth_date.isoformat(), analysis_end.isoformat()


def _default_indicator_label(entry: dict[str, object], index: int) -> str:
    return (
        str(entry.get("resolved_name", "") or "").strip()
        or str(entry.get("place_query", "") or "").strip()
        or f"Period {index + 1}"
    )


def _boundary_mode_caption(entry: dict[str, object]) -> str:
    boundary_geojson = entry.get("boundary_geojson")
    if isinstance(boundary_geojson, dict) and str(boundary_geojson.get("type", "")).lower() in {
        "polygon",
        "multipolygon",
    }:
        return "Boundary mode will use the selected municipality, district, region, or other place polygon."

    if entry.get("bounding_box") is not None:
        return "Boundary mode will fall back to the geocoder area extent because no polygon boundary was returned."

    return "Boundary mode needs a geocoder result with an area boundary or area extent."


def _resolve_climatology_reference_period(
    reference_mode: str,
    story_line_start: date,
    story_line_end: date,
    dataset_start: date,
    dataset_end: date,
) -> tuple[date, date, str, str, str]:
    if reference_mode == "climate_normal_1961_2010":
        reference_start = max(dataset_start, date(1961, 1, 1))
        reference_end = min(dataset_end, date(2010, 12, 31))
        if reference_end < reference_start:
            raise ValueError(
                "The 1961-2010 climate-normal reference period is not available in the current "
                "climate-data window."
            )
        date_range_label = f"{reference_start.isoformat()} to {reference_end.isoformat()}"
        metric_value = (
            "1961-2010"
            if reference_start == date(1961, 1, 1) and reference_end == date(2010, 12, 31)
            else date_range_label
        )
        return (
            reference_start,
            reference_end,
            f"Per-location day-of-year climatology from {date_range_label}",
            metric_value,
            f"per-location climatology from {date_range_label}",
        )

    if reference_mode == "story_line_period":
        return (
            story_line_start,
            story_line_end,
            "Per-location day-of-year climatology from the full story line period",
            "Story line period",
            "per-location climatology from the full story line period",
        )

    raise ValueError(f"Unsupported climatology reference period: {reference_mode}")


def _render_reference_period_global_warming_warning(
    reference_start: date,
    reference_end: date,
) -> None:
    try:
        annual_estimates = cached_gistemp_global_mean_estimates()
        mean_anomaly_c, start_year, end_year = average_global_warming_for_period(
            annual_estimates,
            period_start=reference_start,
            period_end=reference_end,
        )
    except Exception:
        st.caption(
            "NASA GISTEMP global-mean context is currently unavailable. Source: "
            f"{NASA_GISTEMP_GLOBAL_MEAN_URL}"
        )
        return

    requested_year_span = (reference_start.year, reference_end.year)
    used_year_label = str(start_year) if start_year == end_year else f"{start_year}-{end_year}"
    clipped_year_note = ""
    if requested_year_span != (start_year, end_year):
        requested_year_label = (
            str(reference_start.year)
            if reference_start.year == reference_end.year
            else f"{reference_start.year}-{reference_end.year}"
        )
        clipped_year_note = (
            f" Using overlapping NASA GISTEMP annual values for {used_year_label} "
            f"instead of the full requested span {requested_year_label}."
        )

    st.warning(
        f"The selected reference period ({used_year_label}) is {mean_anomaly_c:+.2f} C "
        "above NASA GISTEMP annual global temperature 1951-1980 baseline."
        f"{clipped_year_note} Source: {NASA_GISTEMP_GLOBAL_MEAN_URL}"
    )


def _aggregation_mode_caption(
    baseline_label: str,
    aggregation_mode: str,
    analysis_end: date,
    rolling_sample_mode: str,
    rolling_strip_count: int | None,
) -> str:
    if aggregation_mode == "rolling_365_day":
        sampling_text = (
            "sampled monthly"
            if rolling_sample_mode == "monthly"
            else f"sampled into {rolling_strip_count or 60} evenly spaced strips"
        )
        return (
            f"Climatology: {baseline_label}. Stripe period: 365-day moving average, "
            f"{sampling_text}, through {analysis_end.isoformat()}."
        )

    return (
        f"Climatology: {baseline_label}. Stripe period: full calendar years only, so "
        "partial birth and current years are omitted."
    )


def _debug_print(
    enabled: bool,
    label: str,
    payload: object | None = None,
    *,
    period_aliases: dict[str, str] | None = None,
) -> None:
    if not enabled:
        return

    print(f"[mystripes debug] {label}", flush=True)
    if payload is None:
        return

    if isinstance(payload, pd.DataFrame):
        print(_format_debug_frame(payload, period_aliases=period_aliases), flush=True)
        return

    if isinstance(payload, list) and payload and all(isinstance(item, pd.DataFrame) for item in payload):
        for index, frame in enumerate(payload, start=1):
            print(f"[mystripes debug] frame {index}", flush=True)
            print(_format_debug_frame(frame, period_aliases=period_aliases), flush=True)
        return

    print(pformat(_apply_period_aliases(payload, period_aliases), sort_dicts=False), flush=True)


def _format_debug_frame(
    frame: pd.DataFrame,
    max_rows: int = 24,
    *,
    period_aliases: dict[str, str] | None = None,
) -> str:
    display = frame.copy()
    if period_aliases:
        display = display.rename(columns=lambda column: _apply_period_aliases_to_text(str(column), period_aliases))
        for column in display.columns:
            if pd.api.types.is_object_dtype(display[column]) or pd.api.types.is_string_dtype(display[column]):
                display[column] = display[column].map(
                    lambda value: _apply_period_aliases_to_text(value, period_aliases)
                    if pd.notna(value)
                    else value
                )

    summary = f"shape={display.shape} columns={list(display.columns)}"
    if frame.empty:
        return summary + "\n<empty>"

    if len(display) <= max_rows:
        return summary + "\n" + display.to_string(index=False)

    head_rows = max_rows // 2
    tail_rows = max_rows - head_rows
    return (
        summary
        + "\n"
        + display.head(head_rows).to_string(index=False)
        + "\n...\n"
        + display.tail(tail_rows).to_string(index=False)
    )


def _build_debug_period_aliases(periods) -> dict[str, str]:
    return {
        f"Period {index + 1}: {period.label}": f"P{index + 1}"
        for index, period in enumerate(periods)
    }


def _build_debug_period_identifications(periods) -> dict[str, str]:
    return {
        f"P{index + 1}": _debug_period_identifier(period)
        for index, period in enumerate(periods)
    }


def _debug_period_identifier(period, max_length: int = 72) -> str:
    identifier = period.label or period.display_name
    if period.display_name and period.display_name != identifier:
        identifier = f"{identifier} | {period.display_name}"
    if len(identifier) <= max_length:
        return identifier
    return identifier[: max_length - 3] + "..."


def _apply_period_aliases(payload: object, period_aliases: dict[str, str] | None) -> object:
    if not period_aliases:
        return payload
    if isinstance(payload, str):
        return _apply_period_aliases_to_text(payload, period_aliases)
    if isinstance(payload, list):
        return [_apply_period_aliases(item, period_aliases) for item in payload]
    if isinstance(payload, tuple):
        return tuple(_apply_period_aliases(item, period_aliases) for item in payload)
    if isinstance(payload, dict):
        return {
            _apply_period_aliases_to_text(key, period_aliases) if isinstance(key, str) else key:
            _apply_period_aliases(value, period_aliases)
            for key, value in payload.items()
        }
    return payload


def _apply_period_aliases_to_text(value: object, period_aliases: dict[str, str]) -> str:
    text = str(value)
    for long_name, short_name in period_aliases.items():
        text = text.replace(long_name, short_name)
    return text


if __name__ == "__main__":
    main()
