from __future__ import annotations

from datetime import date, timedelta
from html import escape
from time import perf_counter
from uuid import uuid4
from pprint import pformat

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from mystripes.cds import (
    CDSCredentialsMissingError,
    CDSRequestError,
    DEFAULT_CDSAPI_URL,
    clear_local_cds_config,
    fetch_saved_temperature_series,
    get_dataset_window,
    load_local_cds_config,
    resolve_cds_config,
    save_local_cds_config,
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
    ERA5_LAND_REFERENCE_CITATION,
    ERA5_LAND_MONTHLY_DATASET_NAME,
    ERA5_LAND_MONTHLY_DATASET_URL,
    GENERATED_GRAPHICS_CC0_NOTICE,
    SHOW_YOUR_STRIPES_CREDIT,
    SHOW_YOUR_STRIPES_URL,
    SOFTWARE_MIT_NOTICE,
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
    return get_dataset_window()


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
        if place_query_key in st.session_state:
            entry["place_query"] = str(st.session_state[place_query_key])
        if latitude_key in st.session_state:
            entry["latitude_text"] = str(st.session_state[latitude_key])
        if longitude_key in st.session_state:
            entry["longitude_text"] = str(st.session_state[longitude_key])
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
            "latitude_",
            "longitude_",
            "place_query_",
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
        entries.append(entry)

    st.session_state.birth_date = clamped_birth_date
    st.session_state.period_entries = entries
    for index, entry in enumerate(entries):
        st.session_state[f"place_query_{index}"] = str(entry.get("place_query", "") or "")
        st.session_state[f"latitude_{index}"] = str(entry.get("latitude_text", "") or "")
        st.session_state[f"longitude_{index}"] = str(entry.get("longitude_text", "") or "")
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
        parts.append(f"{month_count} months/year")

    return ", ".join(parts)


def _describe_temperature_fetch_event(event: dict[str, object]) -> str:
    stage = str(event.get("stage") or "")
    range_index = int(event.get("range_index") or 0)
    range_count = int(event.get("range_count") or 0)
    range_start = str(event.get("range_start") or "")
    range_end = str(event.get("range_end") or "")

    if stage == "timeline_cache_hit":
        return "Using the saved full timeline from local cache."
    if stage == "timeline_fetch_plan":
        missing_range_count = int(event.get("missing_range_count") or 0)
        noun = "segment" if missing_range_count == 1 else "segments"
        if bool(event.get("has_cached_data")):
            return f"Refreshing {missing_range_count} missing timeline {noun} from CDS."
        return f"Preparing {missing_range_count} timeline {noun} from CDS."
    if stage == "missing_range_started":
        return f"Timeline segment {range_index}/{range_count}: {range_start} to {range_end}."
    if stage == "request_cache_hit":
        return f"Using a saved request chunk for {range_start} to {range_end}."
    if stage == "request_plan":
        request_count = int(event.get("request_count") or 0)
        noun = "batch" if request_count == 1 else "batches"
        return f"Need {request_count} CDS request {noun} for this timeline segment."
    if stage == "request_started":
        batch_label = _request_batch_label(event)
        return f"Downloading CDS {batch_label}." if batch_label else "Downloading CDS data."
    if stage == "request_finished":
        batch_label = _request_batch_label(event)
        return f"Finished CDS {batch_label}." if batch_label else "Finished a CDS request batch."
    if stage == "request_failed":
        message = str(event.get("message") or "")
        return message or "A CDS request failed."
    if stage == "point_fetch_completed":
        return "Parsed downloaded monthly data for the current timeline segment."
    if stage == "missing_range_finished":
        return f"Finished timeline segment {range_index}/{range_count}: {range_start} to {range_end}."
    if stage == "timeline_fetch_completed":
        return "Saved the refreshed timeline to the local cache."
    return "Loading ERA5-Land monthly data..."


def _render_temperature_fetch_progress(
    placeholder,
    *,
    total_locations: int,
    completed_locations: int,
    current_location_index: int,
    current_location_label: str,
    current_detail: str,
    completed_request_batches: int,
    current_request_index: int,
    current_request_count: int,
    current_request_complete: bool,
    started_at: float,
    active: bool,
) -> None:
    normalized_total_locations = max(1, total_locations)
    in_location_fraction = 0.0
    if active and current_request_count > 0:
        completed_in_current = current_request_index
        if not current_request_complete:
            completed_in_current = max(0, completed_in_current - 1)
        in_location_fraction = min(1.0, completed_in_current / current_request_count)
    elif active and current_location_label:
        in_location_fraction = min(0.06, 1.0 / normalized_total_locations)

    overall_fraction = min(1.0, (completed_locations + in_location_fraction) / normalized_total_locations)
    display_fraction = overall_fraction
    if active:
        display_fraction = max(display_fraction, min(0.04, 1.0 / normalized_total_locations))

    elapsed_seconds = perf_counter() - started_at
    rate_parts: list[str] = []
    if elapsed_seconds > 0 and completed_locations > 0:
        rate_parts.append(_format_progress_rate(completed_locations * 60.0 / elapsed_seconds, "locations"))
    if elapsed_seconds > 0 and completed_request_batches > 0:
        rate_parts.append(_format_progress_rate(completed_request_batches * 60.0 / elapsed_seconds, "CDS batches"))

    title = "Preparing ERA5-Land monthly series"
    if current_location_label:
        title = f"Location {current_location_index}/{normalized_total_locations}: {current_location_label}"
    if not active and completed_locations >= total_locations:
        title = f"Loaded {completed_locations}/{normalized_total_locations} locations"

    status_line = current_detail or "Loading ERA5-Land monthly data..."
    summary_left = f"{completed_locations}/{normalized_total_locations} places ready"
    summary_right = " | ".join(rate_parts) if rate_parts else "Waiting for the first completed step..."
    elapsed_text = f"Elapsed {_format_progress_duration(elapsed_seconds)}"
    width_percent = max(0.0, min(100.0, display_fraction * 100.0))
    active_class = " mystripes-progress-fill-active" if active else ""

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
</style>
<div class="mystripes-progress-card">
  <div class="mystripes-progress-kicker">ERA5-Land load progress</div>
  <div class="mystripes-progress-title">{escape(title)}</div>
  <div class="mystripes-progress-detail">{escape(status_line)}</div>
  <div class="mystripes-progress-track">
    <div class="mystripes-progress-fill{active_class}" style="width: {width_percent:.1f}%;"></div>
  </div>
  <div class="mystripes-progress-meta">
    <span>{escape(summary_left)}</span>
    <span>{escape(summary_right)}</span>
    <span>{escape(elapsed_text)}</span>
  </div>
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
        "data and export them as minimal graphic in PNG, SVG, or PDF. Use it to communicate "
        "how you or some entity experienced climate change."
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
        min_value=date(1900, 1, 1),
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
            "Monthly data is downloaded from the ERA5-Land monthly dataset. Radius mode "
            "averages grid cells inside the chosen radius. Boundary mode uses the "
            "municipality, district, region, or other place polygon returned by the "
            "geocoder when available, with a bounding-box fallback when only an area "
            "extent is available."
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
            "The app always downloads the full available ERA5-Land series for each unique "
            "place, then derives the selected reference period from that shared data."
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

    periods_preview, preview_errors = build_periods_from_entries(
        entries=st.session_state.period_entries,
        birth_date=birth_date,
        analysis_end=analysis_end,
        analysis_min_start=dataset_window.min_start,
    )
    debug_period_aliases = _build_debug_period_aliases(periods_preview)
    debug_period_identifications = _build_debug_period_identifications(periods_preview)
    if birth_date < dataset_window.min_start:
        st.caption(
            f"ERA5-Land begins on {dataset_window.min_start.isoformat()}, so the stripes start "
            "there even if your timeline start or birth date is earlier."
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

    generate = st.button("Generate stripes", type="primary", disabled=bool(preview_errors))
    if not generate:
        return

    if active_cds_config is None:
        st.error(
            "Missing CDS credentials. Add them to Streamlit secrets, environment variables, "
            "enter a session-only override, or save a local token from the app sidebar."
        )
        _debug_print(debug_mode, "missing_cds_credentials", period_aliases=debug_period_aliases)
        return

    report_start = max(birth_date, dataset_window.min_start)
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
    except ValueError as exc:
        _debug_print(debug_mode, "reference_period_error", str(exc), period_aliases=debug_period_aliases)
        st.error(str(exc))
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
                "start": dataset_window.min_start.isoformat(),
                "end": effective_report_end.isoformat(),
            },
            "aggregation_mode": aggregation_mode,
            "rolling_sample_mode": rolling_sample_mode,
            "rolling_strip_count": rolling_strip_count,
            "cds_source": active_cds_config.source,
            "watermark": {
                "text": watermark_text,
                "horizontal_align": watermark_horizontal_align,
                "vertical_align": watermark_vertical_align,
                "color": watermark_color,
                "opacity": watermark_opacity,
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
    first_period_history_days = 364 if aggregation_mode == "rolling_365_day" else 0
    unique_periods_by_location = {}
    for period in periods_preview:
        unique_periods_by_location.setdefault(period.location_key, period)

    download_progress_placeholder = st.empty()
    download_started_at = perf_counter()
    total_locations = len(unique_periods_by_location)
    completed_locations = 0
    completed_request_batches = 0
    current_location_index = 0
    current_location_label = ""
    current_detail = "Checking saved timelines and determining whether CDS updates are needed."
    current_request_index = 0
    current_request_count = 0
    current_request_complete = False

    def refresh_download_progress(*, active: bool) -> None:
        _render_temperature_fetch_progress(
            download_progress_placeholder,
            total_locations=total_locations,
            completed_locations=completed_locations,
            current_location_index=current_location_index,
            current_location_label=current_location_label,
            current_detail=current_detail,
            completed_request_batches=completed_request_batches,
            current_request_index=current_request_index,
            current_request_count=current_request_count,
            current_request_complete=current_request_complete,
            started_at=download_started_at,
            active=active,
        )

    refresh_download_progress(active=True)

    try:
        location_frames = {}
        for location_position, (location_key, period) in enumerate(unique_periods_by_location.items(), start=1):
            current_location_index = location_position
            current_location_label = period.display_name
            current_detail = "Checking saved full timeline and missing months for this place."
            current_request_index = 0
            current_request_count = 0
            current_request_complete = False
            refresh_download_progress(active=True)

            def _on_fetch_progress(
                event: dict[str, object],
                *,
                location_label: str = period.display_name,
                location_index: int = location_position,
            ) -> None:
                nonlocal completed_request_batches
                nonlocal current_detail
                nonlocal current_location_index
                nonlocal current_location_label
                nonlocal current_request_complete
                nonlocal current_request_count
                nonlocal current_request_index

                current_location_index = location_index
                current_location_label = location_label
                stage = str(event.get("stage") or "")
                request_count = int(event.get("request_count") or 0)
                request_index = int(event.get("request_index") or 0)
                if request_count > 0:
                    current_request_count = request_count
                if request_index > 0:
                    current_request_index = request_index
                if stage == "request_finished":
                    completed_request_batches += 1
                current_request_complete = stage in {
                    "request_cache_hit",
                    "request_finished",
                    "point_fetch_completed",
                    "missing_range_finished",
                    "timeline_cache_hit",
                    "timeline_fetch_completed",
                }
                current_detail = _describe_temperature_fetch_event(event)
                refresh_download_progress(active=True)

            location_frames[location_key] = fetch_saved_temperature_series(
                config=active_cds_config,
                latitude=period.latitude,
                longitude=period.longitude,
                start_date=dataset_window.min_start,
                end_date=effective_report_end,
                spatial_mode=spatial_mode,
                radius_km=radius_km,
                boundary_geojson=period.boundary_geojson,
                boundary_bbox=period.bounding_box,
                progress_callback=_on_fetch_progress,
            )
            completed_locations += 1
            current_detail = f"{period.display_name} is ready."
            current_request_complete = True
            refresh_download_progress(active=completed_locations < total_locations)

        period_frames = [location_frames[period.location_key] for period in periods_preview]
    except (CDSRequestError, ValueError) as exc:
        _debug_print(debug_mode, "period_fetch_error", str(exc), period_aliases=debug_period_aliases)
        current_detail = (
            f"Stopped while loading {current_location_label or 'the selected places'}: {exc}"
        )
        current_request_complete = True
        refresh_download_progress(active=False)
        st.error(str(exc))
        return

    current_location_index = total_locations
    current_location_label = "All selected places"
    current_detail = "All requested ERA5-Land monthly series are ready."
    current_request_index = 0
    current_request_count = 0
    current_request_complete = True
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
        watermark_max_width_ratio=watermark_max_width_ratio,
        watermark_max_height_ratio=watermark_max_height_ratio,
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
            f"- Climate data access: `{ERA5_LAND_MONTHLY_DATASET_NAME}`\n"
            f"- Dataset page: {ERA5_LAND_MONTHLY_DATASET_URL}\n"
            f"- Inspiration: {SHOW_YOUR_STRIPES_CREDIT} {SHOW_YOUR_STRIPES_URL}\n"
            f"- Copernicus credit notice: {copernicus_credit_notice(current_year)}\n"
            f"- Underlying ERA5-Land reference: {ERA5_LAND_REFERENCE_CITATION}\n"
            f"- Generated graphics: {GENERATED_GRAPHICS_CC0_NOTICE}\n"
            f"- Software: {SOFTWARE_MIT_NOTICE}"
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
                "ERA5-Land dataset window."
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
