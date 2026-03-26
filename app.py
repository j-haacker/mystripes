from __future__ import annotations

from datetime import date, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from personal_warming_stripes.cds import (
    CDSCredentialsMissingError,
    CDSRequestError,
    DEFAULT_CDSAPI_URL,
    clear_local_cds_config,
    fetch_point_temperature_series,
    get_dataset_window,
    load_local_cds_config,
    resolve_cds_config,
    save_local_cds_config,
)
from personal_warming_stripes.geocoding import search_places
from personal_warming_stripes.models import CDSConfig
from personal_warming_stripes.notices import (
    ERA5_LAND_REFERENCE_CITATION,
    ERA5_LAND_MONTHLY_DATASET_NAME,
    ERA5_LAND_MONTHLY_DATASET_URL,
    GENERATED_GRAPHICS_CC0_NOTICE,
    SOFTWARE_MIT_NOTICE,
    copernicus_credit_notice,
)
from personal_warming_stripes.plotting import export_figure_bytes, render_stripes_figure
from personal_warming_stripes.processing import (
    build_periods_from_entries,
    build_stripe_frame,
    calculate_series_mean_temperature,
    calculate_life_period_baseline,
    calculate_weighted_location_baseline,
    combine_period_frames,
    unique_locations,
)

st.set_page_config(
    page_title="Personal Warming Stripes",
    page_icon="||",
    layout="wide",
)


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def cached_dataset_window():
    return get_dataset_window()


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def cached_search_places(query: str):
    return search_places(query)


def main() -> None:
    dataset_window = cached_dataset_window()
    today = date.today()
    analysis_end = min(today, dataset_window.max_end)

    st.title("Personal warming stripes")
    st.write(
        "Generate warming stripes from your own life history using ERA5-Land monthly "
        "temperature data. Add the places you have lived, the dates when you moved, "
        "and export a minimal stripe graphic in PNG, SVG, or PDF."
    )

    if today > dataset_window.max_end:
        st.info(
            f"ERA5-Land monthly data is currently available through "
            f"`{dataset_window.max_end.isoformat()}`. The current period ends there."
        )

    _initialize_state(analysis_end)

    sidebar = st.sidebar
    active_cds_config = _render_cds_access_panel(sidebar)
    sidebar.header("Output")
    birth_date = sidebar.date_input(
        "Birth date",
        value=st.session_state.birth_date,
        min_value=date(1900, 1, 1),
        max_value=analysis_end,
        key="birth_date",
    )
    file_stem = sidebar.text_input("Download name", value="personal-warming-stripes")
    baseline_mode = sidebar.selectbox(
        "Baseline",
        options=(
            "Life-period mean",
            "Weighted 1961-2010 climate normal",
        ),
        help=(
            "The weighted 1961-2010 option makes extra ERA5-Land requests for each unique "
            "location, using the same spatial aggregation mode, and weights them by how "
            "long you lived there."
        ),
    )
    width_px = int(sidebar.number_input("Width (px)", min_value=600, max_value=6000, value=1800, step=100))
    height_px = int(sidebar.number_input("Height (px)", min_value=80, max_value=2400, value=260, step=20))
    png_dpi = int(sidebar.number_input("PNG DPI", min_value=72, max_value=600, value=200, step=10))
    transparent_background = sidebar.checkbox("Transparent background", value=False)
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

    if active_cds_config is None:
        st.info(
            "This app needs a Copernicus Climate Data Store account and API key. Save a local "
            "token from the sidebar for development, enter a session-only override, or "
            "configure `CDSAPI_KEY` in Streamlit secrets for deployment."
        )
    else:
        st.info(
            f"Using CDS credentials from `{active_cds_config.source}`. The app requests "
            "ERA5-Land monthly means from the monthly dataset and snaps the request area to "
            "the native 0.1 degree grid."
        )
    _render_credit_and_license_panel(today.year)

    controls = st.columns((1, 1, 2))
    if controls[0].button("Add period"):
        _add_period_entry(analysis_end)
        st.rerun()
    if controls[1].button("Remove last period", disabled=len(st.session_state.period_entries) == 1):
        _remove_period_entry()
        st.rerun()
    controls[2].caption(
        "Enter one location per life period. The app derives the next period start date from "
        "the previous move date, so you do not need to manage overlapping ranges."
    )

    periods_preview, preview_errors = build_periods_from_entries(
        entries=st.session_state.period_entries,
        birth_date=birth_date,
        analysis_end=analysis_end,
        analysis_min_start=dataset_window.min_start,
    )
    if birth_date < dataset_window.min_start:
        st.caption(
            f"ERA5-Land begins on {dataset_window.min_start.isoformat()}, so the stripes start "
            "there even if your birth date is earlier."
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
            entry["custom_label"] = st.text_input(
                "Optional label",
                value=entry["custom_label"],
                key=f"custom_label_{index}",
                placeholder="Vienna childhood, Berlin years, current home...",
            )
            entry["place_query"] = st.text_input(
                "City or region",
                value=entry["place_query"],
                key=f"place_query_{index}",
                placeholder="Vienna, Austria or Tyrol, Austria",
                help=(
                    "Search by city, state, province, region, or country. Area results use "
                    "a centroid to auto-fill coordinates and can also supply an area "
                    "boundary for boundary aggregation."
                ),
            )

            action_columns = st.columns((1, 2))
            if action_columns[0].button("Find place", key=f"find_place_{index}"):
                if entry["place_query"].strip():
                    try:
                        results = cached_search_places(entry["place_query"].strip())
                    except Exception as exc:
                        st.error(f"Geocoding failed: {exc}")
                        results = []
                    st.session_state[f"geocode_results_{index}"] = [
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
                    if results:
                        _apply_geocoding_choice(index, st.session_state[f"geocode_results_{index}"][0])
                else:
                    st.warning("Enter a place name before searching.")

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
                    "Living here until",
                    value=entry["end_date"] or (analysis_end - timedelta(days=365)),
                    min_value=max(birth_date, dataset_window.min_start),
                    max_value=analysis_end - timedelta(days=1),
                    key=f"end_date_{index}",
                )
            else:
                st.caption(f"Current period ends on `{analysis_end.isoformat()}`.")

    for error in preview_errors:
        st.error(error)

    generate = st.button("Generate stripes", type="primary", disabled=bool(preview_errors))
    if not generate:
        return

    if active_cds_config is None:
        st.error(
            "Missing CDS credentials. Add them to Streamlit secrets, environment variables, "
            "enter a session-only override, or save a local token from the app sidebar."
        )
        return

    with st.spinner("Downloading ERA5-Land monthly data for your life periods..."):
        try:
            period_frames = [
                fetch_point_temperature_series(
                    config=active_cds_config,
                    latitude=period.latitude,
                    longitude=period.longitude,
                    start_date=period.start_date,
                    end_date=period.end_date,
                    spatial_mode=spatial_mode,
                    radius_km=radius_km,
                    boundary_geojson=period.boundary_geojson,
                    boundary_bbox=period.bounding_box,
                )
                for period in periods_preview
            ]
            combined, yearly = combine_period_frames(periods_preview, period_frames)
        except (CDSRequestError, ValueError) as exc:
            st.error(str(exc))
            return

    baseline_label = baseline_mode
    if baseline_mode == "Life-period mean":
        baseline_c = calculate_life_period_baseline(yearly)
    else:
        with st.spinner("Calculating weighted 1961-2010 location baseline..."):
            try:
                baseline_by_location: dict[str, float] = {}
                periods_by_location = {period.location_key: period for period in periods_preview}
                for location_key, (latitude, longitude) in unique_locations(periods_preview).items():
                    period = periods_by_location[location_key]
                    baseline_series = fetch_point_temperature_series(
                        config=active_cds_config,
                        latitude=latitude,
                        longitude=longitude,
                        start_date=date(1961, 1, 1),
                        end_date=date(2010, 12, 31),
                        spatial_mode=spatial_mode,
                        radius_km=radius_km,
                        boundary_geojson=period.boundary_geojson,
                        boundary_bbox=period.bounding_box,
                    )
                    baseline_by_location[location_key] = calculate_series_mean_temperature(baseline_series)
                baseline_c = calculate_weighted_location_baseline(periods_preview, baseline_by_location)
            except CDSRequestError as exc:
                st.error(str(exc))
                return

    stripe_frame = build_stripe_frame(yearly, baseline_c)
    width_inches = width_px / png_dpi
    height_inches = height_px / png_dpi
    figure = render_stripes_figure(
        anomalies=stripe_frame["anomaly_c"].tolist(),
        width_inches=width_inches,
        height_inches=height_inches,
        transparent_background=transparent_background,
    )

    png_bytes = export_figure_bytes(figure, "png", png_dpi, transparent_background)
    svg_bytes = export_figure_bytes(figure, "svg", png_dpi, transparent_background)
    pdf_bytes = export_figure_bytes(figure, "pdf", png_dpi, transparent_background)

    st.subheader("Preview")
    st.image(png_bytes, use_container_width=True)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Years shown", int(stripe_frame["year"].count()))
    metric_columns[1].metric("Baseline", f"{baseline_c:.2f} C")
    metric_columns[2].metric("Warmest anomaly", f"{stripe_frame['anomaly_c'].max():+.2f} C")
    metric_columns[3].metric("Coolest anomaly", f"{stripe_frame['anomaly_c'].min():+.2f} C")
    st.caption(
        f"Baseline mode: {baseline_label}. Current year is included as a partial year through "
        f"{analysis_end.isoformat()}."
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
    st.caption(GENERATED_GRAPHICS_CC0_NOTICE)

    details_tab, yearly_tab = st.tabs(("Periods", "Annual values"))
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
            use_container_width=True,
            hide_index=True,
        )
    with yearly_tab:
        yearly_display = stripe_frame.copy()
        yearly_display["mean_temp_c"] = yearly_display["mean_temp_c"].round(2)
        yearly_display["anomaly_c"] = yearly_display["anomaly_c"].round(2)
        yearly_display["days_covered"] = yearly_display["days_covered"].round(0).astype(int)
        yearly_display["months_covered"] = yearly_display["months_covered"].round(0).astype(int)
        st.dataframe(yearly_display, use_container_width=True, hide_index=True)

    # Free the figure after export.
    plt.close(figure)


def _initialize_state(analysis_end: date) -> None:
    if "period_entries" not in st.session_state:
        st.session_state.period_entries = [_blank_entry()]
    if "birth_date" not in st.session_state:
        st.session_state.birth_date = min(date(1990, 1, 1), analysis_end)
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


def _blank_entry() -> dict[str, object]:
    return {
        "custom_label": "",
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


if __name__ == "__main__":
    main()
