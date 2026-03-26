# Personal Warming Stripes

This app lets non-technical users create a warming-stripes graphic from their own life history. They enter the places they have lived, the dates when they moved, and the app fetches ERA5-Land point time series from the Copernicus Climate Data Store before producing a minimalist stripe export.

## Why this stack

- `Streamlit` keeps the UI simple enough for non-technical users and is easy to deploy from a GitHub repo.
- `cdsapi` uses the official Copernicus Climate Data Store API flow for ERA5-Land downloads.
- `matplotlib` gives exact export control for PNG, SVG, and PDF.

## What the app does

- Accepts a flexible number of living periods between birth and the latest available ERA5-Land date.
- Lets users search for places by name or enter coordinates manually.
- Pulls ERA5-Land `2m_temperature` point series for each period.
- Aggregates hourly data into yearly means.
- Colors each year as a warming stripe against either:
  - the average over the user's own life-period series, or
  - a weighted 1961-2010 baseline across the locations they lived in.
- Exports the graphic as `PNG`, `SVG`, and `PDF`.
- Lets the operator adjust width, height, and PNG DPI.

## Local run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create `.streamlit/secrets.toml` from the example file:

   ```toml
   CDSAPI_URL = "https://cds.climate.copernicus.eu/api"
   CDSAPI_KEY = "your-user-id:your-api-key"
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

## Deployment

Recommended host: Streamlit Community Cloud.

Why:

- It deploys directly from a public GitHub repository.
- It supports per-app secrets in the web UI, which is enough for the CDS API key.
- It is the lowest-friction option for a small public Python app with a Streamlit frontend.

Deployment steps:

1. Push this repository to GitHub.
2. Create a new app in Streamlit Community Cloud and point it at `app.py`.
3. Add `CDSAPI_KEY` and optional `CDSAPI_URL` in the app secrets.
4. Redeploy.

Fallback host: Hugging Face Spaces with the Streamlit SDK if you want another free public host.

## Data and method notes

- Climate data source: ERA5-Land hourly point time series from the Copernicus Climate Data Store.
- Geocoding source: OpenStreetMap Nominatim search API.
- The CDS point service returns the nearest native grid cell, not a station record.
- The current year is shown as a partial year through the latest date exposed by the CDS point dataset.
- If the person was born before `1950-01-02`, the stripes start at `1950-01-02` because ERA5-Land point data does not go earlier.

## Operational notes

- The CDS operator must accept the dataset terms of use in their CDS account before the API key works.
- The Nominatim public service is intended for low-volume use. If traffic grows, switch to a dedicated geocoding service or your own Nominatim instance.
