# Personal Warming Stripes

This app lets non-technical users create a warming-stripes graphic from their own life history. They enter the places they have lived, the dates when they moved, and the app fetches ERA5-Land monthly temperature data from the Copernicus Climate Data Store before producing a minimalist stripe export.

## Why this stack

- `Streamlit` keeps the UI simple enough for non-technical users and is easy to deploy from a GitHub repo.
- `cdsapi` uses the official Copernicus Climate Data Store API flow for ERA5-Land downloads.
- `matplotlib` gives exact export control for PNG, SVG, and PDF.

## What the app does

- Accepts a flexible number of living periods between birth and the latest available ERA5-Land date.
- Lets users search for cities, regions, and countries by name or enter coordinates manually.
- Auto-fills coordinates from the geocoder result, using an area centroid when the result is a polygon or multipolygon.
- Pulls ERA5-Land monthly `2m_temperature` data for each period.
- Uses the nearest single ERA5-Land grid cell by default to keep downloads small.
- Optionally averages grid cells in a chosen radius or inside the selected municipality, district, region, or other place boundary when the geocoder returns a usable area geometry.
- Aggregates monthly values into stripe periods weighted by the number of covered days in each month.
- Offers full calendar years only as the default stripe period, or trailing 365-day windows ending on the latest available month-day in each year.
- Colors each year as a warming stripe against either:
  - the average over the user's own life-period series, or
  - a location-specific baseline, so each part of the timeline is compared to the normal climate of the place lived in at that time.
- Supports location-specific reference periods of `1961-2010`, the person's own lifetime window, or a custom date range.
- Exports the graphic as `PNG`, `SVG`, and `PDF`.
- Lets the operator adjust width, height, and PNG DPI.
- Shows a CDS credit notice and a licensing note for both the exported graphics and the software.

## Local run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Choose one credential option:

   Option A: save a token from the app sidebar.
   The app can store your CDS token in `.streamlit/local_cds_credentials.toml`, which is gitignored.

   Option B: enter a session-only token or legacy `user_id` plus API key in the sidebar.
   These values stay only in Streamlit session state, are never written to disk by the app, and are intended as a fallback if the default configured token fails.

   Option C: create `.streamlit/secrets.toml` from the example file:

   ```toml
   CDSAPI_URL = "https://cds.climate.copernicus.eu/api"
   CDSAPI_KEY = "your-personal-access-token"
   ```

3. Make sure the CDS licence for `reanalysis-era5-land-monthly-means` has been accepted for the account whose token you use.

4. Start the app:

   ```bash
   streamlit run app.py
   ```

### Pixi

If you use Pixi, this repository includes [pixi.toml](pixi.toml) with a ready-to-run task:

```bash
pixi run start
```

Useful shortcut:

```bash
pixi run test
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

When Streamlit secrets are available, the app prefers them over any locally saved credential file. A user can still enter a session-only override during their own browser session without changing the deployed secret.

Fallback host: Hugging Face Spaces with the Streamlit SDK if you want another free public host.

## Data and method notes

- Climate data source: ERA5-Land monthly means from the Copernicus Climate Data Store.
- Geocoding source: OpenStreetMap Nominatim search API.
- Single-cell mode requests only the nearest native 0.1 degree ERA5-Land grid cell, not a station record.
- Radius mode and boundary mode request the minimal bounding area needed for the selected cells, then average the matching grid cells for each month.
- Boundary mode uses the place polygon returned by Nominatim when available; otherwise it falls back to the geocoder's area extent.
- Full-calendar-year mode omits partial birth and current years. Trailing 365-day mode instead uses full 365-day windows ending on the latest available month-day in each year.
- If the person was born before the dataset start date, the stripes start at the first available ERA5-Land monthly date.

## Operational notes

- The CDS operator must accept the dataset terms of use in their CDS account before the API key works.
- Current CDS tokens are personal access tokens. They should be stored as the bare token string, not `user:token`.
- The Nominatim public service is intended for low-volume use. If traffic grows, switch to a dedicated geocoding service or your own Nominatim instance.

## Credits and licenses

- This project was inspired by [ShowYourStripes.info](https://showyourstripes.info/), the #ShowYourStripes project by Professor Ed Hawkins and the University of Reading.
- The app includes a CDS credit notice based on the Copernicus / ECMWF attribution guidance.
- See [NOTICE.md](NOTICE.md) for the project data credit notice and dataset references.
- The software is licensed under [LICENSE](LICENSE).
- The generated graphic composition is offered under [LICENSE-graphics-CC0.md](LICENSE-graphics-CC0.md), with the important caveat that ERA5-Land / Copernicus source-data attribution still applies.
