# MyStripes

![MyStripes preview](https://raw.githubusercontent.com/j-haacker/mystripes/main/mystripes.png)

MyStripes turns a sequence of places and periods into climate strips based on ERA5-Land monthly temperature data. It works well for personal life stations, multi-home stories, projects, teams, tours, campaigns, or any other place-based timeline.

It is directly inspired by [Show Your Stripes](https://showyourstripes.info/), Ed Hawkins' widely shared warming-stripes project. If you have not seen that page yet, go check it out. It is excellent.

The exported strips are intentionally lightweight, so they can be reused in email signatures, presentation decks, reports, posters, profile pages, and web embeds.

## Why this stack

- `Streamlit` keeps the main UI simple enough for non-technical users and easy to deploy from GitHub.
- `mystripes.api` exposes a small Python API for reusable scripts and notebooks.
- `cdsapi` uses the official Copernicus Climate Data Store API flow for ERA5-Land downloads.
- `matplotlib` gives exact export control for PNG, SVG, and PDF.

## What MyStripes does

- Accepts a flexible number of periods and places across one timeline.
- Keeps the personal-life workflow easy, with labels and hints that fit birth, moves, study years, work phases, and current home.
- Lets users search for cities, regions, and countries by name or enter coordinates manually.
- Auto-fills coordinates from the geocoder result, using an area centroid when the result is a polygon or multipolygon.
- Pulls ERA5-Land monthly `2m_temperature` data for each period.
- Uses the nearest single ERA5-Land grid cell by default to keep downloads small.
- Optionally averages grid cells in a chosen radius or inside the selected municipality, district, region, or other place boundary when the geocoder returns a usable area geometry.
- Expands monthly values to daily series before merging periods and aggregating stripe values.
- Offers full calendar years only as the default stripe period, or a 365-day moving average sampled monthly by default or at an evenly spaced fixed strip count.
- Uses a per-location day-of-year climatology built from each location's full downloaded timeline before merging periods into one timeline.
- Caches identical ERA5-Land download requests locally, so regenerating the same strips reuses prior monthly data instead of calling CDS again.
- Exports the graphic as `PNG`, `SVG`, and `PDF`.
- Includes a small Python API for preparing stripe data and plotting or saving the result.

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

4. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## Packaging

This repository now includes [pyproject.toml](pyproject.toml) so the `mystripes` package can be built and published on PyPI.

Typical release steps:

```bash
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

Trusted Publishing is prepared in [.github/workflows/release.yml](.github/workflows/release.yml). After adding this repository as a trusted publisher on PyPI, publishing a release is:

1. Bump the version in [pyproject.toml](pyproject.toml).
2. Commit and push to `main`.
3. Create and push a tag such as `v0.1.0`.
4. Let GitHub Actions build and publish the package to PyPI.

The workflow uses the GitHub environment name `pypi`, so you can add deployment protection rules there if you want a manual approval step before upload.

## Python API

For scripts and notebooks, use `build_stripe_data(...)` to turn periods plus monthly temperature frames into a stripe-data bundle, then pass that bundle to `plot_stripes(...)` to render or save the figure.

```python
from mystripes.api import build_stripe_data, plot_stripes

stripe_data = build_stripe_data(periods=periods, period_data=monthly_frames)
figure = plot_stripes(stripe_data, output_path="mystripes.svg")
```

The first function accepts periods as dicts, lists, or pandas DataFrames, and monthly data as lists, dicts, or pandas DataFrames. The second function plots from the returned bundle and can save `PNG`, `SVG`, or `PDF`.

## Pixi

If you use Pixi, this repository includes [pixi.toml](pixi.toml) with ready-to-run tasks:

```bash
pixi run start
pixi run test
pixi run lock
```

## Deployment

Recommended host for the UI: Streamlit Community Cloud.

Why:

- It deploys directly from a public GitHub repository.
- It supports per-app secrets in the web UI, which is enough for the CDS API key.
- It is the lowest-friction option for a small public Python app with a Streamlit frontend.

Deployment steps:

1. Push this repository to GitHub, ideally under a repo name such as `mystripes`.
2. Create a new app in Streamlit Community Cloud and point it at `app.py`.
3. Add `CDSAPI_KEY`, optional `CDSAPI_URL`, and `GEOAPIFY_API_KEY` in the app secrets.
4. Redeploy.

When Streamlit secrets are available, the app prefers them over any locally saved credential file. A user can still enter a session-only override during their own browser session without changing the deployed secret.

Fallback host: Hugging Face Spaces with the Streamlit SDK if you want another free public host.

## uv.lock

If you want Streamlit Community Cloud to use `uv.lock` for faster installs, you do not need to adopt uv as your normal workflow. After editing dependencies in [pyproject.toml](pyproject.toml), create or refresh the lockfile once from the repo root:

```bash
uv lock
```

If you do not normally use uv, a simple one-off approach is:

```bash
python -m pip install uv
uv lock
python -m pip uninstall uv
```

If you use Pixi here, the same refresh is available as:

```bash
pixi run lock
```

Streamlit Community Cloud checks dependency files in this order: `uv.lock`, `Pipfile`, `environment.yml`, `requirements.txt`, then `pyproject.toml`. So committing `uv.lock` is enough to make the deployment prefer that lockfile over `requirements.txt`.

## Data and method notes

- Climate data source: ERA5-Land monthly means from the Copernicus Climate Data Store.
- Geocoding source: Geoapify geocoding in hosted deployments when `GEOAPIFY_API_KEY` is configured, with OpenStreetMap Nominatim as a fallback.
- Single-cell mode requests only the nearest native 0.1 degree ERA5-Land grid cell, not a station record.
- Radius mode and boundary mode request the minimal bounding area needed for the selected cells, then average the matching grid cells for each month.
- Boundary mode uses the place polygon returned by Nominatim when available; Geoapify-backed searches fall back to the result bounding box because the hosted geocoding response does not provide the same polygon geometry.
- Full-calendar-year mode omits partial first and current years. The 365-day moving-average mode expands monthly values to daily coverage, applies a daily rolling mean, and samples the smoothed series monthly or at an evenly spaced fixed count. When prior data is available, the app keeps up to one year before the displayed start date for the rolling calculation and crops only after smoothing.
- If the timeline starts before the dataset start date, the stripes start at the first available ERA5-Land monthly date.

## Operational notes

- The CDS operator must accept the dataset terms of use in their CDS account before the API key works.
- Current CDS tokens are personal access tokens. They should be stored as the bare token string, not `user:token`.
- Public Nominatim is not a good fit for Streamlit Community Cloud. Configure `GEOAPIFY_API_KEY` for deployment and keep Nominatim as the local fallback.
- Downloaded monthly series are cached under `.mystripes-cache/`, which is local-only and gitignored.
- Geocoding results are cached under the same `.mystripes-cache/` directory to avoid repeated provider requests for identical place searches.

## Credits and licenses

- This project was inspired by [ShowYourStripes.info](https://showyourstripes.info/), the #ShowYourStripes project by Professor Ed Hawkins and the University of Reading.
- The app includes a CDS credit notice based on the Copernicus / ECMWF attribution guidance.
- See [NOTICE.md](NOTICE.md) for the project data credit notice and dataset references.
- The software is licensed under [LICENSE](LICENSE).
- The generated graphic composition is offered under [LICENSE-graphics-CC0.md](LICENSE-graphics-CC0.md), with the important caveat that ERA5-Land / Copernicus source-data attribution still applies.
