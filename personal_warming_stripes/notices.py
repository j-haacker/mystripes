from __future__ import annotations

ERA5_LAND_TIMESERIES_DATASET_NAME = "ERA5 Land hourly time-series data from 1950 to present"
ERA5_LAND_TIMESERIES_DATASET_URL = (
    "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-timeseries"
)
ERA5_LAND_REFERENCE_CITATION = (
    "Mu\u00f1oz Sabater, J. (2019): ERA5-Land hourly data from 1950 to present. "
    "Copernicus Climate Change Service (C3S) Climate Data Store (CDS). "
    "DOI: 10.24381/cds.e2161bac."
)
GENERATED_GRAPHICS_CC0_NOTICE = (
    "The generated graphic composition is offered under CC0 1.0 to the extent the "
    "project or the exporting user holds rights in that composition. This does not "
    "remove attribution or licence obligations that apply to the underlying "
    "Copernicus / ERA5-Land source data."
)
SOFTWARE_MIT_NOTICE = "The software in this repository is licensed under the MIT License."


def copernicus_credit_notice(year: int) -> str:
    return (
        f"Contains modified Copernicus Climate Change Service information {year}. "
        "Neither the European Commission nor ECMWF is responsible for any use that "
        "may be made of the Copernicus information or data it contains."
    )
