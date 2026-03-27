from __future__ import annotations

ERA5_LAND_MONTHLY_DATASET_NAME = "ERA5-Land monthly averaged reanalysis"
ERA5_LAND_MONTHLY_DATASET_URL = (
    "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means"
)
SHOW_YOUR_STRIPES_URL = "https://showyourstripes.info/"
SHOW_YOUR_STRIPES_CREDIT = (
    "Inspired by the #ShowYourStripes project created by Professor Ed Hawkins and "
    "the University of Reading."
)
PROJECT_REPOSITORY_URL = "https://github.com/j-haacker/mystripes"
CONTRIBUTING_GUIDE_URL = f"{PROJECT_REPOSITORY_URL}/blob/main/CONTRIBUTING.md"
ERA5_LAND_REFERENCE_CITATION = (
    "Mu\u00f1oz Sabater, J. et al. (2021): ERA5-Land: a state-of-the-art global "
    "reanalysis dataset for land applications. Earth System Science Data, 13, "
    "4349-4383."
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
