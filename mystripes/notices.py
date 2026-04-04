from __future__ import annotations

ERA5_LAND_MONTHLY_DATASET_NAME = "ERA5-Land monthly averaged reanalysis"
ERA5_LAND_MONTHLY_DATASET_URL = (
    "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means"
)
ERA5_LAND_MONTHLY_DATASET_DOI = "10.24381/cds.68d2bb30"
ERA5_MONTHLY_DATASET_NAME = "ERA5 monthly averaged data on single levels"
ERA5_MONTHLY_DATASET_URL = (
    "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means"
)
ERA5_MONTHLY_DATASET_DOI = "10.24381/cds.f17050d7"
TWCR_MONTHLY_DATASET_NAME = "NOAA 20CRv3 monthly 2 m air temperature"
TWCR_MONTHLY_DATASET_URL = (
    "https://psl.noaa.gov/thredds/catalog/Datasets/20thC_ReanV3/Monthlies/2mSI-MO/catalog.html"
)
SHOW_YOUR_STRIPES_URL = "https://showyourstripes.info/"
SHOW_YOUR_STRIPES_CREDIT = (
    "Inspired by the #ShowYourStripes project created by Professor Ed Hawkins and "
    "the University of Reading."
)
PROJECT_REPOSITORY_URL = "https://github.com/j-haacker/mystripes"
PROJECT_ISSUES_URL = f"{PROJECT_REPOSITORY_URL}/issues"
CONTRIBUTING_GUIDE_URL = f"{PROJECT_REPOSITORY_URL}/blob/main/CONTRIBUTING.md"
ERA5_LAND_REFERENCE_CITATION = (
    "Mu\u00f1oz Sabater, J. et al. (2021): ERA5-Land: a state-of-the-art global "
    "reanalysis dataset for land applications. Earth System Science Data, 13, "
    "4349-4383."
)
ERA5_REFERENCE_CITATION = (
    "Hersbach, H. et al. (2020): The ERA5 global reanalysis. Quarterly Journal of "
    "the Royal Meteorological Society, 146, 1999-2049."
)
TWCR_ACKNOWLEDGEMENT_TEXT = (
    "Support for the Twentieth Century Reanalysis Project version 3 dataset is "
    "provided by the U.S. Department of Energy, Office of Science Biological and "
    "Environmental Research (BER), by the National Oceanic and Atmospheric "
    "Administration Climate Program Office, and by the NOAA Physical Sciences "
    "Laboratory."
)
TWCR_REFERENCE_CITATION = (
    "Slivinski, L. C., and Coauthors, 2021: An Evaluation of the Performance of "
    "the Twentieth Century Reanalysis Version 3. Journal of Climate, 34, 1417-1438."
)
TWCR_FOUNDATIONAL_REFERENCE_CITATION = (
    "Compo, G. P., J. S. Whitaker, P. D. Sardeshmukh, and Coauthors, 2011: The "
    "Twentieth Century Reanalysis Project. Quarterly Journal of the Royal "
    "Meteorological Society, 137, 1-28."
)
NASA_GISTEMP_WEB_CITATION_GUIDANCE = (
    "NASA GISS asks users to cite the GISTEMP webpage with an access date and to "
    "cite the most recent scholarly publication about the data."
)
NASA_GISTEMP_WEB_CITATION_TEMPLATE = (
    "GISTEMP Team: GISS Surface Temperature Analysis (GISTEMP), version 4. NASA "
    "Goddard Institute for Space Studies. Dataset accessed YYYY-MM-DD at "
    "https://data.giss.nasa.gov/gistemp/."
)
NASA_GISTEMP_REFERENCE_CITATION = (
    "Lenssen, N., G. A. Schmidt, M. Hendrickson, P. Jacobs, M. Menne, and R. "
    "Ruedy, 2024: A GISTEMPv4 observational uncertainty ensemble. Journal of "
    "Geophysical Research: Atmospheres, 129(17), e2023JD040179."
)
NASA_GISTEMP_SOURCE_CREDIT = "NASA's Goddard Institute for Space Studies"
NASA_GISTEMP_SHORT_SOURCE_CREDIT = "NASA GISS/GISTEMP"
GENERATED_GRAPHICS_CC0_NOTICE = (
    "The generated graphic composition is offered under CC0 1.0 to the extent the "
    "project or the exporting user holds rights in that composition. This does not "
    "remove attribution or licence obligations that apply to the underlying "
    "Copernicus, NOAA PSL / 20CRv3, or NASA GISS / GISTEMP source data."
)
SOFTWARE_MIT_NOTICE = "The software in this repository is licensed under the MIT License."


def copernicus_credit_notice(year: int) -> str:
    return (
        f"Contains modified Copernicus Climate Change Service information {year}. "
        "Neither the European Commission nor ECMWF is responsible for any use that "
        "may be made of the Copernicus information or data it contains."
    )
