# Data Credit Notice

MyStripes uses monthly climate datasets from the Copernicus Climate Change Service
Climate Data Store, NOAA Physical Sciences Laboratory, and NASA GISS.

This project was inspired by [ShowYourStripes.info](https://showyourstripes.info/), the #ShowYourStripes project by Professor Ed Hawkins and the University of Reading.

## Copernicus climate datasets

MyStripes uses these CDS datasets:

- `ERA5-Land monthly averaged reanalysis`
  <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means>
  DOI: `10.24381/cds.68d2bb30`
- `ERA5 monthly averaged data on single levels`
  <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means>
  DOI: `10.24381/cds.f17050d7`

Both dataset pages currently show the CDS `CC-BY` licence. ECMWF / C3S guidance says
users should cite the relevant CDS catalogue entry and also provide clear and visible
attribution to the Copernicus programme. For public outputs, include:

`Contains modified Copernicus Climate Change Service information [year]. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.`

Underlying references used by this app:

- `Muñoz Sabater, J. et al. (2021): ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. Earth System Science Data, 13, 4349-4383.`
- `Hersbach, H. et al. (2020): The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146, 1999-2049.`

## NOAA 20CRv3 historical fallback

For pre-1940 monthly temperatures, MyStripes uses:

- `NOAA 20CRv3 monthly 2 m air temperature`
  <https://psl.noaa.gov/thredds/catalog/Datasets/20thC_ReanV3/Monthlies/2mSI-MO/catalog.html>

NOAA PSL's `How to Cite Use of 20CR Data` page requests the following
acknowledgment text for papers using version 3:

`Support for the Twentieth Century Reanalysis Project version 3 dataset is provided by the U.S. Department of Energy, Office of Science Biological and Environmental Research (BER), by the National Oceanic and Atmospheric Administration Climate Program Office, and by the NOAA Physical Sciences Laboratory.`

Key references listed on the official 20CR page:

- `Slivinski, L. C., and Coauthors, 2021: An Evaluation of the Performance of the Twentieth Century Reanalysis Version 3. Journal of Climate, 34, 1417-1438.`
- `Compo, G. P., J. S. Whitaker, P. D. Sardeshmukh, and Coauthors, 2011: The Twentieth Century Reanalysis Project. Quarterly Journal of the Royal Meteorological Society, 137, 1-28.`

## NASA GISTEMP global-mean context

The reference-period warming note in the app uses:

- `GISS Surface Temperature Analysis (GISTEMP v4)`
  <https://data.giss.nasa.gov/gistemp/>

NASA GISS asks users to cite the GISTEMP webpage with an access date and to cite the
most recent scholarly publication about the data. The page currently gives this
template for the web citation:

`GISTEMP Team: GISS Surface Temperature Analysis (GISTEMP), version 4. NASA Goddard Institute for Space Studies. Dataset accessed YYYY-MM-DD at https://data.giss.nasa.gov/gistemp/.`

NASA GISS also asks source credit for graphics or derived uses to:

- `NASA's Goddard Institute for Space Studies`
- `NASA GISS/GISTEMP` if space is limited

Recent scholarly reference listed on the official GISTEMP page:

- `Lenssen, N., G. A. Schmidt, M. Hendrickson, P. Jacobs, M. Menne, and R. Ruedy, 2024: A GISTEMPv4 observational uncertainty ensemble. Journal of Geophysical Research: Atmospheres, 129(17), e2023JD040179.`

## Generated graphics

The exported graphic composition is offered under CC0 1.0 as described in [LICENSE-graphics-CC0.md](LICENSE-graphics-CC0.md), to the extent the project or exporting user holds rights in that composition. That dedication does not replace or remove the attribution obligations for the underlying Copernicus, NOAA PSL / 20CRv3, or NASA GISS / GISTEMP source data.
