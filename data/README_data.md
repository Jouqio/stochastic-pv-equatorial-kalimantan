# Data Documentation

## Source
**NASA POWER** (Prediction Of Worldwide Energy Resources) — version 8.2.1  
URL: https://power.larc.nasa.gov/data-access-viewer/

## Location
| Parameter | Value |
|-----------|-------|
| Site name | Bontang, East Kalimantan, Indonesia |
| Latitude  | 0.1333°N |
| Longitude | 117.50°E |
| Köppen climate | Af (Equatorial Rainforest) |
| Elevation | ~15 m a.s.l. |

## Temporal Coverage
| Field | Value |
|-------|-------|
| Start | 1 January 2015 |
| End   | 31 December 2025 |
| Resolution | Monthly averaged |
| Total records | 132 months |

## Parameters

| Column | NASA Code | Description | Units |
|--------|-----------|-------------|-------|
| `GHI` | `ALLSKY_SFC_SW_DWN` | All-sky global horizontal irradiance | kWh/m²/day |
| `T2M` | `T2M` | Temperature at 2 m above surface | °C |
| `RH2M` | `RH2M` | Relative humidity at 2 m | % |
| `WS10M` | `WS10M` | Wind speed at 10 m | m/s |
| `CLOUD_AMT` | `CLOUD_AMT` | Cloud amount (fraction) | % (0–100) |
| `PRECTOTCORR` | `PRECTOTCORR` | Precipitation corrected | mm/day |

## Validation

NASA POWER GHI validation against ground-based pyranometers in tropical Af climates:
- Mean Bias Error (MBE): −3.1% to +4.2%
- RMSE: 16.8–18.3% (monthly averages)
- RMSE during monsoon transitions: 24.1%
- RMSE during dry season: 12.6%

Reference: Urraca et al. (2018). *Solar Energy*, 164, 339–354.

## Climate Indices (Exogenous Variables)

| Index | Full Name | Source URL |
|-------|-----------|------------|
| `ONI` | Oceanic Niño Index (Niño3.4 SST anomaly) | https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php |
| `DMI` | Dipole Mode Index (Indian Ocean Dipole) | https://psl.noaa.gov/gcos_wgsp/Timeseries/DMI/ |

## ENSO Events in Dataset

| Event | Period | Peak ONI | PR Impact |
|-------|--------|----------|-----------|
| El Niño | Jan 2015 – May 2016 | +2.6°C (Nov 2015) | PR ↑ +3% (dry suppresses clouds) |
| Neutral | Jun 2016 – Dec 2019 | ±0.5°C | PR ≈ climatology |
| La Niña | Jan 2020 – Dec 2023 | −1.0°C (34 months) | PR ↓ −1.2% (enhanced cloudiness) |
| Recovery | Jan 2024 – Dec 2025 | −0.3°C | PR recovering |

## How to Access Raw Data

1. Go to https://power.larc.nasa.gov/data-access-viewer/
2. Select **Point** → **Monthly Averaged**
3. Enter coordinates: Lat 0.1333, Lon 117.50
4. Date range: 20150101 to 20251231
5. Parameters: `ALLSKY_SFC_SW_DWN, T2M, RH2M, WS10M, CLOUD_AMT, PRECTOTCORR`
6. Format: CSV → Download

## Citation

Stackhouse, P. W., Jr., et al. (2018). POWER Release 8 (with GIS Applications) Methodology.
*NASA Technical Report NASA/TP-2018-219027*. https://ntrs.nasa.gov/citations/20180005851
