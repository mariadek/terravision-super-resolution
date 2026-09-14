# Sentinel-3 LST Thermal sharpening

This directory contains the **Sentinel-3 Land Surface Temperature (LST) thermal sharpening workflow** developed within the **TERRAVISION Horizon Europe project**. The workflow enhances the spatial resolution of Sentinel-3 LST thermal data from **1 km to 10 m** by exploiting **super-resolved Sentinel-2 multispectral imagery from sentinel-2-l2a-sr-10m** [ICCS STAC Collection sentinel-2-l2a-sr-10m](https://platform-eo.iccs.gr/stac-browser/stac/collections/sentinel-2-l2a-sr-10m).

## Methodology

This thermal sharpening workflow is based on the Data Mining Sharpener (DMS) approach proposed by [Gao et al., 2006](https://doi.org/10.3390/rs4113287). This method utilises SR Sentinel-2 L2A multispectral data as independent features, which are aggregated to match the native 1 km spatial resolution of Sentinel-3 LST data.

```text

                  SENTINEL-3-SLSTR
                         │
                         ▼
                   Find/download
                         │
                         │
SENTINEL-2-SR ───► Find/download
                         │
                         ▼
                   Preprocessing
                         │
             ┌───────────┴──────────┐
             │                      │
         Sentinel-3              Sentinel-2
             │                      │
             └───────────┬──────────┘
                         ▼
                  Thermal sharpening
                         │
                         ▼
                    GeoTIFF/COG
                         │
                         ├──────────────► S3
                         │
                         ▼
                     STAC Item
                         │
                         ▼
                    STAC Catalog
```

## Data

The workflow uses:

* **Sentinel-3 Level-2A (L2A) LST** products at 1 km spatial resolution, retrieved from the CDSE STAC catalogue.
* **SR Sentinel-2  Level-2A (L2A)** multispectral imagery at 10 m spatial resolution, retrieved from the ICCS STAC catalogue and used to provide the high-resolution spatial information for thermal sharpening.

## Installation

Clone via:
```bash
      git clone <repo>
```

1. Create a Python environment (conda):
```bash
      conda env create -f environment.yml
      conda activate thermalsharpening
```

2. Install dependencies with pip
```bash
      pip install -r requirements.txt
```

## Usage

### Python 
```bash
      python src/main.py --user_input ./examples/La_Zarza_aoi.json
```
### Docker
```bash
      USER_INPUT=examples/La_Zarza_aoi.json docker compose run --rm thermalsharpening
```
## Outputs

The primary output is a spatially enhanced **Sentinel-3 LST product** with a target spatial resolution of **10 m**.

## Evaluation


## License

This workflow is distributed under the same open-source license as the main TERRAVISION Super-Resolution repository. See the [`LICENSE`](../LICENSE) file for details.
