# EnMAP Hyperspectral Pansharpening

This directory contains the **EnMAP hyperspectral pansharpening workflow** developed within the **TERRAVISION Horizon Europe project**. The workflow enhances the spatial resolution of EnMAP hyperspectral imagery from **30 m to 10 m** by exploiting high-spatial-resolution **Sentinel-2 multispectral imagery**.

## Methodology

The workflow combines the spectral information contained in the **EnMAP hyperspectral data** with the higher-resolution spatial information provided by **Sentinel-2 10 m multispectral bands** using the **Gram-Schmidt Pan Sharpening** method.

```text
                    ENMAP
                      │
                      ▼
                Find/download
                      │
                      │
SENTINEL-2 ───► Find/download
                      │
                      ▼
                Preprocessing
                      │
          ┌───────────┴──────────┐
          │                      │
        EnMAP                   S2
          │                      │
          └───────────┬──────────┘
                      ▼
               Co-registration
                      │
                      ▼
              GSA Pansharpening
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

* **EnMAP Level-2A (L2A) hyperspectral VNIR/SWIR** products at 30 m spatial resolution, retrieved from the DLR STAC catalogue.
* **Sentinel-2  Level-2A (L2A)** multispectral imagery at 10 m spatial resolution, retrieved from the Copernicus Data Space Ecosystem (CDSE) STAC catalogue and used to provide the high-resolution spatial information for pansharpening.

## Installation

Installation instructions and workflow-specific dependencies are provided in this section.

## Usage

### Data Preparation

Instructions for preparing, preprocessing, and spatially aligning EnMAP and Sentinel-2 data.

### Training

Instructions for training the super-resolution model.

### Inference

Instructions for applying a trained model to EnMAP imagery.

## Outputs

The primary output is a spatially enhanced **EnMAP hyperspectral product** with a target spatial resolution of **10 m**.

## Evaluation

Spatial and spectral quality metrics used to evaluate the pansharpened products are documented here.

## License

This workflow is distributed under the same open-source license as the main TERRAVISION Super-Resolution repository. See the [`LICENSE`](../LICENSE) file for details.
