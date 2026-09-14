# EnMAP Hyperspectral Pansharpening

This directory contains the **EnMAP hyperspectral pansharpening workflow** developed within the **TERRAVISION Horizon Europe project**. The workflow enhances the spatial resolution of EnMAP hyperspectral imagery from **30 m to 10 m** by exploiting high-spatial-resolution **Sentinel-2 multispectral imagery**.

## Methodology

The workflow combines the spectral information contained in the **EnMAP hyperspectral data** with the higher-resolution spatial information provided by **Sentinel-2 10 m multispectral bands** using the **Gram-Schmidt Pan Sharpening** method.

```text

EnMAP search
    ↓
Sentinel-2 matching
    ↓
Download
    ↓
EnMAP / Sentinel-2 preprocessing
    ↓
Coregistration
    ↓
Common-area cropping
    ↓
Pansharpening stage 1
    ↓
Optional AOI crop
    ↓
Pansharpening reconstruction
    ↓
Housekeeping

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

* Docker

AOI_FILE=examples/La_Zarza_aoi.json docker compose run --rm enmap_pansharp

* Python 

python src/main.py --aoi ./examples/La_Zarza_aoi.json

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

The quality of the pansharpened EnMAP products can be evaluated using complementary **spectral and spatial quality metrics** to assess both the preservation of hyperspectral information and the improvement in spatial detail.

Example evaluation metrics include:

* **Spectral Angle Mapper (SAM):** measures the spectral similarity between reference and reconstructed hyperspectral pixels. Lower values indicate better spectral preservation.
* **ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse):** measures the overall relative reconstruction error across spectral bands. Lower values indicate better reconstruction quality.
* **Peak Signal-to-Noise Ratio (PSNR):** quantifies reconstruction quality based on the difference between reference and reconstructed imagery. Higher values indicate better performance.
* **Structural Similarity Index (SSIM):** evaluates the preservation of spatial structure, contrast, and local image characteristics. Higher values indicate greater structural similarity.
* **Correlation Coefficient (CC):** measures the correlation between reference and reconstructed spectral bands. Values closer to 1 indicate stronger agreement.
* **Root Mean Square Error (RMSE):** measures the magnitude of reconstruction errors between reference and predicted data. Lower values indicate better agreement.

Where high-resolution reference hyperspectral data are not available, evaluation can also be performed using **reduced-resolution experiments**, in which the original EnMAP data are spatially degraded and subsequently reconstructed to enable comparison against a known reference.

## License

This workflow is distributed under the same open-source license as the main TERRAVISION Super-Resolution repository. See the [`LICENSE`](../LICENSE) file for details.
