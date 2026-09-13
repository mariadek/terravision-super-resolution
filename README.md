# TERRAVISION Super-Resolution Workflows

<p align="left">
  <img src="TERRAVISION MAIN LOGO 2Colours_1 positive.png" alt="TERRAVISION Logo" height="50">
</p>

This repository contains **Machine Learning (ML) and Deep Learning (DL) super-resolution and spatial enhancement workflows** developed within the **TERRAVISION Horizon Europe project**. TERRAVISION aims to improve the sustainability, efficiency, and environmental performance of the mining industry through advanced **Earth Observation (EO)** technologies, integrating satellite, airborne, and ground-based observations across the mining value chain.

The repository focuses on enhancing the spatial resolution of **multispectral, hyperspectral, and thermal satellite data** while preserving their spectral and physical characteristics. The resulting products are intended to improve the usability of spaceborne EO data for mining-related applications, including mineral exploration, environmental monitoring, risk assessment, and post-closure activities.

---

## Super-Resolution Workflows

The repository currently includes the following spatial enhancement workflows:

### Sentinel-2 Multispectral Super-Resolution

Super-resolution of **Sentinel-2 MSI** imagery to enhance lower-resolution spectral bands to a **common spatial resolution of at least 10 m**, while preserving their spectral characteristics.

### PRISMA Hyperspectral Pansharpening

Enhancement of **PRISMA hyperspectral imagery from 30 m to 5 m** by fusing hyperspectral observations with the simultaneously acquired high-resolution **panchromatic (PAN) band**, while preserving spectral information.

### EnMAP Hyperspectral Spatial Enhancement

ML/DL-based spatial enhancement of **EnMAP hyperspectral imagery**, aiming to increase spatial detail while maintaining the spectral fidelity required for hyperspectral analysis.

### Sentinel-3 LST Thermal Sharpening

ML-based thermal sharpening of **Sentinel-3 Land Surface Temperature (LST)** products by combining coarse-resolution thermal observations with higher-resolution multispectral information.

---

## Repository Scope

| Satellite / Sensor   | Data                                    | Spatial Enhancement                           |
| -------------------- | ----------------------------------------| ----------------------------------------------|
| **Sentinel-2 / MSI** | Multispectral                           | Super-resolution to 10 m                      |
| **PRISMA**           | Hyperspectral + PAN                     | Pansharpening from 30 m to 5 m                |
| **EnMAP**            | Hyperspectral + PseudoPAN (Sentinel-2)  | Pansharpening from 30 m to 10 m               |
| **Sentinel-3**       | Land Surface Temperature                | ML-based thermal sharpening from 1 km to 10 m |

The workflows are designed to be modular and extensible, allowing additional algorithms, datasets, trained models, and evaluation procedures to be incorporated as the project progresses.

---

## Objectives

The main objectives of this repository are to:

* Develop reproducible **ML/DL super-resolution and data-fusion workflows** for spaceborne EO data.
* Enhance the spatial resolution of **multispectral, hyperspectral, and thermal** satellite products.
* Preserve relevant **spectral and physical information** during spatial enhancement.
* Support the use of enhanced EO products for **mining and raw-materials applications**.
* Provide open and reproducible implementations of the methodologies developed within TERRAVISION.

---

## Repository Structure

```text
terravision-super-resolution/
├── sentinel2/
├── prisma/
├── enmap/
├── sentinel3/
├── LICENSE
└── README.md
```

---

## License

This repository is distributed under an open-source license. See the [LICENSE](LICENSE) file for details.

---

## Project Status

**Under active development.** Additional super-resolution algorithms, trained models, datasets, and evaluation tools will be incorporated as the TERRAVISION project progresses.
