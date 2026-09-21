# PRISMA Hyperspectral Pansharpening

This directory contains the **PRISMA hyperspectral pansharpening workflow** developed within the **TERRAVISION Horizon Europe project**. The workflow enhances the spatial resolution of PRISMA hyperspectral imagery from **30 m to 5 m** by exploiting the high-spatial-resolution PRISMA panchromatic image acquired simultaneously with the hyperspectral image.

## Methodology

The workflow combines the spectral information contained in the **PRISMA hyperspectral data** with the higher-resolution spatial information provided by **PRISMA 5 m panchromatic band** using the **Gram-Schmidt Pan Sharpening** method.

```text

                    PRISMA
                      │
                      ▼
                Find/download manually
                      │
                      │
SENTINEL-2 ───► Find/download
                      │
                      ▼
                Preprocessing
                      │
          ┌───────────┴──────────┐
          │                      │
        PRISMA                   S2
          │                      │
          └───────────┬──────────┘
                      ▼
               Co-registration
                      │
                      ▼  
            Common-area cropping
                      │
                      ▼  
            Pansharpening stage 1
                      │
                      ▼  
               Optional AOI crop
                      │
                      ▼  
         Pansharpening reconstruction
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

* **PRISMA Level-2A (L2A) hyperspectral VNIR/SWIR** products at 30 m spatial resolution, retrieved from the PRISMA ASI service.
* **Sentinel-2  Level-2A (L2A)** multispectral imagery at 10 m spatial resolution, retrieved from the Copernicus Data Space Ecosystem (CDSE) STAC catalogue and used to correct geolocation.

## Installation

Clone via:
```bash
      git clone <repo>
```

1. Create a Python environment (conda):
```bash
      conda env create -f environment.yml
      conda activate pansharpen
```

2. Install dependencies with pip
```bash
      pip install -r requirements.txt
```

## Authentication

The workflow downloads data from the Copernicus Data Space Ecosystem (CDSE) and requires authentication.

Credentials should be provided through environment variables and should not be stored directly in the source code or committed to Git.

1. Copernicus Data Space Ecosystem (CDSE)

Create a free CDSE account:

   1. Go to the [Copernicus Data Space Ecosystem](https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/auth?client_id=account-console&redirect_uri=https%3A%2F%2Fidentity.dataspace.copernicus.eu%2Fauth%2Frealms%2FCDSE%2Faccount%2F%23%2Fpersonal-info&state=cea959f4-c939-4f39-bce0-a8757d65d783&response_mode=query&response_type=code&scope=openid&nonce=dc375623-e3cc-4a9a-9f98-40062ef5a9a7&code_challenge=Mm9RCZVAMuUcSbIxg74tyrOXFhJoQ9VC6VEN7EEtP_I&code_challenge_method=S256).
   2. Click Register and create an account.
   3. Verify your email address using the verification email sent by CDSE.
   4. Use the email/username and password associated with your account as your CDSE credentials.

For additional information, see the official CDSE registration documentation.

2. Configure environment variables

Before running the workflow, define the credentials as environment variables.

Using a .env file

If the project supports loading a .env file, create a file named .env in the project root:

CDSE_USERNAME=your_cdse_username
CDSE_PASSWORD=your_cdse_password
ICCS_USERNAME=your_iccs_username
ICCS_PASSWORD=your_iccs_password

Make sure .env is included in .gitignore:

.env

Important: Never commit usernames, passwords, access tokens, or other credentials to the repository.

A .env.example file can be committed to show which variables are required without exposing credentials:

CDSE_USERNAME=
CDSE_PASSWORD=
ICCS_USERNAME=
ICCS_PASSWORD=

## Usage

### Python 
Run with a custom configuration file:

```bash
      python src/main.py --user_input ./examples/La_Zarza_aoi.json --config ./configs/config.yaml
```
Or run without specifying a configuration file:
```bash
      python src/main.py --user_input ./examples/La_Zarza_aoi.json
```
**Note**: The --config argument is optional.

### Docker
```bash
      USER_INPUT=examples/La_Zarza_aoi.json CONFIG=configs/config.yaml docker compose run --rm prisma_pansharp
```
## Outputs

The primary output is a spatially enhanced **PRISMA hyperspectral product** with a target spatial resolution of **5 m**.

## Evaluation

The quality of the pansharpened PRISMA products can be evaluated using complementary **spectral and spatial quality metrics** to assess both the preservation of hyperspectral information and the improvement in spatial detail.

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
