import logging
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

import urllib3
from pystac_client import Client

logger = logging.getLogger(__name__)

DLR_STAC_URL = "https://geoservice.dlr.de/eoc/ogc/stac/v1/"
ENMAP_COLLECTION = "ENMAP_HSI_L2A"

ALLOWED_ASSETS = {"metadata", "image"}


class EnMAPDownloader:
    """Search and download EnMAP scenes from the DLR STAC catalog."""

    def __init__(self):
        self.username = os.environ["ENMAP_USERNAME"]
        self.password = os.environ["ENMAP_PASSWORD"]

        self.http = urllib3.PoolManager()
        self.headers = urllib3.make_headers(
            basic_auth=f"{self.username}:{self.password}"
        )

    def search(self, aoi_geojson, datetime=None):
        """
        Search for EnMAP L2A products intersecting an AOI.

        Parameters
        ----------
        aoi_geojson : dict
            GeoJSON geometry used for the spatial search.
        datetime : str, optional
            STAC datetime expression, e.g. "2025-01-01/2025-01-31".

        Returns
        -------
        pystac.ItemCollection
            Matching STAC items.
        """
        logger.info("Searching EnMAP L2A products")

        catalog = Client.open(DLR_STAC_URL)

        search = catalog.search(
            collections=[ENMAP_COLLECTION],
            intersects=aoi_geojson,
            datetime=datetime,
        )

        items = search.item_collection()

        logger.info("Found %d EnMAP scene(s)", len(items))

        return items

    def download_item(self, item, download_root):
        """
        Download the metadata and image assets of a STAC item.

        Parameters
        ----------
        item : pystac.Item
            STAC item to download.
        download_root : str | Path
            Root directory for downloaded items.

        Returns
        -------
        Path
            Directory containing the downloaded assets.
        """
        download_path = Path(download_root) / item.id
        download_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Downloading EnMAP scene %s to %s",
            item.id,
            download_path,
        )

        for asset_name in ALLOWED_ASSETS:
            asset = item.assets.get(asset_name)

            if asset is None:
                logger.warning(
                    "Asset %r not available for EnMAP scene %s",
                    asset_name,
                    item.id,
                )
                continue

            filename = Path(urlparse(asset.href).path).name

            if not filename:
                logger.warning(
                    "Invalid URL for EnMAP asset %r: %s",
                    asset_name,
                    asset.href,
                )
                continue

            output_path = download_path / filename
            temp_path = output_path.with_name(
                f"{output_path.name}.part"
            )

            if output_path.exists():
                logger.info(
                    "EnMAP asset already downloaded: %s",
                    output_path,
                )
                continue

            logger.info(
                "Downloading EnMAP asset %r: %s",
                asset_name,
                filename,
            )

            try:
                with self.http.request(
                    "GET",
                    asset.href,
                    headers=self.headers,
                    preload_content=False,
                ) as response:

                    if response.status == 404:
                        logger.warning(
                            "EnMAP asset not found (HTTP 404): %s",
                            asset.href,
                        )
                        continue

                    if response.status != 200:
                        raise RuntimeError(
                            f"HTTP {response.status} while downloading "
                            f"{asset.href}"
                        )

                    with temp_path.open("wb") as out_file:
                        shutil.copyfileobj(
                            response,
                            out_file,
                        )

                # Move into place only after a successful download.
                temp_path.replace(output_path)

                logger.info(
                    "EnMAP asset downloaded successfully: %s",
                    output_path,
                )

            except Exception:
                temp_path.unlink(missing_ok=True)

                logger.exception(
                    "Failed to download EnMAP asset %r for scene %s",
                    asset_name,
                    item.id,
                )

                raise

        logger.info(
            "EnMAP scene download complete: %s",
            download_path,
        )

        return download_path