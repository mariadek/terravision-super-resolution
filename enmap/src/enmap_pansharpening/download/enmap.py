import os
import urllib3
import shutil
from urllib.parse import urlparse

from pathlib import Path
from pystac_client import Client

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

DLR_STAC_URL = "https://geoservice.dlr.de/eoc/ogc/stac/v1/"
ENMAP_COLLECTION = "ENMAP_HSI_L2A"

class EnMAPDownloader:

    """ 
    Search and download EnMAP scenes. 

    This class intentionally separates: 
    1. Search parameters 
    2. Remote API/search results 
    3. Downloading scene data 
    """

    def __init__(self):
        self.username = os.environ["ENMAP_USERNAME"]
        self.password = os.environ["ENMAP_PASSWORD"]

    def search(self, aoi_geojson,  datetime=None):

        catalog = Client.open(DLR_STAC_URL)

        search = catalog.search(
            collections = [ENMAP_COLLECTION],
            intersects =  aoi_geojson,
            datetime = datetime,
            )

        items = search.item_collection()

        return items



    def download_item(self, item, download_root):
        """
        Download all assets of a STAC item.

        Parameters
        ----------
        item : pystac.Item
            STAC item to download.
        download_root : str
            Root directory where the item will be downloaded.

        Returns
        -------
        str
            Path to the downloaded item directory.
        """

        http = urllib3.PoolManager()

        header = urllib3.make_headers(
            basic_auth=f"{self.username}:{self.password}"
        )

        download_path = os.path.join(
            download_root,
            item.id,
        )

        os.makedirs(download_path, exist_ok=True)

        print(f"Item: {item.id}")
        print(f"Destination: {download_path}")

        for name, asset in item.assets.items():

            filename = os.path.basename(
                urlparse(asset.href).path
            )

            output_path = os.path.join(
                download_path,
                filename
            )

            temp_path = output_path + ".part"

            # Check if already downloaded
            if os.path.exists(output_path):
                print(f"  Already downloaded: {filename}")
                continue

            print(f"  Downloading {name}: {filename}")

            try:
                with http.request(
                    "GET",
                    asset.href,
                    headers=header,
                    preload_content=False,
                ) as response:

                    if response.status == 404:
                        print(f"File not found: {asset.href}")
                        continue

                    if response.status != 200:
                        raise RuntimeError(
                            f"HTTP {response.status} while downloading "
                            f"{asset.href}"
                        )

                    # Download to temporary file first
                    with open(temp_path, "wb") as out_file:
                        shutil.copyfileobj(response, out_file)

                # Only rename after successful download
                os.replace(temp_path, output_path)

            except Exception:
                # Remove incomplete download
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                raise
        '''
        print(
            f"Finished item {item.id}: "
            f"{os.listdir(download_path)}"
        )
        '''

        return download_path