import os
import shutil
import requests
from pathlib import Path
from pystac_client import Client

from urllib.parse import urlparse
import urllib3

CDSE_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
S3_COLLECTION = "sentinel-3-sl-2-lst-ntc"


class Sentinel3Downloader:

    """ 
    Search and download Sentinel-3 scenes. 

    This class intentionally separates: 
    1. Search parameters 
    2. Remote API/search results 
    3. Downloading scene data 
    """

    def __init__(self):
        self.username = os.environ["CDSE_CLIENT_ID"]
        self.password = os.environ["CDSE_CLIENT_SECRET"]

    def search(self, aoi_geojson, datetime=None, query=None):

        catalog = Client.open(CDSE_STAC_URL)

        search = catalog.search(
            collections = [S3_COLLECTION],
            intersects =  aoi_geojson,
            max_items=5,
            datetime = datetime,
            query= query if query is not None else None, 
            )

        items = search.item_collection()

        return items

    def get_keycloak(self):
        """Obtain Keycloak token from the Copernicus Identity Service."""
        data = {
            "client_id": "cdse-public",
            "username": self.username,
            "password": self.password,
            "grant_type": "password"
        }

        try:
            r = requests.post(
                "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                data=data
            )
            r.raise_for_status()
            return r.json()['access_token']
        except Exception as e:
            raise Exception(f"Keycloak token creation failed. Server response: {r.text}") from e

    def download_item(self, item, download_root):
         """Download the Product asset of a STAC item, unless already downloaded."""
 
         token = self.get_keycloak()
         http = urllib3.PoolManager()
 
         for asset_name, asset in item.assets.items():
 
             if asset_name.lower() != "product":
                 continue
 
             download_url = asset.href
 
             # Create:
             # download_root / collection_id / item_id
             download_path = os.path.join(
                         download_root,
                         item.id,
                     )
             
             os.makedirs(download_path, exist_ok=True)
 
             # Get filename from URL
             filename = f"{item.id}.zip"
 
             if not filename:
                 raise RuntimeError(
                     f"Could not determine filename from URL: {download_url}"
                 )
 
             output_path = os.path.join(
                             download_path,
                             filename
                         )
             temp_path = Path(f"{output_path}.part")
 
             # Don't download an existing file again
             if Path(output_path).exists():
                 print(f"Already downloaded: {output_path}")
                 return str(output_path)
 
             # Remove stale partial download, if one exists
             if temp_path.exists():
                 print(f"Removing incomplete download: {temp_path}")
                 temp_path.unlink()
 
             print(f"Downloading {item.id}")
             print(f"Destination: {output_path}")
 
             try:
                 with http.request(
                     "GET",
                     download_url,
                     headers={
                         "Authorization": f"Bearer {token}",
                     },
                     preload_content=False,
                     redirect=True,
                     timeout=urllib3.Timeout(
                         connect=30.0,
                         read=300.0,
                     ),
                 ) as response:
 
                     if response.status != 200:
                         raise RuntimeError(
                             f"Download failed for {item.id}: "
                             f"HTTP {response.status}"
                         )
 
                     with open(temp_path, "wb") as out_file:
                         shutil.copyfileobj(
                             response,
                             out_file,
                             length=1024 * 1024,  # 1 MB chunks
                         )
 
                 # Only make the file official after successful download
                 os.replace(temp_path, output_path)
 
                 print(f"Downloaded successfully: {output_path}")
 
             except Exception:
                 # Remove incomplete download
                 if temp_path.exists():
                     temp_path.unlink()
 
                 raise
 
             return str(output_path)
 
         raise ValueError(
             f"No 'Product' asset found for STAC item {item.id}"
         )
 
 