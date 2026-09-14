import os
import json
import time
import shutil
import boto3
import requests
from pathlib import Path
from pystac_client import Client
from tqdm import tqdm

from urllib.parse import urlparse
import urllib3

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env.example")


CDSE_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
S2_COLLECTION = "sentinel-2-l2a"


class Sentinel2Downloader: # change to s3 

    """ 
    http://localhost:8888/notebooks/My_Documents/Development_Space/Mineral_Mapping/CDSE_S3_Download.ipynb
    Search and download Sentinel-2 scenes. 

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
            collections = [S2_COLLECTION],
            intersects =  aoi_geojson,
            max_items=5,
            datetime = datetime,
            query= query if query is not None else None, 
            )

        items = search.item_collection()

        return items

    def get_access_token(self, config):
        """
        Retrieve an access token from the authentication server.
        This token is used for subsequent API calls.
        """
        auth_data = {
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
        }
        response = requests.post(config["auth_server_url"], data=auth_data, verify=True, allow_redirects=False)
        if response.status_code == 200:
            return json.loads(response.text)["access_token"]
        else:
            print(f"Failed to retrieve access token. Status code: {response.status_code}")
            exit(1)

    def get_temporary_s3_credentials(self, headers):
        """
        Create temporary S3 credentials by calling the S3 keys manager API.
        """
        credentials_response = requests.post("https://s3-keys-manager.cloudferro.com/api/user/credentials", headers=headers)
        if credentials_response.status_code == 200:
            s3_credentials = credentials_response.json()
            print("Temporary S3 credentials created successfully.")
            print(f"access: {s3_credentials['access_id']}")
            print(f"secret: {s3_credentials['secret']}")
            return s3_credentials
        else:
            print(f"Failed to create temporary S3 credentials. Status code: {credentials_response.status_code}")
            print("Product download aborted.")
            exit(1)

    def traverse_and_download_s3(self, s3_resource, bucket_name, base_s3_path, local_path, failed_downloads):
        """
        Traverse the S3 bucket and download all files under the specified prefix.
        """
        bucket = s3_resource.Bucket(bucket_name)
        files = bucket.objects.filter(Prefix=base_s3_path)

        for obj in files:
            s3_key = obj.key
            #relative_path = os.path.relpath(s3_key, base_s3_path)
            relative_path = base_s3_path.split('/')[-1]
            local_path_file = os.path.join(local_path, relative_path)
            local_dir = os.path.dirname(local_path_file)
            os.makedirs(local_dir, exist_ok=True)
            self.download_file_s3(s3_resource.meta.client, bucket_name, s3_key, local_path_file, failed_downloads)

    def format_filename(self, filename, length=40):
        """
        Format a filename to a fixed length, truncating if necessary.
        """
        if len(filename) > length:
            return filename[:length - 3] + '...'
        else:
            return filename.ljust(length)

    def download_file_s3(self, s3, bucket_name, s3_key, local_path, failed_downloads):
        """
        Download a file from S3 with a progress bar.
        Track failed downloads in a list.
        """
        try:
            file_size = s3.head_object(Bucket=bucket_name, Key=s3_key)['ContentLength']
            formatted_filename = self.format_filename(os.path.basename(local_path))
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=formatted_filename, ncols=80, bar_format='{desc:.40}|{bar:20}| {percentage:3.0f}% {n_fmt}/{total_fmt}B') as pbar:
                def progress_callback(bytes_transferred):
                    pbar.update(bytes_transferred)

                s3.download_file(bucket_name, s3_key, local_path, Callback=progress_callback)
        except Exception as e:
            print(f"Failed to download {s3_key}. Error: {e}")
            failed_downloads.append(s3_key)

    def download_item(self, item, download_root):
        """Download the Product asset of a STAC item, unless already downloaded."""

        token = self.get_keycloak()
        http = urllib3.PoolManager()

        for asset_name, asset in item.assets.items():

            if asset_name != "Product":
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

    def download_s3_asset(self, item, asset_title, download_root):
            """Download the Product asset of a STAC item, unless already downloaded."""

    
            for asset_name, asset in item.assets.items():
    
                if asset_name != asset_title:
                    continue
    
                download_url = asset.href
                filename = os.path.basename(download_url)

                path = download_url.replace("s3:", "").strip()

                bucket_name, base_s3_path = path.lstrip("/").split("/", 1)
    
                # Create:
                # download_root / collection_id / item_id
                download_path = os.path.join(
                            download_root,
                            item.id,
                        )
                
                os.makedirs(download_path, exist_ok=True)

                output_path = os.path.join(download_path, filename)

                if Path(output_path).exists():
                    print(f"Already downloaded: {output_path}")
                    return str(output_path)

                config = {
                    "auth_server_url": "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                    "odata_base_url": "https://catalogue.dataspace.copernicus.eu/odata/v1/Products",
                    "s3_endpoint_url": "https://eodata.dataspace.copernicus.eu",
                }
        
                access_token = self.get_access_token(config)
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json"
                }
    
                s3_credentials = self.get_temporary_s3_credentials(headers)
                time.sleep(5)
    
                s3_resource = boto3.resource('s3',
                                        endpoint_url=config["s3_endpoint_url"],
                                        aws_access_key_id=s3_credentials["access_id"],
                                        aws_secret_access_key=s3_credentials["secret"])
    
                failed_downloads = []
                self.traverse_and_download_s3(s3_resource, bucket_name, base_s3_path, download_path, failed_downloads)

                if not failed_downloads:
                    print("Product download complete.")
                else:
                    print("Product download incomplete:")
                    for failed_file in failed_downloads:
                        print(f"- {failed_file}")

                # Step 7: Delete the temporary S3 credentials
                delete_response = requests.delete(f"https://s3-keys-manager.cloudferro.com/api/user/credentials/access_id/{s3_credentials['access_id']}", headers=headers)
                if delete_response.status_code == 204:
                    print("Temporary S3 credentials deleted successfully.")
                else:
                    print(f"Failed to delete temporary S3 credentials. Status code: {delete_response.status_code}")
                        
                return str(output_path)
    
            

