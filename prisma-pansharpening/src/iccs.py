import os
import time
import sys
import shutil
import requests
from pathlib import Path
from pystac_client import Client

import httpx
from tqdm import tqdm
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env.example")

ICCS_STAC_URL = "https://platform-eo.iccs.gr/stac/" # STAC_ROOT

KEYCLOAK_TOKEN_URL   = "https://auth-eo.iccs.gr/realms/eo-platform/protocol/openid-connect/token"
CLIENT_ID            = "iccs-eo-public"
TOKEN_REFRESH_BUFFER = 60

class KeycloakAuth:
    """Manages Keycloak token acquisition and automatic refresh."""

    def __init__(self, username: str, password: str):
        self.username      = username
        self.password      = password
        self.token         = None
        self.expires_at    = 0
        self.refresh_token = None

    def _fetch_token(self) -> None:
        print("Authenticating with Keycloak...")
        with httpx.Client(timeout=30) as c:
            r = c.post(
                KEYCLOAK_TOKEN_URL,
                data={
                    "grant_type": "password",
                    "client_id":  CLIENT_ID,
                    "username":   self.username,
                    "password":   self.password,
                },
            )
            if r.status_code != 200:
                print(f"Authentication failed: {r.text}")
                sys.exit(1)
            data = r.json()
            self.token         = data["access_token"]
            self.refresh_token = data.get("refresh_token")
            self.expires_at    = time.time() + data.get("expires_in", 300) - TOKEN_REFRESH_BUFFER
            print("Token obtained successfully")

    def _refresh(self) -> None:
        if not self.refresh_token:
            self._fetch_token()
            return
        print("Refreshing token...")
        with httpx.Client(timeout=30) as c:
            r = c.post(
                KEYCLOAK_TOKEN_URL,
                data={
                    "grant_type":    "refresh_token",
                    "client_id":     CLIENT_ID,
                    "refresh_token": self.refresh_token,
                },
            )
            if r.status_code != 200:
                log.warning("Token refresh failed, re-authenticating...")
                self._fetch_token()
                return
            data = r.json()
            self.token         = data["access_token"]
            self.refresh_token = data.get("refresh_token")
            self.expires_at    = time.time() + data.get("expires_in", 300) - TOKEN_REFRESH_BUFFER
            print("Token refreshed successfully")

    def get_token(self) -> str:
        """Returns a valid token, refreshing it if necessary."""
        if self.token is None or time.time() >= self.expires_at:
            if self.token is None:
                self._fetch_token()
            else:
                self._refresh()
        return self.token

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.get_token()}"}

class ICCSSTAC:

    """ 
    Search and download Sentinel-2 scenes. 

    This class intentionally separates: 
    1. Search parameters 
    2. Remote API/search results 
    3. Downloading scene data 
    """

    def __init__(self):
        self.username = os.environ["ICCS_STAC_USRNAME"]
        self.password = os.environ["ICCS_STAC_PASSWRD"]
        self.auth = KeycloakAuth(self.username, self.password)

    def search(self, collection, aoi_geojson, ids=None, datetime=None, query=None):

        catalog = Client.open(ICCS_STAC_URL, headers= self.auth.headers)
        

        search = catalog.search(
            collections = [collection],
            intersects =  aoi_geojson,
            max_items=10,
            datetime = datetime,
            ids=ids,
            query= query if query is not None else None, 
            )

        items = search.item_collection()

        return items

    
    def download_item(self, item, download_root) -> str:
        """Download an item asset, unless already downloaded."""

        url = item.data_href

        needs_auth = ICCS_STAC_URL in url
        headers = self.auth.headers if needs_auth else {}

        # Create:
        # download_root / scene_id
        download_path = os.path.join(
            download_root,
            item.scene_id,
        )

        os.makedirs(download_path, exist_ok=True)

        filename = f"{item.scene_id}.tiff"

        output_path = os.path.join(
            download_path,
            filename,
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

        print(f"Downloading {item.scene_id}")
        print(f"Destination: {output_path}")

        try:
            with httpx.stream(
                "GET",
                url,
                headers=headers,
                timeout=600,
                follow_redirects=True,
            ) as response:

                if response.status_code != 200:
                    raise RuntimeError(
                        f"Download failed for {item.scene_id}: "
                        f"HTTP {response.status_code}"
                    )

                total = int(response.headers.get("content-length", 0))

                with open(temp_path, "wb") as out_file, tqdm(
                    total=total if total > 0 else None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    leave=False,
                ) as bar:

                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        out_file.write(chunk)
                        bar.update(len(chunk))

            # Only make the file official after successful download
            os.replace(temp_path, output_path)

            print(f"Downloaded successfully: {output_path}")

            return str(output_path)

        except Exception:
            # Remove incomplete download
            if temp_path.exists():
                temp_path.unlink()

            raise