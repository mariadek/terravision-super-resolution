import logging
import os
import time
from pathlib import Path
from typing import Any

import boto3
import requests
import urllib3
from pystac_client import Client
from tqdm import tqdm


logger = logging.getLogger(__name__)

CDSE_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
S2_COLLECTION = "sentinel-2-l2a"
AUTH_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
S3_CREDENTIALS_URL = "https://s3-keys-manager.cloudferro.com/api/user/credentials"
S3_ENDPOINT_URL = "https://eodata.dataspace.copernicus.eu"


class Sentinel2Downloader:
    """Search for Sentinel-2 scenes and download HTTP or S3 assets."""

    def __init__(self, username: str | None = None, password: str | None = None) -> None:
        self.username = username or os.getenv("CDSE_USERNAME")
        self.password = password or os.getenv("CDSE_PASSWORD")

        if not self.username or not self.password:
            raise RuntimeError(
                "Missing CDSE credentials. Set CDSE_USERNAME and CDSE_PASSWORD."
            )

        self.catalog = Client.open(CDSE_STAC_URL)
        self.http = urllib3.PoolManager()

    def search(
        self,
        aoi_geojson: dict[str, Any],
        datetime: str | None = None,
        query: dict[str, Any] | None = None,
        max_items: int = 5,
    ):
        """Search the Sentinel-2 L2A STAC collection."""
        search = self.catalog.search(
            collections=[S2_COLLECTION],
            intersects=aoi_geojson,
            datetime=datetime,
            query=query,
            max_items=max_items,
        )
        return search.item_collection()

    def get_access_token(self) -> str:
        """Return a CDSE OAuth access token."""
        response = requests.post(
            AUTH_URL,
            data={
                "client_id": "cdse-public",
                "grant_type": "password",
                "username": self.username,
                "password": self.password,
            },
            timeout=30,
        )

        if not response.ok:
            raise RuntimeError(
                f"CDSE authentication failed ({response.status_code}): {response.text}"
            )

        token = response.json().get("access_token")
        if not token:
            raise RuntimeError(
                "CDSE authentication response did not contain an access token."
            )

        return token

    def get_temporary_s3_credentials(self, access_token: str) -> dict[str, str]:
        """Create temporary S3 credentials for the CDSE EO data endpoint."""
        response = requests.post(
            S3_CREDENTIALS_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
            timeout=30,
        )

        if not response.ok:
            raise RuntimeError(
                f"Failed to create temporary S3 credentials "
                f"({response.status_code}): {response.text}"
            )

        return response.json()

    @staticmethod
    def delete_temporary_s3_credentials(
        access_token: str,
        access_id: str,
    ) -> None:
        """Delete temporary S3 credentials."""
        response = requests.delete(
            f"{S3_CREDENTIALS_URL}/access_id/{access_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        )

        if response.status_code != 204:
            raise RuntimeError(
                f"Failed to delete temporary S3 credentials "
                f"({response.status_code})."
            )

    @staticmethod
    def _format_filename(filename: str, length: int = 40) -> str:
        if len(filename) > length:
            return filename[: length - 3] + "..."
        return filename.ljust(length)

    def _download_s3_file(
        self,
        s3_client,
        bucket_name: str,
        s3_key: str,
        local_path: Path,
    ) -> None:
        """Download one S3 object with a progress bar."""
        local_path.parent.mkdir(parents=True, exist_ok=True)

        file_size = s3_client.head_object(
            Bucket=bucket_name,
            Key=s3_key,
        )["ContentLength"]

        description = self._format_filename(local_path.name)

        logger.info(
            "Downloading Sentinel-2 S3 object: %s -> %s",
            s3_key,
            local_path,
        )

        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=description,
            ncols=80,
            bar_format="{desc:.40}|{bar:20}| {percentage:3.0f}% {n_fmt}/{total_fmt}B",
        ) as progress:
            s3_client.download_file(
                bucket_name,
                s3_key,
                str(local_path),
                Callback=progress.update,
            )

        logger.info("Downloaded Sentinel-2 S3 object: %s", local_path)

    def _download_s3_prefix(
        self,
        s3_resource,
        bucket_name: str,
        prefix: str,
        local_root: Path,
    ) -> list[str]:
        """Download every object under an S3 prefix while preserving its tree."""
        failures: list[str] = []
        bucket = s3_resource.Bucket(bucket_name)

        logger.info(
            "Downloading S3 prefix s3://%s/%s to %s",
            bucket_name,
            prefix,
            local_root,
        )

        for obj in bucket.objects.filter(Prefix=prefix):
            if obj.key.endswith("/"):
                continue

            relative = Path(obj.key).relative_to(prefix.rstrip("/"))
            local_path = local_root / relative

            try:
                self._download_s3_file(
                    s3_resource.meta.client,
                    bucket_name,
                    obj.key,
                    local_path,
                )
            except Exception as exc:
                logger.error(
                    "Failed to download S3 object %s: %s",
                    obj.key,
                    exc,
                )
                failures.append(obj.key)

        if failures:
            logger.warning(
                "S3 prefix download completed with %d failure(s).",
                len(failures),
            )
        else:
            logger.info(
                "S3 prefix download completed successfully: %s",
                local_root,
            )

        return failures

    def download_item(
        self,
        item,
        download_root: str | Path,
    ) -> str:
        """Download an item's ``Product`` HTTP asset as a ZIP file."""
        asset = item.assets.get("Product")

        if asset is None:
            raise ValueError(
                f"No 'Product' asset found for STAC item {item.id}"
            )

        destination_dir = Path(download_root) / item.id
        destination_dir.mkdir(parents=True, exist_ok=True)

        output_path = destination_dir / f"{item.id}.zip"
        temp_path = output_path.with_suffix(output_path.suffix + ".part")

        if output_path.exists():
            logger.info(
                "Sentinel-2 product already downloaded: %s",
                output_path,
            )
            return str(output_path)

        temp_path.unlink(missing_ok=True)

        logger.info(
            "Downloading Sentinel-2 product %s to %s",
            item.id,
            output_path,
        )

        token = self.get_access_token()

        try:
            with self.http.request(
                "GET",
                asset.href,
                headers={"Authorization": f"Bearer {token}"},
                preload_content=False,
                redirect=True,
                timeout=urllib3.Timeout(connect=30.0, read=300.0),
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Download failed for {item.id}: HTTP {response.status}"
                    )

                with temp_path.open("wb") as output_file:
                    while True:
                        chunk = response.read(1024 * 1024)

                        if not chunk:
                            break

                        output_file.write(chunk)

            temp_path.replace(output_path)

        except Exception:
            temp_path.unlink(missing_ok=True)

            logger.exception(
                "Failed to download Sentinel-2 product %s",
                item.id,
            )
            raise

        logger.info(
            "Sentinel-2 product downloaded successfully: %s",
            output_path,
        )

        return str(destination_dir)

    def download_s3_assets(
        self,
        item,
        asset_titles: tuple[str, ...],
        download_root: str | Path,
    ) -> dict[str, str]:
        """Download multiple S3 assets from the same STAC item."""

        destination_dir = Path(download_root) / item.id
        destination_dir.mkdir(parents=True, exist_ok=True)

        assets_to_download = {}

        for asset_title in asset_titles:
            asset = item.assets.get(asset_title)

            if asset is None:
                raise ValueError(
                    f"No {asset_title!r} asset found for STAC item {item.id}"
                )

            if not asset.href.startswith("s3://"):
                raise ValueError(
                    f"Asset {asset_title!r} is not an S3 URL: {asset.href}"
                )

            bucket_name, object_key = asset.href[5:].split("/", 1)
            filename = Path(object_key).name
            destination_path = destination_dir / filename

            assets_to_download[asset_title] = {
                "bucket_name": bucket_name,
                "object_key": object_key,
                "destination_path": destination_path,
            }

        access_token = self.get_access_token()
        credentials = self.get_temporary_s3_credentials(access_token)
        access_id = credentials["access_id"]

        try:
            logger.info(
                "Waiting for temporary S3 credentials to become active"
            )
            time.sleep(5)

            s3_resource = boto3.resource(
                "s3",
                endpoint_url=S3_ENDPOINT_URL,
                aws_access_key_id=access_id,
                aws_secret_access_key=credentials["secret"],
                region_name="default",
            )

            downloaded_paths = {}

            for asset_title, asset_info in assets_to_download.items():
                destination_path = asset_info["destination_path"]

                if destination_path.is_file():
                    logger.info(
                        "Sentinel-2 asset already exists: %s",
                        destination_path,
                    )
                    downloaded_paths[asset_title] = str(destination_path)
                    continue

                logger.info(
                    "Downloading Sentinel-2 asset %s for scene %s",
                    asset_title,
                    item.id,
                )

                try:
                    self._download_s3_file(
                        s3_resource.meta.client,
                        asset_info["bucket_name"],
                        asset_info["object_key"],
                        destination_path,
                    )

                except Exception as exc:
                    logger.exception(
                        "Failed to download Sentinel-2 asset %s for scene %s",
                        asset_title,
                        item.id,
                    )

                    raise RuntimeError(
                        f"Failed to download S3 asset {asset_title!r} "
                        f"for item {item.id}: {exc}"
                    ) from exc

                logger.info(
                    "Sentinel-2 asset download complete: %s",
                    destination_path,
                )

                downloaded_paths[asset_title] = str(destination_path)

            return downloaded_paths

        finally:
            self.delete_temporary_s3_credentials(
                access_token,
                access_id,
            )