import os
import time
import json
import httpx
import logging
import requests
import pystac_client
from datetime import datetime, timezone
from pystac import (
    Item, Asset, Collection, Summaries,
    Extent, SpatialExtent, TemporalExtent,
    Provider, MediaType,
)

logger = logging.getLogger(__name__)

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


def collection_exists(stac_url: str, collection_id: str, auth: dict) -> bool:

    catalog = pystac_client.Client.open(stac_url, 
        request_modifier=lambda request: (request.headers.update(auth.headers) or request),
    )

    return any(
        collection.id == collection_id
        for collection in catalog.get_collections()
    )

def create_enmap_l2a_pansharpened_10m_collection():

    S3_ENDPOINT = "https://platform-eo-storage.iccs.gr"

    STAC_ROOT = os.getenv("STAC_ROOT", "https://platform-eo.iccs.gr/stac")

    STAC_EXTENSIONS = [
        "https://stac-extensions.github.io/processing/v1.1.0/schema.json",
        "https://stac-extensions.github.io/projection/v1.1.0/schema.json",
        "https://stac-extensions.github.io/product/v0.1.0/schema.json",
        "https://stac-extensions.github.io/alternate-assets/v1.2.0/schema.json",
        "https://stac-extensions.github.io/sat/v1.1.0/schema.json",
        ]


    collection = Collection(
        id="enmap-l2a-pansharpened-10m",
        description=(
            "The enmap-l2a-pansharpened-10m collection contains 10 m spatial resolution EnMAP Level 2A "
            "hyperspectral images derived by pansharpening them with a pseudopanchromatic image produced " 
            "by the average of the 4 Sentinel-2 bands with 10m spatial resolution. This collection holds "
            "nomical quality EnMAP L2A HSI products were the water vapour bands were removed." 
            "This collection is hosted by ICCS for the TERRAVISION project."
        ),
        extent=Extent(
            spatial=SpatialExtent([[-180, -90, 180, 90]]),
            temporal=TemporalExtent([[datetime(2022, 4, 27, tzinfo=timezone.utc), None]]),
        ),
        title="EnMAP L2A HSI Pansharpened Products 10m",
        stac_extensions=STAC_EXTENSIONS,
        keywords=[
            "EnMAP", "Hyperspectral", "HSI", "VNIR", "SWIR", "10m", "L2A",
        ],
        providers=[
            Provider(name="DLR/EOC Geoservice",  roles=["producer", "processor", "licensor"], url="https://geoservice.dlr.de"),
            Provider(name="ICCS", roles=["host", "processor"],                 url="https://platform-eo.iccs.gr"),
        ],
        license="CC-BY-4.0",
    )

    # ── Summaries ─────────────────────────────────────────────────────────────
    collection.summaries.add("platform",            ["ENMAP"])
    collection.summaries.add("instrument",          ["HSI"])
    collection.summaries.add("orbit type",          ["LEO"])
    collection.summaries.add("sensor type",         ["OPTICAL"])
    collection.summaries.add("gsd",                 [10])
    collection.summaries.add("processing:level",    ["L2A"])
    collection.summaries.add("processing:facility", ["DLR", "ICCS"])
    collection.summaries.add("hsi:wavelength_min",  [420])
    collection.summaries.add("hsi:wavelength_max",  [2450])


    collection.extra_fields["item_assets"] = {
        # ── Ancillary ─────────────────────────────────────────────────────────
        "thumbnail": {
            "type":  MediaType.PNG,
            "title": "RGB Quicklook",
            "roles": ["thumbnail"],
        },
    }

    collection.add_asset(
        "thumbnail",
        Asset(
            href=f"{S3_ENDPOINT}/public/stac-thumbnail/enmap_pansharpened.png",
            media_type=MediaType.PNG,
            title="EnMAP L2A HSI Pansharpened Products 10m",
            roles=["thumbnail"],
        ),
    )
    
    return collection


def search_ICCS_collection(aoi, datetime_range, collection_id, auth):
    """
    Find Sentinel-3 LST thermal-sharpened (TS) products
    intersecting an AOI within a given datetime range.

    Returns
    -------
    list[pystac.Item] or None
        Matching STAC items if products already exist.
        None if no products are found.
    """

    logger.info(f"Querying ICCS STAC for {collection_id} products")
    logger.info("AOI type: %s", aoi.get("type"))
    logger.info("Date range: %s", datetime_range)

    try:
        logger.info("Connecting to ICCS STAC catalog...")

        iccs_client = pystac_client.Client.open(
                "https://platform-eo.iccs.gr/stac/",
                request_modifier=lambda request: (request.headers.update(auth.headers) or request),
                timeout=180.0
            )

        logger.info("Connected to STAC catalog")

        logger.info(
            f"Searching for {collection_id} products in AOI and date range..."
        )

        search = iccs_client.search(
            collections=[collection_id],
            intersects=aoi,
            datetime=datetime_range,
            max_items=100
        )

        items = list(search.items())

        logger.info(
            "STAC query succeeded. Found %d items.",
            len(items)
        )

        # Products already exist -> stop this workflow
        if items:
            logger.info(
                "The requested image has already been "
                "processed. It is available on ICCS STAC."
            )

            return items

        # No products -> caller should proceed with sharpening
        logger.warning(
            f"No {collection_id} found. "
            "Proceeding to pansharpening..."
        )

        return None

    except Exception:
        logger.exception("STAC query failed")
        return None