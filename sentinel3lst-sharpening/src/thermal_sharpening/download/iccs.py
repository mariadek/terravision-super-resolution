import logging
import pystac_client

logger = logging.getLogger(__name__)


def find_ICCS_S3TS(aoi, datetime_range):
    """
    Find Sentinel-3 LST thermal-sharpened (TS) products
    intersecting an AOI within a given datetime range.

    Returns
    -------
    list[pystac.Item] or None
        Matching STAC items if products already exist.
        None if no products are found.
    """

    logger.info("Querying ICCS STAC for thermal sharpened products")
    logger.info("AOI type: %s", aoi.get("type"))
    logger.info("Date range: %s", datetime_range)

    try:
        logger.info("Connecting to ICCS STAC catalog...")

        iccs_client = pystac_client.Client.open(
            "https://platform-eo.iccs.gr/stac/",
            timeout=180.0
        )

        logger.info("Connected to STAC catalog")

        logger.info(
            "Searching for TS products in AOI and date range..."
        )

        search = iccs_client.search(
            collections=["sentinel-3-sl-2-lst-ntc-ts"],
            intersects=aoi,
            datetime=datetime_range,
            max_items=1000
        )

        items = list(search.items())

        logger.info(
            "STAC query succeeded. Found %d items.",
            len(items)
        )

        # Products already exist -> stop this workflow
        if items:
            logger.info(
                "The requested Sentinel-3 image has already been "
                "thermally sharpened. It is available on ICCS STAC."
            )

            return items

        # No products -> caller should proceed with sharpening
        logger.warning(
            "No thermal sharpened products found. "
            "Proceeding to thermal sharpening..."
        )

        return None

    except Exception:
        logger.exception("STAC query failed")
        return None