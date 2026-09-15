import os
import time
from boto3 import Session

from botocore.exceptions import ClientError, BotoCoreError
from boto3.s3.transfer import TransferConfig
from rich.progress import (
    Progress,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)

import logging

logger = logging.getLogger(__name__)

def create_s3_client(s3_client_id, s3_client_secret):
    """Create an S3 client and verify that the credentials work."""

    logger.info("ICCS S3 Username: %s", s3_client_id)
    logger.info("ICCS Password loaded: %s", bool(s3_client_secret))

    session = Session(
        aws_access_key_id=s3_client_id,
        aws_secret_access_key=s3_client_secret,
    )

    s3_client = session.client(
        "s3",
        endpoint_url="https://platform-eo-storage.iccs.gr",
    )

    try:
        response = s3_client.list_buckets()

        logger.info("S3 authentication successful")
        logger.info(
            "Available buckets: %s",
            [bucket["Name"] for bucket in response.get("Buckets", [])],
        )

        return s3_client

    except ClientError as e:
        error = e.response.get("Error", {})
        logger.error(
            "S3 authentication/API request failed: %s - %s",
            error.get("Code"),
            error.get("Message"),
        )
        raise

    except BotoCoreError as e:
        logger.error("Could not communicate with S3: %s", e)
        raise

def upload_if_not_exists_safe(
    client,
    filename,
    bucket,
    key,
    max_retries=3
):
    filesize = os.path.getsize(filename)

    for attempt in range(max_retries):
        try:
            with Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
            ) as progress:

                task = progress.add_task(
                    f"Uploading {os.path.basename(filename)}",
                    total=filesize
                )

                def progress_callback(bytes_transferred):
                    progress.update(task, advance=bytes_transferred)

                client.upload_file(
                    filename,
                    bucket,
                    key,
                    ExtraArgs={
                        "Metadata": {
                            "uploaded_by": "safe_uploader"
                        },
                        "ContentType": "application/octet-stream"
                    },
                    Config=TransferConfig(
                        multipart_threshold=8 * 1024 * 1024,
                        multipart_chunksize=8 * 1024 * 1024,
                        max_concurrency=4,
                        use_threads=True
                    ),
                    Callback=progress_callback
                )

            print(f"✅ Uploaded: s3://{bucket}/{key}")
            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]

            if error_code in ["AccessDenied", "403"]:
                raise PermissionError(
                    f"❌ Access denied to s3://{bucket}/{key}"
                )

            if error_code in [
                "SlowDown",
                "ServiceUnavailable",
                "InternalError"
            ]:
                wait = 2 ** attempt
                print(f"⚠️ Retrying in {wait}s...")
                time.sleep(wait)
                continue

            raise

    raise RuntimeError(
        f"Failed to upload after {max_retries} attempts."
    )