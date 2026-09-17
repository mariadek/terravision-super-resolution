import os
import time
import logging
from pathlib import Path, PurePosixPath

import boto3
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig
from rich.progress import (
    Progress,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)

logger = logging.getLogger(__name__)
logging.getLogger("boto3").setLevel(logging.WARNING)

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


def put_output_to_s3(
    output_path,
    bucket,
    collection_dir,
    s3_client
) -> bool:
    """
    Upload an output file to the specified S3 collection.
    """

    output_path = Path(output_path)

    key = (
        f"{collection_dir.strip('/')}/"
        f"{output_path.name}"
    )


    return upload_if_not_exists_safe(
        client=s3_client,
        filename=output_path,
        bucket=bucket,
        key=key,
    )