from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Scene:
    id: str
    item: str
    data_href: str
    acquisition_datetime: datetime
    cloud_cover: float
    crs: str
    footprint: object
    xml_path: Path | None = None
    data_path: Path | None = None
    xml_href: str | None = None
    overall_quality: str | None = None
