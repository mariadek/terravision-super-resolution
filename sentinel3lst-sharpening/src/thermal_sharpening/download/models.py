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

    intersection_percentage: float = 0.0
    time_difference_minutes: float = 0.0
