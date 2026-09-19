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

@dataclass
class PreprocessedPair:
    """Files and metadata produced for one EnMAP/Sentinel-2 pair."""

    enmap_path: Path
    sentinel2_b04_path: Path
    sentinel2_pan_path: Path
    wavelength: list[str]
    fwhm: list[float]
