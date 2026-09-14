import rasterio
import numpy as np
import scipy.ndimage as ndi
from osgeo import gdal
from numba import njit, stencil
from shapely.geometry import shape

def intersection_percentage(aoi_geojson, multipolygon_geojson):
    # Convert both geometries to shapely
    aoi_geom = shape(aoi_geojson)
    multipoly = shape(multipolygon_geojson)

    # Compute intersection
    intersection = aoi_geom.intersection(multipoly)

    if intersection.is_empty:
        return 0.0

    intersection_area = intersection.area
    aoi_area = aoi_geom.area

    return (intersection_area / aoi_area) * 100

def mask_extractor(mask, highresfile):

    ds10 = gdal.Open(highresfile)
    ds20 = gdal.Open(mask)

    output = mask.replace('_SCL_20m.jp2', '_SCL_10m.tiff')

    driver = gdal.GetDriverByName('GTiff')
    # RasterXSize - columns
    # RasterYSize - rows
    outdata = driver.Create(
        output,
        ds10.RasterXSize,
        ds10.RasterYSize,
        1,
        gdal.GDT_UInt16,
        options=[
            "COMPRESS=DEFLATE",
            "TILED=YES",
            "PREDICTOR=2",
        ],
    )
    outdata.SetGeoTransform(ds10.GetGeoTransform())
    outdata.SetProjection(ds10.GetProjection())

    data20 = ds20.ReadAsArray(0, 0, ds20.RasterXSize, ds20.RasterYSize, buf_xsize = ds10.RasterXSize, buf_ysize = ds10.RasterYSize)

    outdata.WriteArray(data20, xoff = 0, yoff = 0)
    outdata.FlushCache()

    outdata = None
    data20 = None
    ds20 = None
    ds10 = None

    return output


def binomialSmoother(data):
    def filterFunction(footprint):
        weight = [1, 2, 1, 2, 4, 2, 1, 2, 1]
        # Don't smooth land and invalid pixels
        if np.isnan(footprint[4]):
            return footprint[4]

        footprintSum = 0
        weightSum = 0
        for i in range(len(weight)):
            # Don't use land and invalid pixels in smoothing of other pixels
            if not np.isnan(footprint[i]):
                footprintSum = footprintSum + weight[i] * footprint[i]
                weightSum = weightSum + weight[i]
        try:
            ans = footprintSum/weightSum
        except ZeroDivisionError:
            ans = footprint[4]
        return ans

    smoothedData = ndi.filters.generic_filter(data, filterFunction, 3)

    return smoothedData

@njit
def removeEdgeNaNs(a, i, j):
    values = np.array([a[i-1, j], a[i+1, j], a[i, j-1], a[i, j+1]])
    values = values[~np.isnan(values)]  # Remove NaN values manually
    if values.size == 0:
        return np.nan  # Return NaN if all elements are NaN
    return values.mean()  # Compute mean of non-NaN values
