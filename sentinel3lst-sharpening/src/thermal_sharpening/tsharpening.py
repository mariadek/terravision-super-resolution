import math
import os
import re
import tempfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from osgeo import gdal
from sklearn import ensemble, linear_model, tree
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

import thermal_sharpening.utils.utils as utils # Required by residual smoothing / edge filling.


BAND_DESCRIPTIONS = {
    "B1 (443 nm)": "B1",
    "B2 (490 nm)": "B2",
    "B3 (560 nm)": "B3",
    "B4 (665 nm)": "B4",
    "B5 (705 nm)": "B5",
    "B6 (740 nm)": "B6",
    "B7 (783 nm)": "B7",
    "B8 (842 nm)": "B8",
    "B8A (865 nm)": "B8A",
    "B9 (945 nm)": "B9",
    "B11 (1610 nm)": "B11",
    "B12 (2190 nm)": "B12",
}

FEATURE_ORDER = tuple(BAND_DESCRIPTIONS.values())
NODATA_LR = -32768.0
NODATA_OUT = -9999.0

class DecisionTreeSharpener:
    """Decision-tree based sharpening/disaggregation of low-resolution imagery.

    This version keeps the original public API while fixing several correctness,
    performance, and maintainability issues:

    * no mutable default arguments
    * validates GDAL files and expected Sentinel-2 bands
    * vectorizes quality masking instead of looping pixel-by-pixel
    * accumulates records in a list instead of repeatedly concatenating DataFrames
    * implements the documented automatic CV threshold (80th percentile)
    * gives more homogeneous samples larger weights
    * supports ``movingWindowSize <= 0`` as global-only regression
    * masks non-finite predictor rows before calling scikit-learn
    * avoids undefined local/global output arrays
    """

    def __init__(
        self,
        highResFile,
        lowResFile,
        highResQualityFile,
        lowResQualityFile,
        lowResGoodQualityFlags=None,
        highResGoodQualityFlags=None,
        cvHomogeneityThreshold=0,
        movingWindowSize=0,
        disaggregatingTemperature=False,
        perLeafLinearRegression=True,
        linearRegressionExtrapolationRatio=0.25,
        regressorOpt=None,
        baggingRegressorOpt=None,
        scaleFactor=100,
        predictionWorkers=None,
    ):
        self.highResFile = highResFile
        self.lowResFile = lowResFile
        self.highResQualityFile = highResQualityFile
        self.lowResQualityFile = lowResQualityFile

        self.lowResGoodQualityFlags = list(lowResGoodQualityFlags or [])
        self.highResGoodQualityFlags = list(highResGoodQualityFlags or [])

        self.cvHomogeneityThreshold = float(cvHomogeneityThreshold)
        self.autoAdjustCvThreshold = self.cvHomogeneityThreshold <= 0
        self.percentileThreshold = 80.0

        self.movingWindowSize = float(movingWindowSize)
        self.movingWindowExtension = self.movingWindowSize * 0.25
        self.windowExtents = []

        self.disaggregatingTemperature = bool(disaggregatingTemperature)
        self.perLeafLinearRegression = bool(perLeafLinearRegression)
        self.linearRegressionExtrapolationRatio = float(
            linearRegressionExtrapolationRatio
        )

        self.regressorOpt = dict(regressorOpt or {})
        self.baggingRegressorOpt = dict(baggingRegressorOpt or {})
        self.scaleFactor = int(scaleFactor)

        if self.scaleFactor <= 0:
            raise ValueError("scaleFactor must be a positive integer.")

        # Parallel spatial prediction. A conservative automatic default avoids
        # excessive memory use because every worker holds a 12-band window.
        cpu_count = os.cpu_count() or 1
        if predictionWorkers is None:
            self.predictionWorkers = min(4, cpu_count)
        else:
            self.predictionWorkers = max(1, int(predictionWorkers))

        self.df = pd.DataFrame()
        self.reg = []

    # ------------------------------------------------------------------
    # GDAL / data helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _open_raster(path, label):
        ds = gdal.Open(str(path))
        if ds is None:
            raise FileNotFoundError(f"Could not open {label}: {path}")
        return ds

    @staticmethod
    def _output_stem(path):
        return Path(path).stem

    def _get_band_map(self, dataset):
        bands = {}
        for band_index in range(1, dataset.RasterCount + 1):
            band = dataset.GetRasterBand(band_index)
            key = BAND_DESCRIPTIONS.get(band.GetDescription())
            if key is not None:
                bands[key] = band

        missing = [name for name in FEATURE_ORDER if name not in bands]
        if missing:
            raise ValueError(
                "High-resolution image is missing required band descriptions: "
                + ", ".join(missing)
            )
        return bands

    @staticmethod
    def _read_band_window(bands, xoff, yoff, xsize, ysize):
        arrays = [
            bands[name].ReadAsArray(xoff, yoff, xsize, ysize).astype(np.float32)
            for name in FEATURE_ORDER
        ]
        return np.stack(arrays, axis=-1)

    def _quality_mask(self, quality_data):
        if not self.highResGoodQualityFlags:
            return np.ones(quality_data.shape, dtype=bool)
        return np.isin(quality_data, self.highResGoodQualityFlags)

    def _low_quality_mask(self, quality_data):
        if not self.lowResGoodQualityFlags:
            return np.ones(quality_data.shape, dtype=bool)
        return np.isin(quality_data, self.lowResGoodQualityFlags)

    @staticmethod
    def _mean_cv(feature_cube, valid_mask):
        """Return mean coefficient of variation across feature bands."""
        if not np.any(valid_mask):
            return np.nan, None

        masked = np.where(valid_mask[..., None], feature_cube, np.nan)
        means = np.nanmean(masked, axis=(0, 1))
        stds = np.nanstd(masked, axis=(0, 1))

        # Avoid divide-by-zero and meaningless CV values.
        with np.errstate(divide="ignore", invalid="ignore"):
            cvs = np.divide(
                stds,
                np.abs(means),
                out=np.full_like(stds, np.nan, dtype=float),
                where=np.abs(means) > np.finfo(float).eps,
            )

        if not np.any(np.isfinite(cvs)):
            return np.nan, None

        features = np.nanmean(masked, axis=(0, 1))
        return float(np.nanmean(cvs)), features

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def trainSharpener(self):
        scene_hr = self._open_raster(self.highResFile, "high-resolution image")
        mask_hr = self._open_raster(
            self.highResQualityFile, "high-resolution quality image"
        )
        scene_lr = self._open_raster(self.lowResFile, "low-resolution image")
        mask_lr = self._open_raster(
            self.lowResQualityFile, "low-resolution quality image"
        )

        try:
            bands = self._get_band_map(scene_hr)
            size_x = scene_hr.RasterXSize
            size_y = scene_hr.RasterYSize

            low_data = scene_lr.ReadAsArray()
            low_quality = mask_lr.ReadAsArray()

            if low_data.ndim != 2 or low_quality.ndim != 2:
                raise ValueError("Low-resolution data and quality rasters must be 2-D.")
            if low_data.shape != low_quality.shape:
                raise ValueError(
                    "Low-resolution image and quality raster must have the same shape."
                )

            valid_lr = (
                np.isfinite(low_data)
                & (low_data != NODATA_LR)
                & self._low_quality_mask(low_quality)
            )
            candidate_pixels = np.argwhere(valid_lr)

            logger.info(
                "Extracting %d candidate samples...",
                len(candidate_pixels),
            )
            records = []

            for i, j in tqdm(candidate_pixels, desc="Samples"):
                xoff = int(j * self.scaleFactor)
                yoff = int(i * self.scaleFactor)

                xsize = min(self.scaleFactor, size_x - xoff)
                ysize = min(self.scaleFactor, size_y - yoff)
                if xsize <= 0 or ysize <= 0:
                    continue

                quality = mask_hr.ReadAsArray(xoff, yoff, xsize, ysize)
                valid_hr = self._quality_mask(quality)
                if not np.any(valid_hr):
                    continue

                cube = self._read_band_window(
                    bands, xoff, yoff, xsize, ysize
                )
                cv, features = self._mean_cv(cube, valid_hr)

                if features is None or not np.isfinite(cv):
                    continue
                if not np.all(np.isfinite(features)):
                    continue

                records.append(
                    {
                        "temperature": float(low_data[i, j]),
                        "s2-col": xoff,
                        "s2-row": yoff,
                        "cv": cv,
                        "features": features,
                    }
                )

            if not records:
                raise RuntimeError("No valid training samples were extracted.")

            df = pd.DataFrame.from_records(records)

            if self.autoAdjustCvThreshold:
                self.cvHomogeneityThreshold = float(
                    np.nanpercentile(df["cv"].to_numpy(), self.percentileThreshold)
                )
                logger.info(
                    "Automatically selected CV homogeneity threshold: %.6g",
                    self.cvHomogeneityThreshold,
                )

            df = df.loc[df["cv"] <= self.cvHomogeneityThreshold].copy()
            if df.empty:
                raise RuntimeError(
                    "No training samples satisfy the homogeneity threshold "
                    f"{self.cvHomogeneityThreshold}."
                )

            # Create a boolean raster on the LOW-RESOLUTION grid:
            #   1 = homogeneous sample used for training
            #   0 = not used / not homogeneous / invalid
            homogeneous_mask = np.zeros(low_data.shape, dtype=np.uint8)

            sample_rows = (
                df["s2-row"].to_numpy(dtype=np.int64) // self.scaleFactor
            )
            sample_cols = (
                df["s2-col"].to_numpy(dtype=np.int64) // self.scaleFactor
            )

            valid_indices = (
                (sample_rows >= 0)
                & (sample_rows < homogeneous_mask.shape[0])
                & (sample_cols >= 0)
                & (sample_cols < homogeneous_mask.shape[1])
            )
            homogeneous_mask[
                sample_rows[valid_indices],
                sample_cols[valid_indices],
            ] = 1

            low_res_path = Path(self.lowResFile)

            homogeneous_mask_path = (
                low_res_path.parent
                / f"{low_res_path.stem}_homogeneous_samples.tif"
            )
            
            mask_driver = gdal.GetDriverByName("GTiff")
            homogeneous_ds = mask_driver.Create(
                str(homogeneous_mask_path),
                homogeneous_mask.shape[1],
                homogeneous_mask.shape[0],
                1,
                gdal.GDT_Byte,
                options=["COMPRESS=LZW"],
            )
            if homogeneous_ds is None:
                raise RuntimeError(
                    f"Could not create homogeneous-sample mask: "
                    f"{homogeneous_mask_path}"
                )

            homogeneous_ds.SetGeoTransform(scene_lr.GetGeoTransform())
            homogeneous_ds.SetProjection(scene_lr.GetProjection())
            homogeneous_band = homogeneous_ds.GetRasterBand(1)
            homogeneous_band.WriteArray(homogeneous_mask)
            homogeneous_band.SetDescription("Homogeneous training samples")
            homogeneous_band.SetMetadataItem("VALUE_0", "Not homogeneous / not used")
            homogeneous_band.SetMetadataItem("VALUE_1", "Homogeneous sample used for training")
            homogeneous_band.FlushCache()
            homogeneous_ds.FlushCache()
            homogeneous_ds = None

            # Higher weight for more homogeneous samples.
            threshold = max(
                self.cvHomogeneityThreshold, np.finfo(float).eps
            )
            df["weight"] = np.clip(1.0 - df["cv"] / threshold, 0.05, 1.0)
            self.df = df[
                ["temperature", "s2-col", "s2-row", "weight", "features"]
            ].reset_index(drop=True)

            samples_path = (
                Path(self.lowResFile).parent
                / f"{Path(self.lowResFile).stem}_samples.csv"
            )
            self.df.to_csv(samples_path, index=False)

            logger.info("Candidate samples: %d", len(candidate_pixels))
            logger.info("Homogeneous samples used: %d", len(self.df))
            logger.info("Homogeneous sample mask: %s", homogeneous_mask_path)

            windows, extents = self._build_windows(size_x, size_y)
            self.windowExtents = extents
            self.reg = [None] * len(windows)

            logger.info("Training...")
            for idx, window in enumerate(tqdm(windows, desc="Models")):
                local = idx < len(windows) - 1
                y0, y1, x0, x1 = window

                selected = self.df.loc[
                    (self.df["s2-row"] >= y0)
                    & (self.df["s2-row"] < y1)
                    & (self.df["s2-col"] >= x0)
                    & (self.df["s2-col"] < x1)
                ]

                if selected.empty:
                    continue

                X = np.vstack(selected["features"].to_numpy())
                y = selected["temperature"].to_numpy(dtype=float)
                weight = selected["weight"].to_numpy(dtype=float)

                self.reg[idx] = self._doFit(y, X, weight, local)

        finally:
            scene_hr = None
            mask_hr = None
            scene_lr = None
            mask_lr = None

    def _build_windows(self, size_x, size_y):
        # movingWindowSize <= 0 means global-only regression.
        if self.movingWindowSize <= 0:
            return [[0, size_y, 0, size_x]], []

        windows = []
        extents = []
        n_rows = int(math.ceil(size_y / self.movingWindowSize))
        n_cols = int(math.ceil(size_x / self.movingWindowSize))

        for y in range(n_rows):
            for x in range(n_cols):
                windows.append(
                    [
                        int(max(y * self.movingWindowSize - self.movingWindowExtension, 0)),
                        int(min((y + 1) * self.movingWindowSize + self.movingWindowExtension, size_y)),
                        int(max(x * self.movingWindowSize - self.movingWindowExtension, 0)),
                        int(min((x + 1) * self.movingWindowSize + self.movingWindowExtension, size_x)),
                    ]
                )
                extents.append(
                    [
                        int(y * self.movingWindowSize),
                        int(min((y + 1) * self.movingWindowSize, size_y)),
                        int(x * self.movingWindowSize),
                        int(min((x + 1) * self.movingWindowSize, size_x)),
                    ]
                )

        # Final model is always global.
        windows.append([0, size_y, 0, size_x])
        return windows, extents

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def applySharpener(self):
        if not self.reg:
            raise RuntimeError("trainSharpener() must be called before applySharpener().")

        # The main thread owns output GDAL datasets. Worker threads only read
        # inputs and calculate arrays, which avoids unsafe concurrent writes.
        scene_hr = self._open_raster(self.highResFile, "high-resolution image")

        try:
            size_x = scene_hr.RasterXSize
            size_y = scene_hr.RasterYSize

            out_dir = Path(self.lowResFile).parent
            stem = self._output_stem(self.lowResFile)
            driver = gdal.GetDriverByName("GTiff")

            local_path = out_dir / f"Local_{stem}.tiff"
            global_path = out_dir / f"Global_{stem}.tiff"

            out_local = self._create_output(
                driver, local_path, scene_hr, size_x, size_y
            )
            out_global = self._create_output(
                driver, global_path, scene_hr, size_x, size_y
            )

            global_reg = self.reg[-1]
            if global_reg is None:
                raise RuntimeError("Global model could not be trained.")

            # Global-only mode is represented by a single full-image extent.
            extents = self.windowExtents or [[0, size_y, 0, size_x]]
            workers = min(self.predictionWorkers, len(extents))

            logger.info(
                "Predicting %d windows with %d worker%s...",
                len(extents),
                workers,
                "s" if workers != 1 else "",
            )

            if workers == 1:
                iterator = enumerate(extents)
                for idx, extent in tqdm(
                    iterator, total=len(extents), desc="Sharpening"
                ):
                    _, extent, local_pred, global_pred = self._predict_extent(
                        idx, extent, global_reg
                    )
                    y0, y1, x0, x1 = extent
                    out_local.GetRasterBand(1).WriteArray(
                        np.nan_to_num(local_pred, nan=NODATA_OUT), xoff=x0, yoff=y0
                    )
                    out_global.GetRasterBand(1).WriteArray(
                        np.nan_to_num(global_pred, nan=NODATA_OUT), xoff=x0, yoff=y0
                    )
            else:
                # Keep only a small number of windows in flight so memory use
                # stays bounded. Each prediction window contains 12 bands.
                max_in_flight = workers * 2
                next_index = 0
                pending = {}

                with ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="sharpener",
                ) as executor, tqdm(
                    total=len(extents),
                    desc=f"Sharpening ({workers} threads)",
                ) as progress:

                    while (
                        next_index < len(extents)
                        and len(pending) < max_in_flight
                    ):
                        future = executor.submit(
                            self._predict_extent,
                            next_index,
                            extents[next_index],
                            global_reg,
                        )
                        pending[future] = next_index
                        next_index += 1

                    while pending:
                        done, _ = wait(
                            pending,
                            return_when=FIRST_COMPLETED,
                        )

                        for future in done:
                            pending.pop(future)
                            _, extent, local_pred, global_pred = future.result()

                            y0, y1, x0, x1 = extent
                            out_local.GetRasterBand(1).WriteArray(
                                np.nan_to_num(local_pred, nan=NODATA_OUT), xoff=x0, yoff=y0
                            )
                            out_global.GetRasterBand(1).WriteArray(
                                np.nan_to_num(global_pred, nan=NODATA_OUT), xoff=x0, yoff=y0
                            )
                            progress.update(1)

                            if next_index < len(extents):
                                new_future = executor.submit(
                                    self._predict_extent,
                                    next_index,
                                    extents[next_index],
                                    global_reg,
                                )
                                pending[new_future] = next_index
                                next_index += 1

            out_local.FlushCache()
            out_global.FlushCache()
            out_local = None
            out_global = None

            return str(local_path), str(global_path)

        finally:
            scene_hr = None

    def _predict_extent(self, idx, extent, global_reg):
        """Predict one spatial window.

        Every worker opens independent GDAL read handles. This is deliberately
        separated from output writing because sharing writable GDAL datasets
        across threads is unsafe.
        """
        scene_hr = self._open_raster(
            self.highResFile, "high-resolution image"
        )
        mask_hr = self._open_raster(
            self.highResQualityFile, "high-resolution quality image"
        )

        try:
            bands = self._get_band_map(scene_hr)

            y0, y1, x0, x1 = extent
            xsize = x1 - x0
            ysize = y1 - y0

            quality = mask_hr.ReadAsArray(
                x0, y0, xsize, ysize
            )
            good_quality = self._quality_mask(quality)

            cube = self._read_band_window(
                bands, x0, y0, xsize, ysize
            )

            global_pred = self._doPredict(cube, global_reg)
            global_pred = np.where(
                good_quality, global_pred, np.nan
            ).astype(np.float32, copy=False)

            if self.windowExtents:
                local_reg = self.reg[idx]
                if local_reg is not None:
                    local_pred = self._doPredict(cube, local_reg)
                    local_pred = np.where(
                        good_quality, local_pred, np.nan
                    )
                else:
                    local_pred = global_pred.copy()
            else:
                local_pred = global_pred.copy()

            local_pred = local_pred.astype(np.float32, copy=False)

            return idx, extent, local_pred, global_pred

        finally:
            scene_hr = None
            mask_hr = None

    @staticmethod
    def _create_output(driver, path, template, size_x, size_y):
        out = driver.Create(
            str(path),
            size_x,
            size_y,
            1,
            gdal.GDT_Float32,
            options=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=3", "BIGTIFF=IF_SAFER"],
        )
        if out is None:
            raise RuntimeError(f"Could not create output raster: {path}")
        out.SetGeoTransform(template.GetGeoTransform())
        out.SetProjection(template.GetProjection())
        out.GetRasterBand(1).SetNoDataValue(NODATA_OUT)
        return out

    # ------------------------------------------------------------------
    # Local/global combination and residual analysis
    # ------------------------------------------------------------------

    def combination(self, local_result, global_result, block_size=1024):
        """Combine local/global predictions without loading full HR rasters into RAM.

        The previous implementation materialized two high-resolution residual arrays,
        two prediction arrays, and several weight arrays simultaneously. For large
        scenes this can trigger the container OOM killer, which looks like an abrupt
        Docker stop with no Python traceback.

        Residuals are now calculated on the low-resolution grid, warped to temporary
        on-disk GeoTIFFs, and the final weighting is performed block-by-block.
        """
        block_size = max(128, int(block_size))
        logger.info("Calculating low-resolution residuals for combination...")
        local_residual_lr = self._calculateResidualLR(local_result)
        global_residual_lr = self._calculateResidualLR(global_result)

        local_ds = self._open_raster(local_result, "local result")
        global_ds = self._open_raster(global_result, "global result")
        try:
            if (
                local_ds.RasterXSize != global_ds.RasterXSize
                or local_ds.RasterYSize != global_ds.RasterYSize
            ):
                raise ValueError("Local and global result rasters must have the same size.")

            size_x = local_ds.RasterXSize
            size_y = local_ds.RasterYSize
            out_path = (
                Path(self.lowResFile).parent
                / f"Combined_{self._output_stem(self.lowResFile)}.tiff"
            )
            driver = gdal.GetDriverByName("GTiff")
            out = self._create_output(driver, out_path, local_ds, size_x, size_y)

            with tempfile.TemporaryDirectory(prefix="sharpener_residual_") as tmpdir:
                local_residual_path = Path(tmpdir) / "local_residual_hr.tif"
                global_residual_path = Path(tmpdir) / "global_residual_hr.tif"

                self._warpResidualLRToHR(
                    local_residual_lr, local_ds, local_residual_path
                )
                self._warpResidualLRToHR(
                    global_residual_lr, global_ds, global_residual_path
                )

                # LR residual arrays are no longer needed once warped to disk.
                del local_residual_lr, global_residual_lr

                local_res_ds = self._open_raster(
                    local_residual_path, "local residual"
                )
                global_res_ds = self._open_raster(
                    global_residual_path, "global residual"
                )
                try:
                    local_band = local_ds.GetRasterBand(1)
                    global_band = global_ds.GetRasterBand(1)
                    local_res_band = local_res_ds.GetRasterBand(1)
                    global_res_band = global_res_ds.GetRasterBand(1)
                    out_band = out.GetRasterBand(1)

                    total_blocks = math.ceil(size_x / block_size) * math.ceil(
                        size_y / block_size
                    )
                    logger.info(
                        "Combining predictions in %d blocks (block size %d)...",
                        total_blocks,
                        block_size,
                    )

                    with tqdm(total=total_blocks, desc="Combining") as progress:
                        for yoff in range(0, size_y, block_size):
                            ysize = min(block_size, size_y - yoff)
                            for xoff in range(0, size_x, block_size):
                                xsize = min(block_size, size_x - xoff)

                                local_values = local_band.ReadAsArray(
                                    xoff, yoff, xsize, ysize
                                ).astype(np.float32, copy=False)
                                global_values = global_band.ReadAsArray(
                                    xoff, yoff, xsize, ysize
                                ).astype(np.float32, copy=False)
                                local_res = local_res_band.ReadAsArray(
                                    xoff, yoff, xsize, ysize
                                ).astype(np.float32, copy=False)
                                global_res = global_res_band.ReadAsArray(
                                    xoff, yoff, xsize, ysize
                                ).astype(np.float32, copy=False)

                                local_valid = (
                                    np.isfinite(local_values)
                                    & (local_values != NODATA_OUT)
                                )
                                global_valid = (
                                    np.isfinite(global_values)
                                    & (global_values != NODATA_OUT)
                                )

                                # Equivalent to inverse-squared-residual weighting:
                                # (1/L^2) / ((1/L^2) + (1/G^2)) = G^2/(L^2+G^2).
                                # This form avoids enormous inverse values near zero.
                                l2 = np.square(local_res, dtype=np.float32)
                                g2 = np.square(global_res, dtype=np.float32)
                                denom = l2 + g2
                                ww = np.full(denom.shape, 0.5, dtype=np.float32)
                                weight_valid = (
                                    np.isfinite(denom)
                                    & np.isfinite(g2)
                                    & (denom > np.finfo(np.float32).eps)
                                )
                                np.divide(g2, denom, out=ww, where=weight_valid)

                                combined = np.full(
                                    local_values.shape, NODATA_OUT, dtype=np.float32
                                )
                                both = local_valid & global_valid
                                combined[both] = (
                                    local_values[both] * ww[both]
                                    + global_values[both] * (1.0 - ww[both])
                                )
                                # If only one model has a valid value, keep it rather
                                # than turning that pixel into NoData.
                                only_local = local_valid & ~global_valid
                                only_global = global_valid & ~local_valid
                                combined[only_local] = local_values[only_local]
                                combined[only_global] = global_values[only_global]

                                out_band.WriteArray(combined, xoff=xoff, yoff=yoff)
                                progress.update(1)
                finally:
                    local_res_ds = None
                    global_res_ds = None

            out.FlushCache()
            out = None
            return str(out_path)
        finally:
            local_ds = None
            global_ds = None

    def _calculateResidualLR(self, result):
        """Calculate and smooth residuals only on the low-resolution grid."""
        if utils is None:
            raise ImportError(
                "The residual-analysis methods require the project-specific "
                "'utils' module (binomialSmoother/removeEdgeNaNs)."
            )

        scene_lr = self._open_raster(self.lowResFile, "low-resolution image")
        mask_lr = self._open_raster(
            self.lowResQualityFile, "low-resolution quality image"
        )
        result_ds = self._open_raster(result, "disaggregated result")
        try:
            low_data = scene_lr.ReadAsArray().astype(np.float64, copy=False)
            low_quality = mask_lr.ReadAsArray()
            valid_lr = (
                np.isfinite(low_data)
                & (low_data != NODATA_LR)
                & self._low_quality_mask(low_quality)
            )

            size_x = result_ds.RasterXSize
            size_y = result_ds.RasterYSize
            residual_lr = np.full(low_data.shape, np.nan, dtype=np.float64)
            result_band = result_ds.GetRasterBand(1)

            for i, j in np.argwhere(valid_lr):
                xoff = int(j * self.scaleFactor)
                yoff = int(i * self.scaleFactor)
                xsize = min(self.scaleFactor, size_x - xoff)
                ysize = min(self.scaleFactor, size_y - yoff)
                if xsize <= 0 or ysize <= 0:
                    continue

                values = result_band.ReadAsArray(
                    xoff, yoff, xsize, ysize
                ).astype(np.float64, copy=False)
                values[values == NODATA_OUT] = np.nan
                if not np.any(np.isfinite(values)):
                    continue

                if self.disaggregatingTemperature:
                    aggregated = np.nanmean(values ** 4)
                    residual_lr[i, j] = low_data[i, j] ** 4 - aggregated
                else:
                    residual_lr[i, j] = low_data[i, j] - np.nanmean(values)

            return np.asarray(utils.binomialSmoother(residual_lr), dtype=np.float32)
        finally:
            scene_lr = None
            mask_lr = None
            result_ds = None

    def _warpResidualLRToHR(self, residual_lr, hr_template, output_path):
        """Warp an LR residual array to an on-disk HR raster."""
        scene_lr = self._open_raster(self.lowResFile, "low-resolution image")
        try:
            mem_driver = gdal.GetDriverByName("MEM")
            mem = mem_driver.Create(
                "", residual_lr.shape[1], residual_lr.shape[0], 1, gdal.GDT_Float32
            )
            mem.SetGeoTransform(scene_lr.GetGeoTransform())
            mem.SetProjection(scene_lr.GetProjection())
            mem_band = mem.GetRasterBand(1)
            mem_band.SetNoDataValue(NODATA_OUT)
            mem_band.WriteArray(np.nan_to_num(residual_lr, nan=NODATA_OUT))

            gt = hr_template.GetGeoTransform()
            minx = gt[0]
            maxy = gt[3]
            maxx = minx + gt[1] * hr_template.RasterXSize
            miny = maxy + gt[5] * hr_template.RasterYSize

            warped = gdal.Warp(
                str(output_path),
                mem,
                format="GTiff",
                dstSRS=hr_template.GetProjection(),
                xRes=abs(gt[1]),
                yRes=abs(gt[5]),
                outputBounds=(minx, miny, maxx, maxy),
                resampleAlg="bilinear",
                srcNodata=NODATA_OUT,
                dstNodata=NODATA_OUT,
                creationOptions=[
                    "TILED=YES",
                    "COMPRESS=DEFLATE",
                    "PREDICTOR=3",
                    "BIGTIFF=IF_SAFER",
                ],
            )
            if warped is None:
                raise RuntimeError(f"GDAL residual upsampling failed: {output_path}")
            warped.FlushCache()
            warped = None
            mem = None
        finally:
            scene_lr = None

    def _calculateResidual(self, result):
        if utils is None:
            raise ImportError(
                "The residual-analysis methods require the project-specific "
                "'utils' module (binomialSmoother/removeEdgeNaNs)."
            )

        scene_lr = self._open_raster(self.lowResFile, "low-resolution image")
        mask_lr = self._open_raster(
            self.lowResQualityFile, "low-resolution quality image"
        )
        result_ds = self._open_raster(result, "disaggregated result")

        try:
            low_data = scene_lr.ReadAsArray().astype(float)
            low_quality = mask_lr.ReadAsArray()
            valid_lr = (
                np.isfinite(low_data)
                & (low_data != NODATA_LR)
                & self._low_quality_mask(low_quality)
            )

            size_x = result_ds.RasterXSize
            size_y = result_ds.RasterYSize
            residual_lr = np.full(low_data.shape, np.nan, dtype=float)

            for i, j in np.argwhere(valid_lr):
                xoff = int(j * self.scaleFactor)
                yoff = int(i * self.scaleFactor)
                xsize = min(self.scaleFactor, size_x - xoff)
                ysize = min(self.scaleFactor, size_y - yoff)
                if xsize <= 0 or ysize <= 0:
                    continue

                values = result_ds.GetRasterBand(1).ReadAsArray(
                    xoff, yoff, xsize, ysize
                ).astype(float)
                values[values == NODATA_OUT] = np.nan

                if not np.any(np.isfinite(values)):
                    continue

                if self.disaggregatingTemperature:
                    aggregated = np.nanmean(values ** 4)
                    residual_lr[i, j] = low_data[i, j] ** 4 - aggregated
                else:
                    aggregated = np.nanmean(values)
                    residual_lr[i, j] = low_data[i, j] - aggregated

            residual_smoothed = utils.binomialSmoother(residual_lr)

            mem_driver = gdal.GetDriverByName("MEM")
            mem = mem_driver.Create(
                "", low_data.shape[1], low_data.shape[0], 1, gdal.GDT_Float32
            )
            mem.SetGeoTransform(scene_lr.GetGeoTransform())
            mem.SetProjection(scene_lr.GetProjection())
            mem.GetRasterBand(1).SetNoDataValue(np.nan)
            mem.GetRasterBand(1).WriteArray(residual_smoothed)

            gt = result_ds.GetGeoTransform()
            minx = gt[0]
            maxy = gt[3]
            maxx = minx + gt[1] * result_ds.RasterXSize
            miny = maxy + gt[5] * result_ds.RasterYSize

            warped = gdal.Warp(
                "",
                mem,
                format="MEM",
                dstSRS=result_ds.GetProjection(),
                xRes=abs(gt[1]),
                yRes=abs(gt[5]),
                outputBounds=(minx, miny, maxx, maxy),
                resampleAlg="bilinear",
            )
            if warped is None:
                raise RuntimeError("GDAL residual upsampling failed.")

            residual_hr = warped.GetRasterBand(1).ReadAsArray().astype(float)

            # Preserve the original project-specific edge repair, but only where needed.
            nan_locations = np.argwhere(np.isnan(residual_hr))
            for i, j in nan_locations:
                if 0 < i < residual_hr.shape[0] - 1 and 0 < j < residual_hr.shape[1] - 1:
                    residual_hr[i, j] = utils.removeEdgeNaNs(residual_hr, i, j)

            return residual_hr, residual_lr

        finally:
            scene_lr = None
            mask_lr = None
            result_ds = None

    def residualAnalysis(self, disaggregatedFile, output_dir):
        residual_hr, residual_lr = self._calculateResidual(disaggregatedFile)

        scene_hr = self._open_raster(disaggregatedFile, "disaggregated image")
        scene_lr = self._open_raster(self.lowResFile, "low-resolution image")

        try:
            values = scene_hr.GetRasterBand(1).ReadAsArray().astype(float)
            values[values == NODATA_OUT] = np.nan

            if self.disaggregatingTemperature:
                corrected_energy = residual_hr + values ** 4
                corrected = np.where(
                    corrected_energy >= 0, corrected_energy ** 0.25, np.nan
                )
            else:
                corrected = values + residual_hr

            output_dir = Path(output_dir)
            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            sen3_id = Path(self.lowResFile).stem.removeprefix("Subset_")
            sen2_parts = Path(self.highResFile).stem.removeprefix("SR_").split("_")

            satellite = sen2_parts[0]  # e.g. S2B

            tile = next(
                part
                for part in sen2_parts
                if re.fullmatch(r"T\d{2}[A-Z]{3}", part)
            )

            output_path = output_dir / f"TS_{sen3_id}_{satellite}_{tile}.tiff"

            driver = gdal.GetDriverByName("GTiff")
            out = self._create_output(
                driver, output_path, scene_hr, scene_hr.RasterXSize, scene_hr.RasterYSize
            )
            out.GetRasterBand(1).WriteArray(np.nan_to_num(corrected, nan=NODATA_OUT))
            out.FlushCache()
            out = None

            if self.disaggregatingTemperature:
                residual_lr_display = (
                    np.maximum(residual_lr + 273.15 ** 4, 0) ** 0.25 - 273.15
                )
            else:
                residual_lr_display = residual_lr

            res_path = (
                Path(self.lowResFile).parent
                / f"Corrected_Res_LR_{self._output_stem(self.lowResFile)}.tiff"
            )
            out_res = self._create_output(
                driver,
                res_path,
                scene_lr,
                scene_lr.RasterXSize,
                scene_lr.RasterYSize,
            )
            out_res.GetRasterBand(1).WriteArray(np.nan_to_num(residual_lr_display, nan=NODATA_OUT))
            out_res.FlushCache()
            out_res = None

            logger.info(
                "LR residual bias: %.6f",
                np.nanmean(residual_lr_display),
            )

            logger.info(
                "LR residual RMSD: %.6f",
                np.sqrt(np.nanmean(residual_lr_display ** 2)),
            )
            return str(output_path)
        finally:
            scene_hr = None
            scene_lr = None

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    def _doFit(self, goodData_LR, goodData_HR, weight, local):
        options = dict(self.regressorOpt)
        options["max_leaf_nodes"] = 10 if local else 30
        n_samples = int(goodData_HR.shape[0])
        if n_samples < 2:
            raise ValueError("At least two training samples are required to fit a model.")
        options["min_samples_leaf"] = min(10, max(1, n_samples // 3))

        if self.perLeafLinearRegression:
            base_regressor = DecisionTreeRegressorWithLinearLeafRegression(
                linearRegressionExtrapolationRatio=self.linearRegressionExtrapolationRatio,
                decisionTreeRegressorOpt=options,
            )
        else:
            base_regressor = tree.DecisionTreeRegressor(**options)

        reg = ensemble.BaggingRegressor(
            estimator=base_regressor, **self.baggingRegressorOpt
        )

        # A local window can occasionally contain fewer samples than max_samples.
        if goodData_HR.shape[0] <= 1:
            reg.max_samples = 1.0

        return reg.fit(goodData_HR, goodData_LR, sample_weight=weight)

    @staticmethod
    def _doPredict(inData, reg):
        original_shape = inData.shape
        bands = original_shape[2] if len(original_shape) == 3 else 1
        flat = inData.reshape((-1, bands))

        valid = np.all(np.isfinite(flat), axis=1)
        predicted = np.full(flat.shape[0], np.nan, dtype=float)
        if np.any(valid):
            predicted[valid] = reg.predict(flat[valid])

        return predicted.reshape((original_shape[0], original_shape[1]))


class DecisionTreeRegressorWithLinearLeafRegression(tree.DecisionTreeRegressor):
    """Decision tree with a Bayesian linear regression fitted inside each leaf."""

    def __init__(
        self,
        linearRegressionExtrapolationRatio=0.25,
        decisionTreeRegressorOpt=None,
    ):
        # scikit-learn estimators must store constructor parameters *unchanged*
        # so clone() can reconstruct the estimator exactly. Do not copy/modify
        # decisionTreeRegressorOpt here.
        self.linearRegressionExtrapolationRatio = linearRegressionExtrapolationRatio
        self.decisionTreeRegressorOpt = decisionTreeRegressorOpt

        options = {} if decisionTreeRegressorOpt is None else decisionTreeRegressorOpt
        super().__init__(**options)

        # Fitted state; recreated in fit().
        self.leafParameters = {}

    def fit(self, X, y, sample_weight=None, **fit_params):
        super().fit(X, y, sample_weight=sample_weight, **fit_params)

        # IMPORTANT: identify leaves by node ID, not by their predicted value.
        leaf_ids = self.apply(X)
        self.leafParameters = {}

        for leaf_id in np.unique(leaf_ids):
            idx = leaf_ids == leaf_id
            if not np.any(idx):
                continue

            lr = linear_model.BayesianRidge()
            lr.fit(
                X[idx, :],
                y[idx],
                sample_weight=None if sample_weight is None else sample_weight[idx],
            )
            self.leafParameters[int(leaf_id)] = {
                "linearRegression": lr,
                "max": float(np.max(y[idx])),
                "min": float(np.min(y[idx])),
            }

        return self

    def predict(self, X, **predict_params):
        y = super().predict(X, **predict_params)
        leaf_ids = self.apply(X)

        for leaf_id, params in self.leafParameters.items():
            idx = leaf_ids == leaf_id
            if not np.any(idx):
                continue

            values = params["linearRegression"].predict(X[idx, :])
            extrapolation_range = self.linearRegressionExtrapolationRatio * (
                params["max"] - params["min"]
            )
            y[idx] = np.clip(
                values,
                params["min"] - extrapolation_range,
                params["max"] + extrapolation_range,
            )

        return y
