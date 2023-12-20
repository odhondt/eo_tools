import os
import rasterio
from rasterio.warp import reproject, calculate_default_transform
from rasterio.merge import merge
from rasterio import MemoryFile
from rasterio.enums import Resampling
from rasterio import mask
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

import logging

log = logging.getLogger(__name__)


def process_s2_tiles(
    paths, bands=["B4", "B3", "B2"], shp=None, aoi_name=None, outputs_prefix="/tmp"
):
    """Merge Sentinel-2 tiles by grouping them by data take ID. Writes bands as individual COG (Cloud Optimized GeoTIFF) files in a sub-folder.

    Args:
        paths (str): List of paths pointing to Sentinel-2 zipped products like e.g. EODAG download outputs.
        bands (list, optional): list of bands to process. A single string (e.g. "B11") is also valid, as well as "all" to merge all bands.  Defaults to ["B4", "B3", "B2"].
        shp (shapely geometry, optional): If shp is provided, the outputs will be cropped to the geometry. Defaults to None.
        aoi_name (str, optional): adds a suffix to the subfolder name. Useful when the same products are cropped with different geometries, or simply to add a short description of the subfolder content. Defaults to None.
        outputs_prefix (str, optional): path where the subfolder will be created. Defaults to "/tmp".
    """
    df_bands = s2_band_info()

    if not isinstance(bands, list):
        if bands == "all":
            bands_ = list(df_bands["band"])
        else:
            bands_ = [bands]
    else:
        bands_ = bands

    # identify distinct products
    dict_products = {}

    for path in paths:
        with rasterio.open(path) as ds:
            tags = ds.tags()
            pid = tags["DATATAKE_1_ID"]
            if pid not in dict_products.keys():
                dict_products[pid] = [path]
            else:
                dict_products[pid].append(path)

    # merge granules from the same product
    out_dirs = []
    for pid, path_list in dict_products.items():
        # TODO: better dir naming (eg S2A-L1C-2023-12-08-hhmmss like S1)
        if aoi_name is None:
            out_dir = f"{outputs_prefix}/{pid}"
        else:
            out_dir = f"{outputs_prefix}/{pid}_{aoi_name}"
        out_dirs.append(out_dir)
        log.info(f"---- Processing data take {pid}")
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for band in bands_:
            log.info(f"-- Band {band}")
            to_merge = []
            for path in path_list:
                # open granule
                with rasterio.open(path) as src:
                    log.info(f"-- Tile {path}")
                    row = df_bands.loc[band]
                    upscale_factor = int(row["resolution"] / 10)

                    # open sub dataset and read band
                    with rasterio.open(src.subdatasets[row["subd"]]) as subds:
                        prof = subds.profile.copy()
                        src_crs = subds.crs

                        # upsample to 10m
                        if upscale_factor > 1:
                            raster = subds.read(
                                int(row["idx"]),
                                out_shape=(
                                    1,
                                    int(subds.height * upscale_factor),
                                    int(subds.width * upscale_factor),
                                ),
                                resampling=Resampling.bilinear,
                            )
                            src_transform = subds.transform * subds.transform.scale(
                                (subds.width / raster.shape[-1]),
                                (subds.height / raster.shape[-2]),
                            )
                        else:
                            raster = subds.read(int(row["idx"]))
                            src_transform = subds.transform

                        # calculate transform to reproject
                        dst_crs = "EPSG:4326"
                        transform, width, height = calculate_default_transform(
                            subds.crs,
                            dst_crs,
                            raster.shape[-1],
                            raster.shape[-2],
                            *subds.bounds,
                        )
                        prof.update(
                            {
                                "crs": dst_crs,
                                "transform": transform,
                                "width": width,
                                "height": height,
                                "count": 1,
                                "driver": "COG",
                                "compress": "deflate",
                            }
                        )
                        # remove COG unsupported options
                        del prof["blockxsize"]
                        del prof["blockysize"]
                        del prof["tiled"]

                    # reproject to EPSG:4326
                    with MemoryFile() as memfile:
                        with memfile.open(**prof) as tmp_ds:
                            reproject(
                                source=raster,
                                destination=rasterio.band(tmp_ds, 1),
                                src_transform=src_transform,
                                src_crs=src_crs,
                                dst_transform=transform,
                                dst_crs=dst_crs,
                                resampling=Resampling.bilinear,
                            )

                            # crop if needed
                            if shp is not None:
                                raster, transform = mask.mask(tmp_ds, [shp], crop=True)
                                prof.update(
                                    {
                                        "width": raster.shape[-1],
                                        "height": raster.shape[-2],
                                        "transform": transform,
                                    }
                                )
                            else:
                                raster = tmp_ds.read()
                    # Hardcoded quantification
                    # TODO: apply bias correction for processor > 4
                    raster = raster.astype(np.float32) / 10000
                    prof.update(
                        {
                            "dtype": "float32",
                        }
                    )
                    # create dataset to merge
                    memfile = MemoryFile()
                    tmp_ds = memfile.open(**prof)
                    tmp_ds.write(raster)

                    # add to merge list
                    to_merge.append(tmp_ds)

            log.info(f"-- Merging {len(to_merge)} tiles")
            arr_merge, trans_merge = merge(to_merge)
            for ds in to_merge:
                ds.close()
            prof.update(
                {
                    "height": arr_merge.shape[1],
                    "width": arr_merge.shape[2],
                    "transform": trans_merge,
                    "nodata": 0,
                }
            )
            with rasterio.open(f"{out_dir}/{band}.tif", "w", **prof) as dst:
                dst.write(arr_merge)
    return out_dirs


def make_s2_rgb(input_dir):
    bands = ["B4", "B3", "B2"]
    band_files = [Path(path).name for path in glob(f"{input_dir}/*.tif")]
    for band in bands:
        if not f"{band}.tif" in band_files:
            raise FileNotFoundError(
                "Missing band. Please create RGB (B4, B3, B2) bands with process_s2_tiles."
            )

    with rasterio.open(f"{input_dir}/B4.tif") as ds:
        prof = ds.profile.copy()

    prof.update({"count": 3, "dtype": "uint8"})

    with rasterio.open(f"{input_dir}/RGB.tif", "w", **prof) as dst:
        for i, band in enumerate(bands):
            with rasterio.open(f"{input_dir}/{band}.tif") as src:
                data = (255 * src.read(1).clip(0, 1)).astype("uint8")
                dst.write(data, i + 1)


# TODO: improve descriptions
def s2_band_info():
    """Returns a pandas dataframe with information about Sentinel-2 bands."""
    # Band order was obtained with rasterio by looping on tags for each subdataset band
    df_bands = pd.DataFrame(
        {
            "band": [
                "B4",
                "B3",
                "B2",
                "B8",
                "B5",
                "B6",
                "B7",
                "B8A",
                "B11",
                "B12",
                "B1",
                "B9",
                "B10",
            ],
            "resolution": [10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 60, 60, 60],
            "subd": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2],
            "idx": [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 1, 2, 3],
            "description": [
                "Blue",
                "Green",
                "Red",
                "VNIR",
                "VNIR",
                "VNIR",
                "VNIR",
                "VNIR",
                "SWIR",
                "SWIR",
                "Coastal",
                "SWIR",
                "SWIR",
            ],
        }
    ).set_index("band")
    return df_bands
