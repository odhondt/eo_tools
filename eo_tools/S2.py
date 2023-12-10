import os
import rasterio
from rasterio.warp import reproject, calculate_default_transform
from rasterio.merge import merge
from rasterio import MemoryFile
from rasterio.enums import Resampling
from rasterio import mask
import numpy as np
import pandas as pd

import logging

log = logging.getLogger(__name__)

# TODO: add "all" options for bands or "10m", "20m", "60m", "RGB" + parse single band (not a list)
# TODO: cog
def merge_s2_tiles(paths, bands=["B4", "B3", "B2"], shp=None, aoi_name = None, outputs_prefix="/tmp"):

    # identify distinct products
    dict_products = {}

    df_bands = _make_df_bands()

    for path in paths:
        with rasterio.open(path) as ds:
            tags = ds.tags()
            pid = tags["DATATAKE_1_ID"]
            if pid not in dict_products.keys():
                dict_products[pid] = [path]
            else:
                dict_products[pid].append(path)

    # merge granules from the same product
    for pid, path_list in dict_products.items():
        if aoi_name is None:
            out_dir = f"{outputs_prefix}/{pid}"
        else:
            out_dir = f"{outputs_prefix}/{pid}_{aoi_name}"
        log.info(f"---- Processing data take {pid}")
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for band in bands:
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
                                "driver": "GTiff",
                                "compress": "deflate",
                            }
                        )

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
            # TODO: (optional) stacking?
            with rasterio.open(f"{out_dir}/{band}.tif", "w", **prof) as dst:
                dst.write(arr_merge)

# TODO: improve descriptions
def _make_df_bands():
    df_bands = pd.DataFrame(
        {
            "band": [
                "B2",
                "B3",
                "B4",
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