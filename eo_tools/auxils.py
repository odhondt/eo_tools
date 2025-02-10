from pyroSAR import identify_many
import os, shutil
from datetime import datetime as dt
import geopandas as gpd
from shapely.geometry import Polygon
import configparser
from zipfile import ZipFile
from glob import glob
import re
import xml.etree.ElementTree as ET
import pandas as pd
import geopandas as gpd
import numpy as np

import logging

log = logging.getLogger(__name__)


def remove(path, verb=True):
    """param <path> could either be relative or absolute."""
    if os.path.isfile(path) or os.path.islink(path):
        if verb:
            log.info(f"Removing {path}")
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        if verb:
            log.info(f"Removing {path}")
        shutil.rmtree(path)  # remove dir and all contains

## get metadata from zip file for specific polarization and subswaths
def load_metadata(zip_path, subswath, polarization):
    if zip_path.endswith(".zip"):
        archive = ZipFile(zip_path)
        archive_files = archive.namelist()
    else:
        archive_files = glob(f"{zip_path}/**", recursive=True)
    regex_filter = r"s1(?:a|b|c)-iw\d-slc-(?:vv|vh|hh|hv)-.*\.xml"
    metadata_file_list = []
    for item in archive_files:
        if "calibration" in item or "rfi" in item:
            continue
        match = re.search(regex_filter, item)
        if match:
            metadata_file_list.append(item)
    target_file = None
    for item in metadata_file_list:
        if subswath.lower() in item and polarization.lower() in item:
            target_file = item
    if zip_path.endswith(".zip"):
        return archive.open(target_file)
    else:
        return open(target_file)


## get total number of bursts and their coordinates from metadata
def parse_location_grid(metadata):
    tree = ET.parse(metadata)
    root = tree.getroot()
    lines = []
    coord_list = []
    for grid_list in root.iter("geolocationGrid"):
        for point in grid_list:
            for item in point:
                lat = item.find("latitude").text
                lon = item.find("longitude").text
                line = item.find("line").text
                lines.append(line)
                coord_list.append((float(lat), float(lon)))
    total_num_bursts = len(set(lines)) - 1

    return total_num_bursts, coord_list


## get subswath geometry from each burst
def parse_subswath_geometry(coord_list, total_num_bursts):
    def get_coords(index, coord_list):
        coord = coord_list[index]
        assert isinstance(coord[1], float)
        assert isinstance(coord[0], float)
        return coord[1], coord[0]

    bursts_dict = {}
    top_right_idx = 0
    top_left_idx = 20
    bottom_left_idx = 41
    bottom_right_idx = 21

    for burst_num in range(1, total_num_bursts + 1):
        burst_polygon = Polygon(
            [
                [
                    get_coords(top_right_idx, coord_list)[0],
                    get_coords(top_right_idx, coord_list)[1],
                ],  # Top right
                [
                    get_coords(top_left_idx, coord_list)[0],
                    get_coords(top_left_idx, coord_list)[1],
                ],  # Top left
                [
                    get_coords(bottom_left_idx, coord_list)[0],
                    get_coords(bottom_left_idx, coord_list)[1],
                ],  # Bottom left
                [
                    get_coords(bottom_right_idx, coord_list)[0],
                    get_coords(bottom_right_idx, coord_list)[1],
                ],  # Bottom right
            ]
        )

        top_right_idx += 21
        top_left_idx += 21
        bottom_left_idx += 21
        bottom_right_idx += 21

        bursts_dict[burst_num] = burst_polygon

    return bursts_dict


## get geometry of individual bursts
def get_burst_geometry(path, target_subswaths, polarization):
    df_all = gpd.GeoDataFrame(
        columns=["subswath", "burst", "geometry"], crs="EPSG:4326"
    )
    if not isinstance(target_subswaths, list):
        target_subswaths_ = [target_subswaths]
    else:
        target_subswaths_ = target_subswaths

    for subswath in target_subswaths_:
        if subswath not in ["IW1", "IW2", "IW3"]:
            raise ValueError("Invalid subswath name (options are: IW1, IW2 or IW3)")
        meta = load_metadata(
            zip_path=path, subswath=subswath, polarization=polarization
        )
        total_num_bursts, coord_list = parse_location_grid(meta)
        subswath_geom = parse_subswath_geometry(coord_list, total_num_bursts)
        df = gpd.GeoDataFrame(
            {
                "subswath": [subswath.upper()] * len(subswath_geom),
                "burst": [x for x in subswath_geom.keys()],
                "geometry": [x for x in subswath_geom.values()],
            },
            crs="EPSG:4326",
        )
        df_all = gpd.GeoDataFrame(pd.concat([df_all, df]), crs="EPSG:4326")
    return df_all


def block_process(img, block_size, overlap_size, fun, *fun_args, **kwargs):
    """
    Block processing of a multi-channel 2-D image (or tuple of images) with an arbitrary function. Blocks can overlap. In this case, the overlap is added to the block size.

    Args:
        img (array or tuple): Input image or tuple of arrays with shape (nl, nc, ...)
        fun (callable): Function to apply. The first arguments must be the inputs.
        block_size (tuple of ints): Height and width of blocks.
        overlap_size (tuple of ints, optional): Height and width of overlaps.
        *fun_args: Additional positional arguments for the function.
        **kwargs: Additional keyword arguments.

    Returns:
        array: Processed output image with the same shape as the input image (or tuples).

    Raises:
        ValueError: If overlap is less than 1 or if the output shape is incompatible with the input shape.

    Notes:
        The function to be applied has to have arguments of the form (in1, in2, ..., par1, par2, ...)
        with inputs grouped at the beginning. If not, write a wrapper that follows this order.
    """
    # Validate block_size
    if not isinstance(block_size, tuple) or not all(
        isinstance(b, int) and b >= 1 for b in block_size
    ):
        raise TypeError(
            "block_size must be a tuple of integers, each greater than or equal to 1."
        )

    # Validate overlap
    if not isinstance(overlap_size, tuple) or not all(
        isinstance(o, int) and o >= 0 for o in overlap_size
    ):
        raise TypeError("overlap must be a tuple of non-negative integers.")

    if len(block_size) != 2:
        raise ValueError("block_size must be of length 2.")

    if len(overlap_size) != 2:
        raise ValueError("overlap must be of length 2.")

    # Parse block and overlap sizes
    block_height, block_width = block_size
    olap_height, olap_width = overlap_size

    # Extract dimensions of the input image
    if isinstance(img, tuple):
        ih, iw = img[0].shape[:2]
    else:
        ih, iw = img.shape[:2]

    # padding makes the code much simpler
    pad_left, pad_right = (olap_width, iw - block_width * (iw // block_width))
    pad_top, pad_bottom = (olap_height, ih - block_height * (ih // block_height))

    img_ = (
        np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)))
        if not isinstance(img, tuple)
        else tuple(
            np.pad(x, ((pad_top, pad_bottom), (pad_left, pad_right))) for x in img
        )
    )

    # Preallocate output image with the same shape as the input image
    imgout = (
        np.zeros_like(img_) if not isinstance(img_, tuple) else np.zeros_like(img_[0])
    )

    for i in range(pad_top, img_.shape[0], block_height):
        for j in range(pad_left, img_.shape[1], block_width):

            # slice inputs
            sl = (
                slice(i - olap_height, i + block_height + olap_height),
                slice(j - olap_width, j + block_width + olap_width),
            )

            if isinstance(img, tuple):
                blk = tuple(x[sl] for x in img_)
            else:
                blk = (img_[sl],)

            # process and crop
            processed_block = fun(*blk, *fun_args, **kwargs)[
                olap_height : olap_height + block_height,
                olap_width : olap_width + block_width,
            ]

            # slice outputs
            sl2 = (
                slice(i, i + block_height),
                slice(j, j + block_width),
            )
            imgout[sl2] = processed_block

    if isinstance(imgout, tuple):
        oh, ow = imgout[0].shape
    else:
        oh, ow = imgout.shape
    return imgout[pad_top : oh - pad_bottom, pad_left : ow - pad_right]
