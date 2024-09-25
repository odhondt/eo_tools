
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

##function to clean up temporary elements

def remove(path, verb=True):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        if verb:
            log.info(f"Removing {path}")
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        if verb:
            log.info(f"Removing {path}")
        shutil.rmtree(path)  # remove dir and all contains

## return dictonary from config file
def get_config(config_file, proc_section):
    if not os.path.isfile(config_file):
        raise FileNotFoundError("Config file {} does not exist.".format(config_file))
    
    parser = configparser.ConfigParser(allow_no_value=True, converters={'_datetime': _parse_datetime})
    parser.read(config_file)
    out_dict = {}

    try:
        proc_sec = parser[proc_section]
    except KeyError:
        raise KeyError("Section '{}' does not exist in config file {}".format(proc_section, config_file))

    for k, v in proc_sec.items():
        if k == 'download':
            if v.lower() == 'true':
                v = True
        if k == 'processes':
            v = int(v)
        if k.endswith('date'):
            v = proc_sec.get_datetime(k)
        if k == 'int_proc':
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
        if k == 'coh_proc':
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
        if k == 'ha_proc':
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
        if k == 'ext_dem_egm':
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
        if k == 'clean_tmpdir':
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
        if k == 'osvfail':
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
        if k == 'msk_nodatval':
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
        if k == 'ext_dem':
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
        if k == 'subset':
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
        if k == 'l2db_arg':
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
        if k == 't_crs':
            v = int(v)
        if k == 'cohwinrg':
            v = int(v)
        if k == 'cohwinaz':
            v = int(v)
        if k == 'filtersizex':
            v = int(v)
        if k == 'filtersizey':
            v = int(v)
        if k == 'ml_rglook':
            v = int(v)
        if k == 'ml_azlook':
            v = int(v)
        if k == 'decomp_win_size':
            v = int(v)
        if k == 'ext_dem_nodatval':
            v = int(v)
        if k == 'res_int':
            v = int(v)
        if k == 'res_coh':
            v = int(v)
        if k == 'res_ha':
            v = int(v)
        if k == 'ext_dem_file':
            if v == "None":
                v =  None
            else:
                v = v
        if k == 'osvpath':
            if v == "None":
                v =  None
            else:
                v = v
        if k == 'iws':
            v = v.split(',')
        out_dict[k] = v
        if k == 'decompfeats':
            v = v.split(',')
        out_dict[k] = v
        if k == 'gpt_paras':
            if v == "None":
                v =  None
            else:
                v = v.split(',')
    return out_dict
##get datetime from strings such as filenames
def _parse_datetime(s):
    """Custom converter for configparser:
    https://docs.python.org/3/library/configparser.html#customizing-parser-behaviour"""
    if 'T' in s:
        try:
            return dt.strptime(s, '%Y-%m-%dT%H:%M:%S')
        except ValueError as e:
            raise Exception("Parameters 'mindate/maxdate': Could not parse '{}' with datetime format "
                            "'%Y-%m-%dT%H:%M:%S'".format(s)) from e
    else:
        try:
            return dt.strptime(s, '%Y-%m-%d')
        except ValueError as e:
            raise Exception("Parameters 'mindate/maxdate': Could not parse '{}' with datetime format "
                            "'%Y-%m-%d'".format(s)) from e
## group files in nested lists based on common parameter
def group_by_info(infiles, group= None):
##sort files by characteristic of S-1 data (e.g. orbit number, platform, ...)
    info= identify_many(infiles, sortkey= group)
    ##extract file paths of sorted files
    fps_lst = [fp.scene for fp in info]

    ##extract and identify unique keys
    groups= []
    for o in info:
        orb= eval("o."+ group)
        groups.append(orb)

    query_group= groups.count(groups[0]) == len(groups)
    unique_groups= list(set(groups))

    out_files=[]
    if query_group == True:
        out_files=infiles
    else:
        group_idx= [] 
        #index files of key
        for a in unique_groups:
            tmp_groups=[]
            for idx, elem in enumerate(groups):
                    if(a == elem):
                        tmp_groups.append(idx)

            group_idx.append(tmp_groups) 
        ###group by same keyword 
        for i in range(0, len(group_idx)):
            fpsN= list(map(fps_lst.__getitem__, group_idx[i]))
            out_files.append(fpsN)
        
    return(out_files)
## get metadata from zip file for specific polarization and subswaths
def load_metadata(zip_path, subswath, polarization):
    if zip_path.endswith(".zip"):
        archive = ZipFile(zip_path)
        archive_files = archive.namelist()
    else:
        archive_files = glob(f"{zip_path}/**", recursive=True)
    regex_filter = r's1(?:a|b)-iw\d-slc-(?:vv|vh|hh|hv)-.*\.xml'
    metadata_file_list = []
    for item in archive_files:
        if 'calibration' in item or 'rfi' in item:
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
    for grid_list in root.iter('geolocationGrid'):
        for point in grid_list:
            for item in point:
                lat = item.find('latitude').text
                lon = item.find('longitude').text
                line = item.find('line').text
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
                [get_coords(top_right_idx, coord_list)[0], get_coords(top_right_idx, coord_list)[1]],  # Top right
                [get_coords(top_left_idx, coord_list)[0], get_coords(top_left_idx, coord_list)[1]],  # Top left
                [get_coords(bottom_left_idx, coord_list)[0], get_coords(bottom_left_idx, coord_list)[1]],  # Bottom left
                [get_coords(bottom_right_idx, coord_list)[0], get_coords(bottom_right_idx, coord_list)[1]] # Bottom right
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
    df_all = gpd.GeoDataFrame(columns=['subswath', 'burst', 'geometry'], crs='EPSG:4326')
    if not isinstance(target_subswaths, list):
        target_subswaths_ = [target_subswaths]
    else:
        target_subswaths_ = target_subswaths

    for subswath in target_subswaths_:
        if subswath not in ['IW1', 'IW2', 'IW3']:
            raise ValueError("Invalid subswath name (options are: IW1, IW2 or IW3)")
        meta = load_metadata(zip_path = path, subswath = subswath, polarization = polarization)
        total_num_bursts, coord_list = parse_location_grid(meta)
        subswath_geom = parse_subswath_geometry(coord_list, total_num_bursts)
        df = gpd.GeoDataFrame(
                    {'subswath': [subswath.upper()] * len(subswath_geom),
                     'burst': [x for x in subswath_geom.keys()],
                     'geometry': [x for x in subswath_geom.values()]
                    },
                    crs='EPSG:4326'
                )
        df_all = gpd.GeoDataFrame(pd.concat([df_all, df]), crs='EPSG:4326')
    return(df_all)

def process_overlapping_windows(input_array, block_size, overlap, process_fn, *args, **kwargs):
    """Apply a processing function to overlapping blocks of an input array with additional arguments.
    
    Args:
        input_array (numpy.ndarray): The input array of arbitrary dimensions.
        block_size (tuple): Size of the block along each dimension (must match the dimensions of input_array).
        overlap (tuple): Size of the overlap along each dimension (same shape as block_size).
        process_fn (callable): Function to apply on each block. Must return a block of the same shape as input.
        *args: Positional arguments to pass to the processing function.
        **kwargs: Keyword arguments to pass to the processing function.
    
    Returns:
        numpy.ndarray: An array of the same shape as input_array, with the processed blocks.
    
    Raises:
        ValueError: If block_size or overlap have invalid values.
        TypeError: If block_size or overlap are not tuples of integers.
    """
    # Validate block_size and overlap
    if not isinstance(block_size, tuple) or not all(isinstance(b, int) and b >= 1 for b in block_size):
        raise TypeError("block_size must be a tuple of integers, each greater than or equal to 1.")
    
    if not isinstance(overlap, tuple) or not all(isinstance(o, int) and o >= 0 for o in overlap):
        raise TypeError("overlap must be a tuple of non-negative integers.")

    if len(block_size) != input_array.ndim:
        raise ValueError("block_size must have the same number of dimensions as input_array.")
    
    if len(overlap) != input_array.ndim:
        raise ValueError("overlap must have the same number of dimensions as input_array.")
    
    if any(o >= b for o, b in zip(overlap, block_size)):
        raise ValueError("Each element of overlap must be less than the corresponding block_size.")
    
    output_array = np.zeros_like(input_array)
    input_shape = input_array.shape
    
    steps = [
        max(1, b - o)  # Calculate the step size for each dimension (block size minus overlap)
        for b, o in zip(block_size, overlap)
    ]

    # Iterate over the starting indices of each block
    idxs = [
        np.arange(0, input_shape[dim] - block_size[dim] + steps[dim], steps[dim])
        for dim in range(len(input_shape))
    ]
    
    # Use np.ndindex to iterate over each block's starting indices in a multidimensional loop
    for index in np.ndindex(*[len(i) for i in idxs]):
        slices = tuple(
            slice(idxs[dim][i], idxs[dim][i] + block_size[dim])
            for dim, i in enumerate(index)
        )
        block = input_array[slices]
        
        # Apply the processing function with *args and **kwargs
        processed_block = process_fn(block, *args, **kwargs)
        
        # Overlap removal: calculate the slices to place processed block
        slice_with_overlap = tuple(
            slice(overlap[dim] // 2, block_size[dim] - (overlap[dim] // 2 + overlap[dim] % 2))
            for dim in range(len(block_size))
        )
        
        # Calculate where to place the processed block into output_array
        # Adjust the output slices by accounting for overlap
        output_slices = tuple(
            slice(idxs[dim][i] + overlap[dim] // 2, idxs[dim][i] + block_size[dim] - (overlap[dim] // 2 + overlap[dim] % 2))
            for dim, i in enumerate(index)
        )
        
        # Write processed block to the output array
        output_array[output_slices] = processed_block[slice_with_overlap]

    return output_array

