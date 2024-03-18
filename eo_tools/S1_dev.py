import os
from pathlib import Path
import xmltodict
import numpy as np
import rasterio


def geocode_burst(safe_dir, iw=1, pol="vv", burst_idx=1):
    if not os.isdir(safe_dir):
        raise ValueError("Directory not found.")

    pth_tiff = Path(f"{safe_dir}/measurement/").glob(f"*iw{iw}*{pol}*.tiff")
    pth_xml = Path(f"{safe_dir}/annotation/").glob(f"*iw{iw}*{pol}*.xml")
    pth_tiff = list(pth_tiff)[0]
    pth_xml = list(pth_xml)[0]

    meta = read_metadata(pth_xml)

    # general info
    image_info = meta["product"]["imageAnnotation"]["imageInformation"]
    azimuth_time_interval = image_info["azimuthTimeInterval"]
    slant_range_time = image_info["slantRangeTime"]
    range_pixel_spacing = image_info["rangePixelSpacing"]
    product_info = meta["product"]["generalAnnotation"]["productInformation"]
    range_sampling_rate = product_info["rangeSamplingRate"]
    radar_frequency = product_info["radarFrequency"]

    # look for burst info
    burst_info = meta["product"]["swathTiming"]
    lines_per_burst = int(burst_info["linesPerBurst"])
    samples_per_burst = int(burst_info["samplesPerBurst"])
    burst_count = int(burst_info["burstList"]["@count"])
    first_burst = burst_info["burstList"]["burst"][0]
    az_time = first_burst["azimuthTime"]

    first_line = (burst_idx - 1) * lines_per_burst
    im, gcps, gcps_crs = read_chunk(
        pth_tiff, first_line=first_line, number_of_lines=lines_per_burst
    )


def read_metadata(pth_xml):
    # read product XML
    with open(pth_xml) as f:
        meta = xmltodict.parse(f.read())
    return meta


def read_chunk(pth_tiff, first_line=0, number_of_lines=1500):
    with rasterio.open(pth_tiff) as src:
        # prof = src.profile.copy()
        gcps, gcps_crs = src.gcps
        arr = np.log(
            np.abs(src.read(1, window=(0, first_line, src.width, number_of_lines))) + 1
        )
        # TODO: use filter?
        gcps_burst = [
            it
            for it in gcps
            if it.row >= first_line and it.row <= first_line + number_of_lines
        ]
        return arr, gcps_burst, gcps_crs
