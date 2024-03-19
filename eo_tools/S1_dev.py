import os
from pathlib import Path
import xmltodict
import numpy as np
import rasterio
from scipy.interpolate import CubicHermiteSpline
from dateutil.parser import isoparse
from eo_tools.dem import retrieve_dem
from shapely.geometry import box


def geocode_burst(safe_dir, iw=1, pol="vv", burst_idx=1, file_dem="autodem.tif"):
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

    # state vectors
    orbit_list = meta["product"]["generalAnnotation"]["orbitList"]
    orbit_count = orbit_list["@count"]
    state_vectors = orbit_list["orbit"]
    
    interp_orb, interp_orb_v = orbit_interpolator(state_vectors)

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

def orbit_interpolator(state_vectors):
    tt0 = isoparse(state_vectors[0]["time"])
    tt = [(isoparse(it["time"]) - tt0).total_seconds() for it in state_vectors]
    xx = [float(it["position"]["x"]) for it in state_vectors]
    yy = [float(it["position"]["y"]) for it in state_vectors]
    zz = [float(it["position"]["z"]) for it in state_vectors]

    vxx = [float(it["velocity"]["x"]) for it in state_vectors]
    vyy = [float(it["velocity"]["y"]) for it in state_vectors]
    vzz = [float(it["velocity"]["z"]) for it in state_vectors]

    interp_orb = CubicHermiteSpline(
        tt, np.array([xx, yy, zz]).T, np.array([vxx, vyy, vzz]).T
    )
    interp_orb_v = interp_orb.derivative(1)
    return interp_orb, interp_orb_v

def auto_dem(file_dem, gcps_burst):
    minmax  = lambda x: (x.min(), x.max())
    xmin, xmax = minmax(np.array([p.x for p in gcps_burst]))
    ymin, ymax = minmax(np.array([p.y for p in gcps_burst]))
    shp = box(xmin, ymin, xmax, ymax)


    if not os.path.exists(file_dem):
        retrieve_dem(shp, file_dem)