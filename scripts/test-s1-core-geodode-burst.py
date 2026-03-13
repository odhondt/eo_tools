# %%
# Uncomment the next block to test conda imports

# import sys
# sys.path.remove("/eo_tools")
# sys.path.append("/eo_tools/") # workaround to include eo_tools_dev
# import eo_tools
# print(f"EO-Tools imported from:")
# print(f"{eo_tools.__file__=}")

from eo_tools.S1.core import S1IWSwath
from eo_tools.S1.core import align, coregister
from eo_tools.S1.util import presum
from eo_tools_dev.util import show_cog, serve_map, palette_phi
from math import pi
import folium
from folium import LayerControl

import numpy as np
import os

import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("numexpr").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


# %%
# change with directory containing your S1 products
data_dir = "/data/S1"
# change with directory containing the results
out_dir = "/data/res/test-s1-core-geocode-burst"
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

# replace with already downloaded and unzipped products (see the other notebooks to download such products)
primary_path = f"{data_dir}/S1A_IW_SLC__1SDV_20241205T172346_20241205T172413_056860_06FB65_2BBF.SAFE.zip"
secondary_path = f"{data_dir}/S1A_IW_SLC__1SDV_20230916T063730_20230916T063757_050349_060FCD_6814.zip"

# subswath to process
iw = 1
# polarization
pol = "vv"
# DEM upsampling
up = 1.8

min_burst = 5
max_burst = 5

# %%
prm = S1IWSwath(primary_path, iw=iw, pol=pol)
overlap = np.round(prm.compute_burst_overlap(2)).astype(int)

lut = []
amps = []
dems = []
for burst_idx in range(min_burst, max_burst + 1):
    log.info(f"---- Processing burst {burst_idx} ----")

    # compute geocoding LUTs for master and slave bursts
    dem_file = prm.fetch_dem_burst(burst_idx, dem_dir="/data/tmp", force_download=False, dem_name="cop-dem-glo-30")
    az_p2g, rg_p2g = prm.geocode_burst(dem_file, burst_idx=burst_idx, dem_upsampling=up)

    # read primary and secondary burst raster
    arr_p = prm.read_burst(burst_idx, True)

    # radiometric calibration
    cal_p = prm.calibration_factor(burst_idx, cal_type="beta")
    arr_p /= cal_p


    # normalize complex coherences
    amps.append(np.abs(arr_p))
    lut.append((az_p2g, rg_p2g))
    dems.append(dem_file)

from eo_tools.S1.core import stitch_bursts

img = stitch_bursts(amps, overlap)

# %%
from eo_tools.S1.core import resample
import rioxarray as riox
from rioxarray.merge import merge_arrays
from eo_tools.auxils import remove

mlt_az = 1
mlt_rg = 4

off = 0
H = int(overlap / 2)
amp_out = presum(img, mlt_az, mlt_rg)
naz = amps[0].shape[0]
list_amp = []
files_to_remove = []
for i in range(min_burst, max_burst + 1):
    log.info(f"Resample burst {i}")
    az_mst, rg_mst = lut[i - min_burst]
    dem_file = dems[i - min_burst]
    cnd = (az_mst >= H - 4) & (az_mst < naz - H + 4)
    az_mst2 = az_mst.copy()
    rg_mst2 = rg_mst.copy()
    az_mst2[~cnd] = np.nan
    rg_mst2[~cnd] = np.nan

    amp_file = f"{out_dir}/remap_burst_{i}_amp.tif"
    files_to_remove.append(amp_file)

    # does the job but not very elegant
    if i == min_burst:
        off2 = off
    else:
        off2 = off - H
    resample(
        amp_out,
        dem_file,
        amp_file,
        (az_mst2 + off2) / mlt_az,
        (rg_mst2) / mlt_rg,
        kernel="bicubic",
    )
    if i == min_burst:
        off += naz - H
    else:
        off += naz - 2 * H

    list_amp.append(riox.open_rasterio(amp_file))

merged_amp = merge_arrays(list_amp)
merged_amp.rio.to_raster(f"{out_dir}/merged_amp.tif")

for fname in files_to_remove:
    remove(fname)
