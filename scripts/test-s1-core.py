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
logging.getLogger("numexpr").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


# %%
# change with directory containing your S1 products
data_dir = "/data/S1"
# change with directory containing the results
out_dir = "/data/res/test-s1-core"
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

# replace with already downloaded and unzipped products (see the other notebooks to download such products)
primary_dir = f"{data_dir}/S1A_IW_SLC__1SDV_20230904T063730_20230904T063757_050174_0609E3_DAA1.zip"
secondary_dir = f"{data_dir}/S1A_IW_SLC__1SDV_20230916T063730_20230916T063757_050349_060FCD_6814.zip"

# subswath to process
iw = 1
# polarization
pol = "vv"
# DEM upsampling
up = 1.8

min_burst = 3
max_burst = 6

# %%
ifgs = []
lut = []
dems = []

prm = S1IWSwath(primary_dir, iw=iw, pol=pol)
sec = S1IWSwath(secondary_dir, iw=iw, pol=pol)
overlap = np.round(prm.compute_burst_overlap(2)).astype(int)

for burst_idx in range(min_burst, max_burst + 1):
    log.info(f"---- Processing burst {burst_idx} ----")

    # compute geocoding LUTs for master and slave bursts
    file_dem = prm.fetch_dem_burst(burst_idx, dir_dem="/data/tmp", force_download=False)
    az_p2g, rg_p2g, _ = prm.geocode_burst(
        file_dem, burst_idx=burst_idx, dem_upsampling=up
    )
    az_s2g, rg_s2g, _ = sec.geocode_burst(
        file_dem, burst_idx=burst_idx, dem_upsampling=up
    )

    # read primary and secondary burst raster
    arr_p = prm.read_burst(burst_idx, True)
    arr_s = sec.read_burst(burst_idx, True)

    # radiometric calibration 
    cal_p = prm.calibration_factor(burst_idx, cal_type="beta")
    arr_p /= cal_p
    cal_s = sec.calibration_factor(burst_idx, cal_type="beta")
    arr_s /= cal_s

    # deramp secondary
    pdb_s = sec.deramp_burst(burst_idx)
    arr_s_de = arr_s * np.exp(1j * pdb_s).astype(np.complex64)

    # project slave LUT into master grid
    az_s2p, rg_s2p = coregister(arr_p, az_p2g, rg_p2g, az_s2g, rg_s2g)

    # warp raster slave and deramping phase
    arr_s2p = align(arr_s_de, az_s2p, rg_s2p, kernel="bicubic")
    pdb_s2p = align(pdb_s, az_s2p, rg_s2p, kernel="bicubic")

    # reramp slave
    arr_s2p = arr_s2p * np.exp(-1j * pdb_s2p).astype(np.complex64)

    # compute topographic phases
    rg_p = np.zeros(arr_s.shape[0])[:, None] + np.arange(0, arr_s.shape[1])
    pht_p = prm.phi_topo(rg_p).reshape(*arr_p.shape)
    pht_s = sec.phi_topo(rg_s2p.ravel()).reshape(*arr_p.shape)
    pha_topo = np.exp(-1j * (pht_p - pht_s)).astype(np.complex64)

    # interferogram without topographic phase
    ifg = arr_s2p.conj() * arr_p * pha_topo.conj()

    # normalize complex coherences
    ifgs.append(ifg)
    lut.append((az_p2g, rg_p2g))
    dems.append(file_dem)

# %%
from eo_tools.S1.core import fast_esd

fast_esd(ifgs, overlap)

# %%
from eo_tools.S1.core import stitch_bursts

img = stitch_bursts(ifgs, overlap)


# %%
from eo_tools.S1.core import resample
import rioxarray as riox
from rioxarray.merge import merge_arrays
from eo_tools.auxils import remove

mlt_az = 2
mlt_rg = 8

off = 0
H = int(overlap / 2)
phi_out = presum(img, mlt_az, mlt_rg)
naz = ifgs[0].shape[0]
list_ifg = []
files_to_remove = []
for i in range(min_burst, max_burst + 1):
    log.info(f"Resample burst {i}")
    az_mst, rg_mst = lut[i - min_burst]
    file_dem = dems[i - min_burst]
    cnd = (az_mst >= H - 4) & (az_mst < naz - H + 4)
    az_mst2 = az_mst.copy()
    rg_mst2 = rg_mst.copy()
    az_mst2[~cnd] = np.nan
    rg_mst2[~cnd] = np.nan

    file_ifg = f"{out_dir}/remap_burst_{i}_ifg.tif"
    files_to_remove.append(file_ifg)

    # does the job but not very elegant
    if i == min_burst:
        off2 = off
    else:
        off2 = off - H
    resample(
        phi_out,
        file_dem,
        file_ifg,
        (az_mst2 + off2) / mlt_az,
        (rg_mst2) / mlt_rg,
        kernel="bicubic",
    )
    if i == min_burst:
        off += naz - H
    else:
        off += naz - 2 * H

    list_ifg.append(riox.open_rasterio(file_ifg))

merged_ifg = merge_arrays(list_ifg)
merged_ifg.rio.to_raster(f"{out_dir}/merged_ifg.tif")

for fname in files_to_remove:
    remove(fname)


# %%
import xarray as xr

# Finite no data value for TiTiler
nodata = -9999
# avoid metadata being lost in arithmetic opetations
xr.set_options(keep_attrs=True)
ifg = riox.open_rasterio(f"{out_dir}/merged_ifg.tif")
phi = np.arctan2(ifg[1], ifg[0])
phi = phi.fillna(nodata)
phi.attrs["_FillValue"] = nodata
phi.rio.to_raster(f"{out_dir}/merged_phi.tif", nodata=nodata)

ref_dir = "/data/reference/S1_InSAR_VV_2023-09-04-063730__2023-09-16-063730_Morocco"
m = folium.Map()
_ = show_cog(f"{ref_dir}/phi.tif", m, rescale=f"{-pi},{pi}", colormap=palette_phi())
_ = show_cog(f"{out_dir}/merged_phi.tif", m, rescale=f"{-pi},{pi}", colormap=palette_phi())
LayerControl().add_to(m)

# open in a browser
serve_map(m)