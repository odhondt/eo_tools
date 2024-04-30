from eo_tools.S1.core import S1IWSwath, coregister, align, stitch_bursts

import numpy as np
import xarray as xr
import rasterio as rio
import os

import logging

log = logging.getLogger(__name__)


def preprocess_insar_iw(
    dir_primary,
    dir_secondary,
    dir_out,
    dir_dem="/tmp",
    iw=1,
    pol="vv",
    min_burst=1,
    max_burst=None,
    # force_write=False,
):
    """Pre-process S1 InSAR subswaths pairs

    Args:
        dir_primary (str): directory containing the primary product of the pair
        dir_secondary (str): _description_
        dir_out (str): output directory
        dir_dem (str, optional): directory where DEMs used for geocoding are stored. Defaults to "/tmp".
        iw (int, optional): subswath index. Defaults to 1.
        pol (str, optional): polarization ('vv','vh'). Defaults to "vv".
        min_burst (int, optional): First burst to process. Defaults to 1.
        max_burst (int, optional): Last burst to process. Defaults to None.
        # force_write (bool, optional): Force overwriting results. Defaults to False.
    """
    # compute LUTS and SLC stack for each burst
    # stitch SLC
    # mosaic LUTs

    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)

    prm = S1IWSwath(dir_primary, iw=iw, pol=pol)
    sec = S1IWSwath(dir_secondary, iw=iw, pol=pol)
    overlap = np.round(prm.compute_burst_overlap(2)).astype(int)

    up = 2

    if max_burst is not None:
        _max_burst = max_burst
    else:
        _max_burst = min_burst + 1

    luts_az = []
    luts_rg = []
    bursts_prm = []
    bursts_sec = []
    # process individual bursts
    for burst_idx in range(min_burst, _max_burst + 1):
        log.info(f"---- Processing burst {burst_idx} ----")

        # compute geocoding LUTs for master and slave bursts
        file_dem = prm.fetch_dem_burst(burst_idx, dir_dem, force_download=False)
        az_p2g, rg_p2g = prm.geocode_burst(
            file_dem, burst_idx=burst_idx, dem_upsampling=up
        )
        az_s2g, rg_s2g = sec.geocode_burst(
            file_dem, burst_idx=burst_idx, dem_upsampling=up
        )

        nl, nc = az_p2g.shape

        # read primary and secondary burst raster
        arr_p = prm.read_burst(burst_idx, True)
        arr_s = sec.read_burst(burst_idx, True)

        # deramp secondary
        pdb_s = sec.deramp_burst(burst_idx)
        arr_s_de = arr_s * np.exp(1j * pdb_s).astype(np.complex64)

        # project slave LUT into master grid
        az_s2p, rg_s2p = coregister(arr_p, az_p2g, rg_p2g, az_s2g, rg_s2g)

        # warp raster slave and deramping phase
        arr_s2p = align(arr_p, arr_s_de, az_s2p, rg_s2p, order=3)
        pdb_s2p = align(arr_p, pdb_s, az_s2p, rg_s2p, order=3)

        # reramp slave
        arr_s2p = arr_s2p * np.exp(-1j * pdb_s2p).astype(np.complex64)

        # compute topographic phases
        rg_p = np.zeros(arr_s.shape[0])[:, None] + np.arange(0, arr_s.shape[1])
        pht_p = prm.phi_topo(rg_p).reshape(*arr_p.shape)
        pht_s = sec.phi_topo(rg_s2p.ravel()).reshape(*arr_p.shape)
        pha_topo = np.exp(-1j * (pht_p - pht_s)).astype(np.complex64)

        az_da = _make_da_from_dem(az_p2g, file_dem, up)
        rg_da = _make_da_from_dem(rg_p2g, file_dem, up)
        luts_az.append(az_da)
        luts_rg.append(rg_da)

        arr_s2p = arr_s2p * pha_topo
        bursts_prm.append(arr_p)
        bursts_sec.append(arr_s)

    # stitching bursts
    img_prm = stitch_bursts(bursts_prm, overlap)
    img_sec = stitch_bursts(bursts_sec, overlap)
    profile = dict(
        width=img_prm.shape[1],
        height=img_prm.shape[0],
        dtype=img_prm.real.dtype,
        count=4,
        nodata=0,
    )
    with rio.open(f"{dir_out}/slc_pair.tif", "w", **profile) as ds:
        ds.write(img_prm.real, 1)
        ds.write(img_prm.imag, 2)
        ds.write(img_sec.real, 3)
        ds.write(img_sec.imag, 4)

    _merge_luts(
        prm.lines_per_burst, luts_az, luts_rg, f"{dir_out}/lut.tif", overlap, offset=4
    )


def slc2geo(slc_file, lut_file, out_file, mlt_az, mlt_rg):
    """Geocode slc file using a lookup table with optional multilooking.

    Args:
        slc_file (str): file in the slc geometry
        lut_file (str): file containing a lookup table
        out_file (str): output file
        mlt_az (int): number of looks in the azimuth direction
        mlt_rg (int): number of looks in the range direction
    """
    pass


def _make_da_from_dem(arr, file_dem):

    from rasterio.enums import Resampling

    dem_ds = xr.open_dataset(file_dem, engine="rasterio")
    # new_width = dem_ds.rio.width * up
    # new_height = dem_ds.rio.height * up
    new_width = arr.shape[1]
    new_height = arr.shape[0]

    # upsample dem
    dem_up_ds = dem_ds.rio.reproject(
        dem_ds.rio.crs,
        shape=(int(new_height), int(new_width)),
        resampling=Resampling.bilinear,
    )
    da = xr.DataArray(
        data=arr[None],
        coords=dem_up_ds.coords,
        dims=("band", "y", "x"),
        attrs=dem_up_ds.attrs,
    )
    da.attrs["_FillValue"] = np.nan
    return da


from rioxarray.merge import merge_arrays

# xr.set_options(keep_attrs=True)


def _merge_luts(lines_per_burst, luts_az, luts_rg, file_out, overlap, offset=4):
    off = 0
    H = int(overlap / 2)
    # phi_out = presum(np.nan_to_num(img), 2, 8)
    naz = lines_per_burst
    to_merge = []
    for i in range(len(luts_az)):
        az_mst, rg_mst = luts_az[i].copy(), luts_rg[i].copy()
        cnd = (az_mst >= H - offset) & (az_mst < naz - H + offset)
        az_mst.data[~cnd] = np.nan
        rg_mst.data[~cnd] = np.nan

        # does the job but not very elegant
        if i == 0:
            off2 = off
        else:
            off2 = off - H
        az_mst = az_mst + off2
        if i == 0:
            off += naz - H
        else:
            off += naz - 2 * H
        to_merge.append(xr.concat((az_mst, rg_mst), 0))

    merged = merge_arrays(to_merge)
    merged.rio.to_raster("/data/res/test_merge_az.tif")
