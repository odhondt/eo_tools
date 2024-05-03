from eo_tools.S1.core import S1IWSwath, coregister, align, stitch_bursts

import numpy as np
import xarray as xr
import rasterio as rio
from rasterio.windows import Window
from rioxarray.merge import merge_arrays
import warnings
import os
from scipy.ndimage import map_coordinates

import logging

log = logging.getLogger(__name__)


def preprocess_insar_iw(
    dir_primary,
    dir_secondary,
    dir_out,
    min_burst=1,
    max_burst=1,
    iw=1,
    pol="vv",
    dir_dem="/tmp",
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

    if max_burst - min_burst > 1:
        tmp_prm = f"{dir_out}/tmp_prm.tif"
        tmp_sec = f"{dir_out}/tmp_secondary.tif"
    else:
        tmp_prm = f"{dir_out}/primary.tif"
        tmp_sec = f"{dir_out}/secondary.tif"

    # TODO check burst indices are vlaid
    prm = S1IWSwath(dir_primary, iw=iw, pol=pol)
    sec = S1IWSwath(dir_secondary, iw=iw, pol=pol)
    overlap = np.round(prm.compute_burst_overlap(2)).astype(int)

    naz = prm.lines_per_burst * (max_burst - min_burst + 1)
    nrg = prm.samples_per_burst

    up = 2

    # if max_burst is not None:
    # _max_burst = max_burst
    # else:
    # _max_burst = min_burst + 1

    luts = _process_bursts(
        prm, sec, tmp_prm, tmp_sec, dir_out, dir_dem, naz, nrg, min_burst, max_burst, up
    )
    # stitching bursts
    if max_burst - min_burst > 1:
        _stitch_bursts(
            tmp_sec,
            f"{dir_out}/secondary.tif",
            prm.lines_per_burst,
            max_burst - min_burst + 1,
            overlap,
        )
        _stitch_bursts(
            tmp_prm,
            f"{dir_out}/primary.tif",
            prm.lines_per_burst,
            max_burst - min_burst + 1,
            overlap,
        )

    # _merge_luts(
    # luts_az, luts_rg, f"{dir_out}/lut.tif", prm.lines_per_burst, overlap, offset=4
    # )
    _merge_luts2(luts, f"{dir_out}/lut.tif", prm.lines_per_burst, overlap, offset=4)

    if max_burst - min_burst > 1:
        if os.path.isfile(tmp_prm):
            os.remove(tmp_prm)
        if os.path.isfile(tmp_sec):
            os.remove(tmp_sec)


def slc2geo(slc_file, lut_file, out_file, mlt_az, mlt_rg):
    """Geocode slc file using a lookup table with optional multilooking.

    Args:
        slc_file (str): file in the slc geometry
        lut_file (str): file containing a lookup table
        out_file (str): output file
        mlt_az (int): number of looks in the azimuth direction
        mlt_rg (int): number of looks in the range direction
    """
    
    with rio.open(slc_file) as ds_slc:
        arr = ds_slc.read()
    with rio.open(lut_file) as ds_lut:
        lut = ds_lut.read()
    

    pass


def _process_bursts(
    prm, sec, tmp_prm, tmp_sec, dir_out, dir_dem, naz, nrg, min_burst, max_burst, up
):
    luts = []
    prof_tmp = dict(width=nrg, height=naz, count=1, dtype="complex64", driver="GTiff")
    # process individual bursts
    warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
    # TODO: try clipping and offsetting LUTs before creating the DataArrays
    # TODO: first try to concatenate before creating da (concat not lazy in xarray)
    with rio.open(tmp_prm, "w", **prof_tmp) as ds_prm:
        with rio.open(tmp_sec, "w", **prof_tmp) as ds_sec:

            for burst_idx in range(min_burst, max_burst + 1):
                log.info(f"---- Processing burst {burst_idx} ----")

                # compute geocoding LUTs for master and slave bursts
                file_dem = prm.fetch_dem_burst(burst_idx, dir_dem, force_download=False)
                az_p2g, rg_p2g, dem_profile = prm.geocode_burst(
                    file_dem, burst_idx=burst_idx, dem_upsampling=up
                )
                az_s2g, rg_s2g, dem_profile = sec.geocode_burst(
                    file_dem, burst_idx=burst_idx, dem_upsampling=up
                )

                # read primary and secondary burst raster
                arr_p = prm.read_burst(burst_idx, True)
                arr_s = sec.read_burst(burst_idx, True)

                # deramp secondary
                pdb_s = sec.deramp_burst(burst_idx)
                arr_s_de = arr_s * np.exp(1j * pdb_s).astype(np.complex64)

                # project slave LUT into master grid
                az_s2p, rg_s2p = coregister(arr_p, az_p2g, rg_p2g, az_s2g, rg_s2g)

                # warp raster secondary and deramping phase
                arr_s2p = align(arr_p, arr_s_de, az_s2p, rg_s2p, order=3)
                pdb_s2p = align(arr_p, pdb_s, az_s2p, rg_s2p, order=3)

                # reramp slave
                arr_s2p = arr_s2p * np.exp(-1j * pdb_s2p).astype(np.complex64)

                # compute topographic phases
                rg_p = np.zeros(arr_s.shape[0])[:, None] + np.arange(0, arr_s.shape[1])
                pht_p = prm.phi_topo(rg_p).reshape(*arr_p.shape)
                pht_s = sec.phi_topo(rg_s2p.ravel()).reshape(*arr_p.shape)
                pha_topo = np.exp(-1j * (pht_p - pht_s)).astype(np.complex64)

                # az_da = _make_da_from_dem(az_p2g, dem_profile)
                # rg_da = _make_da_from_dem(rg_p2g, dem_profile)
                lut_da = _make_da_from_dem(np.stack((az_p2g, rg_p2g)), dem_profile)
                lut_da.rio.to_raster(
                    f"{dir_out}/lut_{burst_idx}.tif"
                )  # , windowed=True)  # , tiled=True)
                # az_da.rio.to_raster(f"{dir_out}/az_{burst_idx}.tif")
                # rg_da.rio.to_raster(f"{dir_out}/rg_{burst_idx}.tif")
                luts.append(f"{dir_out}/lut_{burst_idx}.tif")
                # luts_az.append(f"{dir_out}/az_{burst_idx}.tif")
                # luts_rg.append(f"{dir_out}/rg_{burst_idx}.tif")

                arr_s2p = arr_s2p * pha_topo

                first_line = (burst_idx - min_burst) * prm.lines_per_burst
                ds_prm.write(
                    arr_p, 1, window=Window(0, first_line, nrg, prm.lines_per_burst)
                )
                ds_sec.write(
                    arr_s2p,
                    1,
                    window=Window(0, first_line, nrg, prm.lines_per_burst),
                )
    return luts


def _stitch_bursts(
    file_in, file_out, lines_per_burst, burst_count, overlap, off_burst=1
):
    warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
    H = int(overlap / 2)
    naz = lines_per_burst
    with rio.open(file_in) as src:
        nrg = src.width

        if burst_count >= 2:
            siz = (naz - H) * 2 + (burst_count - 2) * (naz - 2 * H)
        elif burst_count == 1:
            siz = naz - H
        else:
            raise ValueError("Empty burst list")

        prof = dict(width=nrg, height=siz, dtype=src.profile["dtype"], count=src.count)
        with rio.open(file_out, "w", **prof) as dst:

            log.info("Stitching bursts to make a continuous image")
            off_dst = 0
            for i in range(burst_count):
                if i == 0:
                    nlines = naz - H
                    off_src = 0
                    # off += nlines
                elif i == burst_count - 1:
                    nlines = naz - H
                    off_src = H
                    # off += nlines
                else:
                    nlines = naz - 2 * H
                    off_src = H

                for j in range(src.count):
                    arr = src.read(
                        j + 1,
                        window=Window(
                            0, (i + off_burst - 1) * naz + off_src, nrg, nlines
                        ),
                    )
                    # print(f"read line {(i - 1 + off_burst) * naz + off_src} to line {off_dst}")
                    dst.write(
                        arr, window=Window(0, off_dst, nrg, nlines), indexes=j + 1
                    )
                off_dst += nlines


def _make_da_from_dem(arr, dem_prof):

    # dem_ds = xr.open_dataset(file_dem, engine="rasterio")
    # new_width = dem_ds.rio.width * up
    # new_height = dem_ds.rio.height * up
    # new_width = arr.shape[1]
    # new_height = arr.shape[0]

    # upsample dem
    # dem_up_ds = dem_ds.rio.reproject(
    #     dem_ds.rio.crs,
    #     shape=(int(new_height), int(new_width)),
    #     resampling=Resampling.bilinear,
    # )
    da = xr.DataArray(
        # data=arr[None],
        data=arr,
        # coords=dem_up_ds.coords,
        dims=("band", "y", "x"),
        # attrs=dem_up_ds.attrs,
    )
    # dem_prof.update({"count": 2})
    da.rio.write_crs(dem_prof["crs"], inplace=True)
    da.rio.write_transform(dem_prof["transform"], inplace=True)
    da.attrs["_FillValue"] = np.nan
    return da


def _merge_luts(files_az, files_rg, file_out, lines_per_burst, overlap, offset=4):
    log.info("Merging LUT")
    off = 0
    H = int(overlap / 2)
    # phi_out = presum(np.nan_to_num(img), 2, 8)
    naz = lines_per_burst
    to_merge = []
    for i, (file_az, file_rg) in enumerate(zip(files_az, files_rg)):
        az = xr.open_dataset(file_az, engine="rasterio")["band_data"][0]
        rg = xr.open_dataset(file_rg, engine="rasterio")["band_data"][0]
        cnd = (az >= H - offset) & (az < naz - H + offset)
        az.data[~cnd] = np.nan
        rg.data[~cnd] = np.nan

        if i == 0:
            off2 = off
        else:
            off2 = off - H
        az += off2
        if i == 0:
            off += naz - H
        else:
            off += naz - 2 * H
        lut_da = xr.concat((az, rg), "band")
        lut_da.attrs["_FillValue"] = np.nan
        lut_da.rio.write_nodata(np.nan)
        to_merge.append(lut_da)

    merged = merge_arrays(to_merge)
    merged.rio.to_raster(file_out, windowed=True)  # tiled=True)


def _merge_luts2(files_lut, file_out, lines_per_burst, overlap, offset=4):
    log.info("Merging LUT")
    off = 0
    H = int(overlap / 2)
    naz = lines_per_burst
    to_merge = []
    for i, file_lut in enumerate(files_lut):
        lut = xr.open_dataset(file_lut, engine="rasterio")["band_data"]
        cnd = (lut[0] >= H - offset) & (lut[0] < naz - H + offset)
        # lut.data[0, ~cnd] = np.nan
        # lut.data[1, ~cnd] = np.nan
        lut = lut.where(xr.broadcast(cnd, lut)[0], np.nan)

        if i == 0:
            off2 = off
        else:
            off2 = off - H
        lut[0] += off2
        # lut.data[0] += off2
        if i == 0:
            off += naz - H
        else:
            off += naz - 2 * H
        lut.attrs["_FillValue"] = np.nan
        lut.rio.write_nodata(np.nan)
        to_merge.append(lut)

    merged = merge_arrays(to_merge)
    merged.rio.to_raster(file_out, windowed=True)
