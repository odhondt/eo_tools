from eo_tools.S1.core import S1IWSwath, coregister, align, stitch_bursts

import numpy as np
import xarray as xr
import rasterio as rio
from rasterio.windows import Window
from rioxarray.merge import merge_arrays
import warnings
import os
from scipy.ndimage import map_coordinates
from eo_tools.S1.util import presum, boxcar
from memory_profiler import profile
import dask.array as da
from rasterio.errors import NotGeoreferencedWarning

import logging

log = logging.getLogger(__name__)


# TODO: make class and attach different paths (primary, secondary, lut) ?
def preprocess_insar_iw(
    dir_primary,
    dir_secondary,
    dir_out,
    iw=1,
    pol="vv",
    min_burst=1,
    max_burst=None,
    dir_dem="/tmp",
    apply_fast_esd=True,
    warp_polynomial_order=3,
    dem_upsampling=2,
    dem_buffer_arc_sec=20,
    dem_force_download=False,
):
    """Pre-process S1 InSAR subswaths pairs. Write coregistered primary and secondary SLC files as well as a lookup table that can be used to geocode rasters in the single-look radar geometry.

    Args:
        dir_primary (str): directory containing the primary SLC product of the pair.
        dir_secondary (str): directory containing the secondary SLC product of the pair.
        dir_out (str): output directory (creating it if does not exist).
        dir_dem (str, optional): directory where DEMs used for geocoding are stored. Defaults to "/tmp".
        iw (int, optional): subswath index. Defaults to 1.
        pol (str, optional): polarization ('vv','vh'). Defaults to "vv".
        min_burst (int, optional): first burst to process. Defaults to 1.
        max_burst (int, optional): fast burst to process. If not set, last burst of the subswath. Defaults to None.
        dir_dem (str, optional): directory where the DEM is downloaded. Must be created beforehand. Defaults to "/tmp".
        apply_fast_esd: (bool, optional): correct the phase to avoid jumps between bursts. This has no effect if only one burst is processed. Defaults to True.
        warp_polynomial_order (int, optional): polynomial order used to align secondary SLC. Defaults to 3.
        dem_upsampling (float, optional): Upsample the DEM, it is recommended to keep the default value. Defaults to 2.
        dem_buffer_arc_sec (float, optional): Increase if the image area is not completely inside the DEM.
        dem_force_download (bool, optional): To reduce execution time, DEM files are stored on disk. Set to True to redownload these files if necessary. Defaults to false.

    Notes:
        DEM-assisted coregistration is performed to align the secondary with the master. A lookup table file is written to allow the geocoding images from the radar (single-look) grid to the geographic coordinates of the DEM. Bursts are stitched together to form continuous images. All output files are in the GeoTiff format that can be handled by most GIS softwares and geospatial raster tools such as GDAL and rasterio. Because they are in the SAR geometry, SLC rasters are not georeferenced.
    """

    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)

    if iw not in [1, 2, 3]:
        ValueError("iw must be 1, 2 or 3")

    if pol not in ["vv", "vh"]:
        ValueError("pol must be 'vv' or 'vh'")

    prm = S1IWSwath(dir_primary, iw=iw, pol=pol)
    sec = S1IWSwath(dir_secondary, iw=iw, pol=pol)

    # TODO make some checks on product orbits, burst_count

    overlap = np.round(prm.compute_burst_overlap(2)).astype(int)

    if max_burst is None:
        max_burst_ = prm.burst_count
    else:
        max_burst_ = max_burst

    if max_burst_ > min_burst:
        tmp_prm = f"{dir_out}/tmp_primary.tif"
        tmp_sec = f"{dir_out}/tmp_secondary.tif"
    elif max_burst_ < min_burst:
        raise ValueError("max_burst must be >= min_burst")
    else:
        tmp_prm = f"{dir_out}/primary.tif"
        tmp_sec = f"{dir_out}/secondary.tif"

    if (
        max_burst_ > prm.burst_count
        or max_burst_ < 1
        or min_burst > prm.burst_count
        or min_burst < 1
    ):
        raise ValueError(
            f"min_burst and max_burst must be values between 1 and {prm.burst_count}"
        )

    naz = prm.lines_per_burst * (max_burst_ - min_burst + 1)
    nrg = prm.samples_per_burst

    luts = _process_bursts(
        prm,
        sec,
        tmp_prm,
        tmp_sec,
        dir_out,
        dir_dem,
        naz,
        nrg,
        min_burst,
        max_burst_,
        dem_upsampling,
        dem_buffer_arc_sec,
        dem_force_download,
        order=warp_polynomial_order,
    )

    if (max_burst_ > min_burst) & apply_fast_esd:
        _apply_fast_esd(
            tmp_prm, tmp_sec, min_burst, max_burst_, prm.lines_per_burst, nrg, overlap
        )

    if max_burst_ > min_burst:
        _stitch_bursts(
            tmp_sec,
            f"{dir_out}/secondary.tif",
            prm.lines_per_burst,
            max_burst_ - min_burst + 1,
            overlap,
        )
        _stitch_bursts(
            tmp_prm,
            f"{dir_out}/primary.tif",
            prm.lines_per_burst,
            max_burst_ - min_burst + 1,
            overlap,
        )

    _merge_luts(luts, f"{dir_out}/lut.tif", prm.lines_per_burst, overlap, offset=4)

    log.info("Cleaning temporary files")
    if max_burst_ > min_burst:
        if os.path.isfile(tmp_prm):
            os.remove(tmp_prm)
        if os.path.isfile(tmp_sec):
            os.remove(tmp_sec)
    for i in range(min_burst, max_burst_ + 1):
        fname = f"{dir_out}/lut_{i}.tif"
        if os.path.isfile(fname):
            os.remove(fname)

    log.info("Done")


def slc2geo(
    slc_file,
    lut_file,
    out_file,
    mlt_az=1,
    mlt_rg=1,
    order=3,
    write_phase=False,
    magnitude_only=False,
):
    """Reproject slc file to a geographic grid using a lookup table with optional multilooking.

    Args:
        slc_file (str): file in the SLC radar geometry
        lut_file (str): file containing a lookup table (output of the `preprocess_insar_iw` function)
        out_file (str): output file
        mlt_az (int): number of looks in the azimuth direction. Defaults to 1.
        mlt_rg (int): number of looks in the range direction. Defaults to 1.
        order (int): order of the polynomial kernel for resampling. Defaults to 3.
        write_phase (bool): writes the array's phase instead of its complex values. Defaults to False.
        magnitude_only (bool): writes the array's magnitude instead of its complex values. Has no effect it `write_phase` is True. Defaults to False.
    Note:
        Multilooking is recommended as it reduces the spatial resolution and mitigates speckle effects.
    """
    log.info("Project image with the lookup table.")
    with rio.open(slc_file) as ds_slc:
        arr = ds_slc.read()
        prof_src = ds_slc.profile.copy()
    with rio.open(lut_file) as ds_lut:
        lut = ds_lut.read()
        prof_dst = ds_lut.profile.copy()

    if prof_src["count"] != 1:
        raise ValueError("Only single band rasters are supported.")

    if not np.iscomplexobj(arr) and write_phase:
        raise ValueError("Cannot compute phase of a real array.")

    valid = ~np.isnan(lut[0]) & ~np.isnan(lut[1])

    if (mlt_az == 1) & (mlt_rg == 1):
        arr_ = arr[0].copy()
    else:
        arr_ = presum(arr[0], mlt_az, mlt_rg)

    # remove nan because of map_coordinates
    msk = np.isnan(arr_)
    arr_[msk] = 0

    if np.iscomplexobj(arr_):
        nodata = np.nan + 1j * np.nan
        arr_out = np.full_like(lut[0], nodata, dtype=prof_src["dtype"])
    else:
        nodata = np.nan
        arr_out = np.full_like(lut[0], nodata, dtype=prof_src["dtype"])
    msk_out = np.ones_like(lut[0], dtype=bool)

    arr_out[valid] = map_coordinates(
        arr_,
        (lut[0][valid] / mlt_az, lut[1][valid] / mlt_rg),
        order=order,
        cval=nodata,
        prefilter=False,
    )
    msk_out[valid] = map_coordinates(
        msk, (lut[0][valid] / mlt_az, lut[1][valid] / mlt_rg), order=0
    )
    if np.iscomplexobj(arr_out):
        arr_out[msk_out] = np.nan + 1j * np.nan
    else:
        arr_out[msk_out] = np.nan

    prof_dst.update({k: prof_src[k] for k in ["count", "dtype", "nodata"]})
    if write_phase:
        phi = np.angle(arr_out)
        nodata = -9999
        phi[np.isnan(phi)] = nodata
        prof_dst.update({"dtype": phi.dtype.name, "nodata": nodata})
        with rio.open(out_file, "w", **prof_dst) as dst:
            dst.write(phi, 1)
    else:
        if magnitude_only:
            mag = np.abs(arr_out)
            nodata = 0
            mag[np.isnan(mag)] = nodata
            prof_dst.update({"dtype": mag.dtype.name, "nodata": nodata})
            with rio.open(out_file, "w", **prof_dst) as dst:
                dst.write(mag, 1)
        else:
            with rio.open(out_file, "w", **prof_dst) as dst:
                dst.write(arr_out, 1)


# TODO optional chunk processing
def interferogram(file_prm, file_sec, file_out):
    """Compute an interferogram from two SLC image files.

    Args:
        file_prm (str): GeoTiff file of the primary SLC image
        file_sec (str): GeoTiff file of the secondary SLC image
        file_out (str): output file
    """
    log.info("Computing interferogram")
    with rio.open(file_prm) as ds_prm:
        prm = ds_prm.read(1)
        prof = ds_prm.profile
    with rio.open(file_sec) as ds_sec:
        sec = ds_sec.read(1)
    ifg = prm * sec.conj()

    warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
    with rio.open(file_out, "w", **prof) as dst:
        dst.write(ifg, 1)


# TODO optional chunk processing
def amplitude(file_in, file_out):
    """Compute the amplitude of a complex-valued image.

    Args:
        file_in (str): GeoTiff file of the primary SLC image
        file_out (str): output file
    """
    log.info("Computing amplitude")
    with rio.open(file_in) as ds_prm:
        prm = ds_prm.read(1)
        prof = ds_prm.profile
    amp = np.abs(prm)

    warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
    prof.update({"dtype": amp.dtype.name})
    with rio.open(file_out, "w", **prof) as dst:
        dst.write(amp, 1)


# TODO optional chunk processing
def coherence(file_prm, file_sec, file_out, box_size=5, magnitude=True):
    """Compute the complex coherence from two SLC image files.

    Args:
        file_prm (str): GeoTiff file of the primary SLC image
        file_sec (str): GeoTiff file of the secondary SLC image
        file_out (str): output file
        box_size (int, optional): Window size in pixels for boxcar filtering. Defaults to 5.
        magnitude (bool, optional): Writes magnitude only. Otherwise a complex valued raster is written. Defaults to True.
    """
    log.info("Computing coherence")

    if isinstance(box_size, list):
        box_az = box_size[0]
        box_rg = box_size[1]
    else:
        box_az = box_size
        box_rg = box_size

    def avg_ampl(arr, box_az, box_rg):
        return np.sqrt(boxcar((arr * arr.conj()).real, box_az, box_rg))

    with rio.open(file_prm) as ds_prm:
        prm = ds_prm.read(1)
        prof = ds_prm.profile
    with rio.open(file_sec) as ds_sec:
        sec = ds_sec.read(1)

    coh = np.full_like(prm, np.nan + 1j * np.nan, dtype=prm.dtype)
    log.info("coherence: cross-correlation")
    coh = prm * sec.conj()
    coh = boxcar(coh, box_az, box_rg)
    with np.errstate(divide="ignore", invalid="ignore"):
        log.info("coherence: normalizing")
        coh /= avg_ampl(prm, box_az, box_rg)
        coh /= avg_ampl(sec, box_az, box_rg)

    if magnitude:
        coh = np.abs(coh)

    warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
    prof.update(dict(dtype=coh.dtype.name))
    with rio.open(file_out, "w", **prof) as dst:
        dst.write(coh, 1)


def coh_dask(file_prm, file_sec, file_out, box_size=5, magnitude=True):
    """Compute the complex coherence from two SLC image files.

    Args:
        file_prm (str): GeoTiff file of the primary SLC image
        file_sec (str): GeoTiff file of the secondary SLC image
        file_out (str): output file
        box_size (int, optional): Window size in pixels for boxcar filtering. Defaults to 5.
        magnitude (bool, optional): Writes magnitude only. Otherwise a complex valued raster is written. Defaults to True.
    """

    log.info("Computing coherence")

    if isinstance(box_size, list):
        box_az = box_size[0]
        box_rg = box_size[1]
    else:
        box_az = box_size
        box_rg = box_size

    ds_prm = xr.open_dataset(file_prm, lock=False, chunks="auto", engine="rasterio")
    ds_sec = xr.open_dataset(file_sec, lock=False, chunks="auto", engine="rasterio")

    # accessing dask arrays
    prm = ds_prm["band_data"][0].data
    sec = ds_sec["band_data"][0].data

    process_args = dict(
        dimaz=box_az,
        dimrg=box_rg,
        depth=(box_az, box_rg),
    )

    coh = da.map_overlap(boxcar, prm * sec.conj(), **process_args, dtype="complex64")

    np.seterr(divide="ignore", invalid="ignore")
    coh /= np.sqrt(
        da.map_overlap(boxcar, (prm * prm.conj()).real, **process_args, dtype="float32")
    )
    coh /= np.sqrt(
        da.map_overlap(boxcar, (sec * sec.conj()).real, **process_args, dtype="float32")
    )

    if magnitude:
        coh = np.abs(coh)
        nodataval = np.nan
    else:
        nodataval = np.nan + 1j * np.nan

    da_coh = xr.DataArray(
        data=coh[None],
        dims=("band", "y", "x"),
    )
    da_coh.rio.write_nodata(nodataval)

    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    da_coh.rio.to_raster(file_out)


# Auxiliary functions which are not supposed to be used outside of the processor
def _process_bursts(
    prm,
    sec,
    tmp_prm,
    tmp_sec,
    dir_out,
    dir_dem,
    naz,
    nrg,
    min_burst,
    max_burst,
    dem_upsampling,
    dem_buffer_arc_sec,
    dem_force_download,
    order,
):
    luts = []
    prof_tmp = dict(
        width=nrg,
        height=naz,
        count=1,
        dtype="complex64",
        driver="GTiff",
        nodata=np.nan,
    )
    warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
    # process individual bursts
    with rio.open(tmp_prm, "w", **prof_tmp) as ds_prm:
        with rio.open(tmp_sec, "w", **prof_tmp) as ds_sec:

            for burst_idx in range(min_burst, max_burst + 1):
                log.info(f"---- Processing burst {burst_idx} ----")

                # compute geocoding LUTs (lookup tables) for master and slave bursts
                file_dem = prm.fetch_dem_burst(
                    burst_idx,
                    dir_dem,
                    buffer_arc_sec=dem_buffer_arc_sec,
                    force_download=dem_force_download,
                )
                az_p2g, rg_p2g, dem_profile = prm.geocode_burst(
                    file_dem, burst_idx=burst_idx, dem_upsampling=dem_upsampling
                )
                az_s2g, rg_s2g, dem_profile = sec.geocode_burst(
                    file_dem, burst_idx=burst_idx, dem_upsampling=dem_upsampling
                )

                # read primary and secondary burst rasters
                arr_p = prm.read_burst(burst_idx, True)
                arr_s = sec.read_burst(burst_idx, True)

                # deramp secondary
                pdb_s = sec.deramp_burst(burst_idx)
                arr_s_de = arr_s * np.exp(1j * pdb_s)  # .astype(np.complex64)

                # project slave LUT into master grid
                az_s2p, rg_s2p = coregister(arr_p, az_p2g, rg_p2g, az_s2g, rg_s2g)

                # warp raster secondary and deramping phase
                arr_s2p = align(arr_p, arr_s_de, az_s2p, rg_s2p, order=order)
                pdb_s2p = align(arr_p, pdb_s, az_s2p, rg_s2p, order=order)

                # reramp slave
                arr_s2p = arr_s2p * np.exp(-1j * pdb_s2p)  # .astype(np.complex64)

                # compute topographic phases
                rg_p = np.zeros(arr_s.shape[0])[:, None] + np.arange(0, arr_s.shape[1])
                pht_p = prm.phi_topo(rg_p).reshape(*arr_p.shape)
                pht_s = sec.phi_topo(rg_s2p.ravel()).reshape(*arr_p.shape)
                pha_topo = np.exp(-1j * (pht_p - pht_s)).astype(np.complex64)

                lut_da = _make_da_from_dem(np.stack((az_p2g, rg_p2g)), dem_profile)
                lut_da.rio.to_raster(f"{dir_out}/lut_{burst_idx}.tif")
                luts.append(f"{dir_out}/lut_{burst_idx}.tif")

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


def _apply_fast_esd(
    tmp_prm_file, tmp_sec_file, min_burst, max_burst, naz, nrg, overlap
):
    """Applies an in-place phase correction to burst (complex) interferograms to mitigate phase jumps between the bursts.
    Based on ideas introduced in:
    Qin, Y.; Perissin, D.; Bai, J. A Common “Stripmap-Like” Interferometric Processing Chain for TOPS and ScanSAR Wide Swath Mode. Remote Sens. 2018, 10, 1504.
    """
    x = np.arange(naz)
    xdown, xup = overlap / 2, naz - 1 - overlap / 2

    def make_ramp(phase_diffs, idx):
        if idx == 0:
            ydown, yup = -phase_diffs[idx] / 2, phase_diffs[idx] / 2
        elif idx == len(phase_diffs):
            ydown, yup = -phase_diffs[idx - 1] / 2, phase_diffs[idx - 1] / 2
        else:
            ydown, yup = -phase_diffs[idx - 1] / 2, phase_diffs[idx] / 2
        slope = (yup - ydown) / (xup - xdown)
        off = ydown - slope * xdown
        ramp = slope * x + off
        return np.exp(1j * (ramp[:, None] + np.zeros((nrg))))

    with rio.open(tmp_prm_file, "r") as ds_prm:
        with rio.open(tmp_sec_file, "r+") as ds_sec:
            # computing cross interferograms in overlapping areas
            log.info("Fast ESD: computing cross interferograms")
            phase_diffs = []
            for burst_idx in range(min_burst, max_burst):
                first_line_tail = (burst_idx - min_burst + 1) * naz - overlap
                first_line_head = (burst_idx - min_burst + 1) * naz
                # read last lines of current burst
                tail_p = ds_prm.read(
                    indexes=1, window=Window(0, first_line_tail, nrg, overlap)
                )
                tail_s = ds_sec.read(
                    indexes=1,
                    window=Window(0, first_line_tail, nrg, overlap),
                )
                # read first lines of next burst
                head_p = ds_prm.read(
                    indexes=1, window=Window(0, first_line_head, nrg, overlap)
                )
                head_s = ds_sec.read(
                    indexes=1,
                    window=Window(0, first_line_head, nrg, overlap),
                )
                ifg1 = tail_p * tail_s.conj()
                ifg2 = head_p * head_s.conj()
                cross_ifg = ifg1 * ifg2.conj()
                dphi_clx = cross_ifg[~np.isnan(cross_ifg)]
                phase_diffs.append(np.angle(dphi_clx.mean()))

            # making phase ramps and applying to secondary
            log.info("Fast ESD: applying phase corrections")
            for burst_idx in range(min_burst, max_burst + 1):
                first_line = (burst_idx - min_burst) * naz
                arr_s = ds_sec.read(
                    indexes=1,
                    window=Window(0, first_line, nrg, naz),
                )
                esd_ramp = make_ramp(phase_diffs, burst_idx - min_burst).astype(
                    np.complex64
                )
                ds_sec.write(
                    arr_s * esd_ramp,
                    indexes=1,
                    window=Window(0, first_line, nrg, naz),
                )


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

        prof = src.profile.copy()
        prof.update(dict(width=nrg, height=siz))
        with rio.open(file_out, "w", **prof) as dst:

            log.info("Stitching bursts to make a continuous image")
            off_dst = 0
            for i in range(burst_count):
                if i == 0:
                    nlines = naz - H
                    off_src = 0
                elif i == burst_count - 1:
                    nlines = naz - H
                    off_src = H
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
                    dst.write(
                        arr, window=Window(0, off_dst, nrg, nlines), indexes=j + 1
                    )
                off_dst += nlines


def _make_da_from_dem(arr, dem_prof):

    da = xr.DataArray(
        data=arr,
        dims=("band", "y", "x"),
    )
    da.rio.write_crs(dem_prof["crs"], inplace=True)
    da.rio.write_transform(dem_prof["transform"], inplace=True)
    da.attrs["_FillValue"] = np.nan
    return da


def _merge_luts(files_lut, file_out, lines_per_burst, overlap, offset=4):
    log.info("Merging LUT")
    off = 0
    H = int(overlap / 2)
    naz = lines_per_burst
    to_merge = []
    for i, file_lut in enumerate(files_lut):
        lut = xr.open_dataset(file_lut, engine="rasterio", cache=False)["band_data"]
        cnd = (lut[0] >= H - offset) & (lut[0] < naz - H + offset)
        lut = lut.where(xr.broadcast(cnd, lut)[0], np.nan)

        if i == 0:
            off2 = off
        else:
            off2 = off - H
        lut[0] += off2
        if i == 0:
            off += naz - H
        else:
            off += naz - 2 * H
        lut.attrs["_FillValue"] = np.nan
        lut.rio.write_nodata(np.nan)
        to_merge.append(lut)

    merged = merge_arrays(to_merge, parse_coordinates=False)
    merged.rio.to_raster(file_out, windowed=True)
