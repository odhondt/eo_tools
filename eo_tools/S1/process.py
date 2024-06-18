from eo_tools.S1.core import S1IWSwath, coregister, align, stitch_bursts

import numpy as np
import xarray as xr
import rasterio as rio
from rasterio.windows import Window
import rioxarray as riox
from rioxarray.merge import merge_arrays
import warnings
import os
from eo_tools.S1.util import presum, boxcar, remap
from eo_tools.auxils import get_burst_geometry
import concurrent
import multiprocessing as mp
import calendar
import dask.array as da
from rasterio.errors import NotGeoreferencedWarning
import logging
from pyroSAR import identify
from typing import Union, List, Tuple
from datetime import datetime

log = logging.getLogger(__name__)
# mp.set_start_method("fork")


def prepare_InSAR(
    dir_mst: str,
    dir_slv: str,
    outputs_prefix: str,
    aoi_name: str = None,
    shp=None,
    pol: Union[str, List[str]] = "full",
    apply_ESD: bool = False,
    subswaths: List[str] = ["IW1", "IW2", "IW3"],
    dem_force_download: bool = False,
):
    """Produce a coregistered pair of Single Look Complex images and associated lookup tables."""

    if aoi_name is None:
        aoi_substr = ""
    else:
        aoi_substr = f"_{aoi_name}"

    # retrieve burst geometries
    gdf_burst_mst = get_burst_geometry(
        dir_mst, target_subswaths=["IW1", "IW2", "IW3"], polarization="VV"
    )
    gdf_burst_slv = get_burst_geometry(
        dir_slv, target_subswaths=["IW1", "IW2", "IW3"], polarization="VV"
    )

    # find what subswaths and bursts intersect AOI
    if shp is not None:
        gdf_burst_mst = gdf_burst_mst[gdf_burst_mst.intersects(shp)]
        gdf_burst_slv = gdf_burst_slv[gdf_burst_slv.intersects(shp)]

    # identify corresponding subswaths
    sel_subsw_mst = gdf_burst_mst["subswath"]
    sel_subsw_slv = gdf_burst_slv["subswath"]
    unique_subswaths = np.unique(np.concatenate((sel_subsw_mst, sel_subsw_slv)))
    unique_subswaths = [it for it in unique_subswaths if it in subswaths]

    # check that polarization is correct
    info_mst = identify(dir_mst)
    if isinstance(pol, str):
        if pol == "full":
            pol = info_mst.polarizations
        else:
            if pol.upper() in info_mst.polarizations:
                pol = [pol]
            else:
                raise RuntimeError(
                    f"polarization {pol} does not exists in the source product"
                )
    elif isinstance(pol, list):
        pol = [x for x in pol if x in info_mst.polarizations]
    else:
        raise RuntimeError("polarizations must be of type str or list")

    # do a check on orbits
    info_slv = identify(dir_slv)
    meta_mst = info_mst.scanMetadata()
    meta_slv = info_slv.scanMetadata()
    orbnum = meta_mst["orbitNumber_rel"]
    if meta_slv["orbitNumber_rel"] != orbnum:
        raise ValueError("Images must be from the same relative orbit.")

    # parse dates
    datestr_mst = meta_mst["start"]
    datestr_slv = meta_slv["start"]
    date_mst = datetime.strptime(datestr_mst, "%Y%m%dT%H%M%S")
    date_slv = datetime.strptime(datestr_slv, "%Y%m%dT%H%M%S")

    id_mst = date_mst.strftime("%Y-%m-%d-%H%M%S")
    id_slv = date_slv.strftime("%Y-%m-%d-%H%M%S")

    out_dir = (
        f"{outputs_prefix}/S1_InSAR_{id_mst}_{id_slv}{aoi_substr}"
    )
    for p in pol:
        for subswath in unique_subswaths:
            log.info(f"---- Processing subswath {subswath} in {p.upper()} polarization")

            # identify bursts to process
            bursts_mst = gdf_burst_mst[gdf_burst_mst["subswath"] == subswath][
                "burst"
            ].values
            burst_mst_min = bursts_mst.min()
            burst_mst_max = bursts_mst.max()

            iw = int(subswath[2])
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            preprocess_insar_iw(
                dir_mst,
                dir_slv,
                out_dir,
                iw=iw,
                pol=p.lower(),
                min_burst=burst_mst_min,
                max_burst=burst_mst_max,
                dir_dem="/tmp",
                apply_fast_esd=apply_ESD,
                warp_kernel="bicubic",
                dem_upsampling=1.8,
                dem_buffer_arc_sec=40,
                dem_force_download=dem_force_download,
            )
            os.rename(f"{out_dir}/primary.tif", f"{out_dir}/{p.lower()}_iw{iw}_prm.tif")
            os.rename(f"{out_dir}/secondary.tif", f"{out_dir}/{p.lower()}_iw{iw}_sec.tif")
            os.rename(f"{out_dir}/lut.tif", f"{out_dir}/{p.lower()}_iw{iw}_lut.tif")


# TODO add parameters:
# - slc -- radar geometry only
# - geocoded
# - box size
# - multilook
# - better options: write_phase, write_amp_prm, write_coh, write_clx_coh, etc
# - warp kernel, and pre-processing options from other function
def process_InSAR(
    dir_mst: str,
    dir_slv: str,
    outputs_prefix: str,
    dir_tmp: str,
    aoi_name: str = None,
    shp=None,
    pol: Union[str, List[str]] = "full",
    coh_only: bool = False,
    intensity: bool = True,
    clear_tmp_files: bool = True,
    resume: bool = False,
    apply_ESD: bool = False,
    subswaths: List[str] = ["IW1", "IW2", "IW3"],
    dem_force_download: bool = False,
    boxcar_coherence: Union[int, List[int]] = [3, 10],
    multilook: List[int] = [1, 4],
):
    # TODO: update docstrings when finished
    """Performs InSAR processing of a pair of SLC Sentinel-1 products, geocode the outputs and writes them as COG (Cloud Optimized GeoTiFF) files.
    AOI crop is optional.

    Args:
        file_mst (str): Master image (SLC Sentinel-1 product). Can be a zip file or a folder containing the product.
        file_slv (str): Slave image (SLC Sentinel-1 product). Can be a zip file or a folder containing the product.
        out_dir (str): Output directory
        tmp_dir (str): Temporary directory to store intermediate files
        aoi_name (str): Optional suffix to describe AOI / experiment
        shp (object, optional): Shapely geometry describing an area of interest as a polygon. If set to None, the whole product is processed. Defaults to None.
        pol (str, optional): Polarimetric channels to process (Either 'VH','VV, 'full' or a list like ['HV', 'VV']). Defaults to "full".
        coh_only (bool, optional): Computes only the InSAR coherence and not the phase. Defaults to False.
        intensity (bool, optional): Adds image intensities. Defaults to True.
        clear_tmp_files (bool, optional): Removes temporary files at the end (recommended). Defaults to True.
        erosion_width (int, optional): Size of the morphological erosion to clean image edges after SNAP geocoding. Defaults to 15.
        resume (bool, optional): Allows to resume the processing when interrupted (use carefully). Defaults to False.
        apply_ESD (bool, optional): enhanced spectral diversity to correct phase jumps between bursts. Defaults to False
        subswaths (list, optional): limit the processing to a list of subswaths like `["IW1", "IW2"]`. Defaults to `["IW1", "IW2", "IW3"]`
    Returns:
        out_dirs (list): Output directories containing COG files.
    Note:
        With products from Copernicus Data Space, processing of some zipped products may lead to errors. This issue can be temporarily fixed by processing the unzipped product instead of the zip file.
    """
    # detailed debug info
    # logging.basicConfig(level=logging.DEBUG)

    if aoi_name is None:
        aoi_substr = ""
    else:
        aoi_substr = f"_{aoi_name}"

    # retrieve burst geometries
    gdf_burst_mst = get_burst_geometry(
        dir_mst, target_subswaths=["IW1", "IW2", "IW3"], polarization="VV"
    )
    gdf_burst_slv = get_burst_geometry(
        dir_slv, target_subswaths=["IW1", "IW2", "IW3"], polarization="VV"
    )

    # find what subswaths and bursts intersect AOI
    if shp is not None:
        gdf_burst_mst = gdf_burst_mst[gdf_burst_mst.intersects(shp)]
        gdf_burst_slv = gdf_burst_slv[gdf_burst_slv.intersects(shp)]

    # identify corresponding subswaths
    sel_subsw_mst = gdf_burst_mst["subswath"]
    sel_subsw_slv = gdf_burst_slv["subswath"]
    unique_subswaths = np.unique(np.concatenate((sel_subsw_mst, sel_subsw_slv)))
    unique_subswaths = [it for it in unique_subswaths if it in subswaths]

    # check that polarization is correct
    info_mst = identify(dir_mst)
    if isinstance(pol, str):
        if pol == "full":
            pol = info_mst.polarizations
        else:
            if pol.upper() in info_mst.polarizations:
                pol = [pol]
            else:
                raise RuntimeError(
                    f"polarization {pol} does not exists in the source product"
                )
    elif isinstance(pol, list):
        pol = [x for x in pol if x in info_mst.polarizations]
    else:
        raise RuntimeError("polarizations must be of type str or list")

    # do a check on orbits
    info_slv = identify(dir_slv)
    meta_mst = info_mst.scanMetadata()
    meta_slv = info_slv.scanMetadata()
    orbnum = meta_mst["orbitNumber_rel"]
    if meta_slv["orbitNumber_rel"] != orbnum:
        raise ValueError("Images must be from the same relative orbit.")

    # parse dates
    datestr_mst = meta_mst["start"]
    datestr_slv = meta_slv["start"]
    date_mst = datetime.strptime(datestr_mst, "%Y%m%dT%H%M%S")
    date_slv = datetime.strptime(datestr_slv, "%Y%m%dT%H%M%S")

    id_mst = date_mst.strftime("%Y-%m-%d-%H%M%S")
    id_slv = date_slv.strftime("%Y-%m-%d-%H%M%S")

    out_dirs = []
    for p in pol:
        tmp_names = []
        phi_to_merge = []
        coh_to_merge = []
        amp_prm_to_merge = []
        amp_sec_to_merge = []
        for subswath in unique_subswaths:
            log.info(f"---- Processing subswath {subswath} in {p.upper()} polarization")

            # identify bursts to process
            bursts_mst = gdf_burst_mst[gdf_burst_mst["subswath"] == subswath][
                "burst"
            ].values
            burst_mst_min = bursts_mst.min()
            burst_mst_max = bursts_mst.max()

            tmp_subdir = (
                # f"{dir_tmp}/{subswath}_{p.upper()}_{id_mst}_{id_slv}{aoi_substr}"
                f"{dir_tmp}/{p.upper()}_{id_mst}_{id_slv}{aoi_substr}/{subswath}"
            )
            tmp_names.append(tmp_subdir)
            iw = int(subswath[2])
            if not os.path.isdir(tmp_subdir):
                os.mkdir(tmp_subdir)
            preprocess_insar_iw(
                dir_mst,
                dir_slv,
                tmp_subdir,
                iw=iw,
                pol=p.lower(),
                min_burst=burst_mst_min,
                max_burst=burst_mst_max,
                dir_dem="/tmp",
                apply_fast_esd=apply_ESD,
                warp_kernel="bicubic",
                dem_upsampling=1.8,
                dem_buffer_arc_sec=40,
                dem_force_download=dem_force_download,
            )

            file_prm = f"{tmp_subdir}/primary.tif"
            file_sec = f"{tmp_subdir}/secondary.tif"
            file_amp_1 = f"{tmp_subdir}/amp_prm.tif"
            file_amp_2 = f"{tmp_subdir}/amp_sec.tif"
            file_coh = f"{tmp_subdir}/coh.tif"
            file_phi_geo = f"{tmp_subdir}/phi_geo.tif"
            file_amp_1_geo = f"{tmp_subdir}/amp_prm_geo.tif"
            file_amp_2_geo = f"{tmp_subdir}/amp_sec_geo.tif"
            file_coh_geo = f"{tmp_subdir}/coh_geo.tif"
            file_lut = f"{tmp_subdir}/lut.tif"

            # TODO isolate in child processes
            # computing amplitude and complex coherence  in the radar geometry
            coherence(file_prm, file_sec, file_coh, boxcar_coherence, False)

            # combined multilooking and geocoding
            # interferometric coherence
            log.info("Geocoding interferometric coherence.")

            _child_process(
                slc2geo,
                (
                    file_coh,
                    file_lut,
                    file_coh_geo,
                    multilook[0],
                    multilook[1],
                    "bicubic",
                    False,
                    True,
                ),
            )

            # coh_to_merge.append(riox.open_rasterio(file_coh_geo))
            coh_to_merge.append(file_coh_geo)

            # interferometric phase
            if not coh_only:
                log.info("Geocoding interferometric phase.")
                _child_process(
                    slc2geo,
                    (
                        file_coh,
                        file_lut,
                        file_phi_geo,
                        multilook[0],
                        multilook[1],
                        "bicubic",
                        True,
                        False,
                    ),
                )
                phi_to_merge.append(file_phi_geo)

            # amplitude of the primary image
            if intensity:
                log.info("Geocoding image amplitudes.")
                _child_process(amplitude, (file_prm, file_amp_1))
                _child_process(amplitude, (file_sec, file_amp_2))
                _child_process(
                    slc2geo,
                    (
                        file_amp_1,
                        file_lut,
                        file_amp_1_geo,
                        multilook[0],
                        multilook[1],
                        "bicubic",
                        False,
                        True,
                    ),
                )
                _child_process(
                    slc2geo,
                    (
                        file_amp_2,
                        file_lut,
                        file_amp_2_geo,
                        multilook[0],
                        multilook[1],
                        "bicubic",
                        False,
                        True,
                    ),
                )
                amp_prm_to_merge.append(file_amp_1_geo)
                amp_sec_to_merge.append(file_amp_2_geo)

        log.info(f"---- Merging and cropping subswaths {unique_subswaths}")
        # create output dir if needed
        out_dir = (
            f"{outputs_prefix}/S1_InSAR_{p.upper()}_{id_mst}__{id_slv}{aoi_substr}"
        )
        out_dirs.append(out_dir)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        # merge & crop
        list_var = [coh_to_merge, phi_to_merge, amp_prm_to_merge, amp_sec_to_merge]
        list_prefix = ["coh", "phi", "amp_prm", "amp_sec"]
        for var_to_merge, prefix in zip(list_var, list_prefix):
            if var_to_merge:
                log.info(f"Merging to file {prefix}_geo.tif")
                if shp:
                    da_to_merge = [riox.open_rasterio(var) for var in var_to_merge]
                    merged = merge_arrays(
                        da_to_merge, parse_coordinates=False
                    ).rio.clip([shp], all_touched=True)
                else:
                    merged = merge_arrays(da_to_merge, parse_coordinates=False)
                file_out = f"{out_dir}/{prefix}_geo.tif"
                merged.rio.to_raster(file_out)
                # del merged

        # if clear_tmp_files:
        #     log.info("---- Removing temporary files.")
        #     for tmp_name in tmp_names:
        #         name_coreg = f"{tmp_dir}/{tmp_name}_coreg"
        #         name_insar = f"{tmp_dir}/{tmp_name}_{substr}"
        #         name_int = f"{tmp_dir}/{tmp_name}_{substr}_int"
        #         dimfiles_to_remove = [name_coreg, name_insar, name_int]
        #         for name in dimfiles_to_remove:
        #             remove(f"{name}.dim")
        #             remove(f"{name}.data")

        #             name_tc = f"{tmp_dir}/{tmp_name}_{substr}_tc"
        #             path_tc = f"{name_tc}.tif"
        #             path_edge = f"{name_tc}_edge.tif"
        #             remove(path_tc)
        #             remove(path_edge)

        #     remove(f"{tmp_dir}/graph_coreg.xml")
        #     remove(f"{tmp_dir}/graph_insar.xml")
        #     remove(f"{tmp_dir}/graph_int.xml")
        #     remove(f"{tmp_dir}/graph_tc.xml")
    return out_dirs


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
    warp_kernel="bicubic",
    dem_upsampling=2,
    dem_buffer_arc_sec=40,
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
        warp_kernel (str, optional): kernel used to align secondary SLC. Possible values are "nearest", "bilinear", "bicubic" and "bicubic6".Defaults to "bilinear".
        dem_upsampling (float, optional): Upsample the DEM, it is recommended to keep the default value. Defaults to 2.
        dem_buffer_arc_sec (float, optional): Increase if the image area is not completely inside the DEM.
        dem_force_download (bool, optional): To reduce execution time, DEM files are stored on disk. Set to True to redownload these files if necessary. Defaults to false.

    Note:
        DEM-assisted coregistration is performed to align the secondary with the master. A lookup table file is written to allow the geocoding images from the radar (single-look) grid to the geographic coordinates of the DEM. Bursts are stitched together to form continuous images. All output files are in the GeoTiff format that can be handled by most GIS softwares and geospatial raster tools such as GDAL and rasterio. Because they are in the SAR geometry, SLC rasters are not georeferenced.
    """

    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)

    if iw not in [1, 2, 3]:
        ValueError("iw must be 1, 2 or 3")

    if pol not in ["vv", "vh"]:
        ValueError("pol must be 'vv' or 'vh'")

    # redundant with burst check
    # may be used in the future for non fully overlapping products
    # info_prm = identify(dir_primary)
    # info_sec = identify(dir_secondary)
    # if info_prm.orbitNumber_rel != info_sec.orbitNumber_rel:
    #     raise ValueError(
    #         "Products should have identical tracks (relative orbit numbers)"
    #     )

    prm = S1IWSwath(dir_primary, iw=iw, pol=pol)
    sec = S1IWSwath(dir_secondary, iw=iw, pol=pol)

    prm_burst_info = prm.meta["product"]["swathTiming"]["burstList"]["burst"]
    sec_burst_info = sec.meta["product"]["swathTiming"]["burstList"]["burst"]

    prm_burst_ids = [bid["burstId"]["#text"] for bid in prm_burst_info]
    sec_burst_ids = [bid["burstId"]["#text"] for bid in sec_burst_info]
    if prm_burst_ids != sec_burst_ids:
        raise NotImplementedError(
            "Products must have identical lists of burst IDs. Please select products with (nearly) identical footprints."
        )

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

    warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
    luts = _child_process(
        _process_bursts,
        (
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
            warp_kernel,
        ),
    )
    # with concurrent.futures.ProcessPoolExecutor(max_workers=1) as e:
    #     args = (
    #         prm,
    #         sec,
    #         tmp_prm,
    #         tmp_sec,
    #         dir_out,
    #         dir_dem,
    #         naz,
    #         nrg,
    #         min_burst,
    #         max_burst_,
    #         dem_upsampling,
    #         dem_buffer_arc_sec,
    #         dem_force_download,
    #         warp_kernel,
    #     )
    #     luts = e.submit(_process_bursts, *args).result()
    if (max_burst_ > min_burst) & apply_fast_esd:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as e:
            args = (
                tmp_prm,
                tmp_sec,
                min_burst,
                max_burst_,
                prm.lines_per_burst,
                nrg,
                overlap,
            )
            e.submit(_apply_fast_esd, *args).result()

    if max_burst_ > min_burst:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as e:
            args = (
                tmp_sec,
                f"{dir_out}/secondary.tif",
                prm.lines_per_burst,
                max_burst_ - min_burst + 1,
                overlap,
            )
            e.submit(_stitch_bursts, *args).result()

        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as e:
            args = (
                tmp_prm,
                f"{dir_out}/primary.tif",
                prm.lines_per_burst,
                max_burst_ - min_burst + 1,
                overlap,
            )
            e.submit(_stitch_bursts, *args).result()

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as e:
        args = (luts, f"{dir_out}/lut.tif", prm.lines_per_burst, overlap, 4)
        e.submit(_merge_luts, *args).result()

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
    kernel="bicubic",
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
        kernel (str): kernel used to align secondary SLC. Possible values are "nearest", "bilinear", "bicubic" and "bicubic6".Defaults to "bilinear".
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

    if (mlt_az == 1) & (mlt_rg == 1):
        arr_ = arr[0].copy()
    else:
        arr_ = presum(arr[0], mlt_az, mlt_rg)

    arr_out = remap(arr_, lut[0] / mlt_az, lut[1] / mlt_rg, kernel)

    prof_dst.update({k: prof_src[k] for k in ["count", "dtype", "nodata"]})
    if write_phase:
        phi = np.angle(arr_out)
        nodata = -9999
        phi[np.isnan(phi)] = nodata
        prof_dst.update(
            {
                "dtype": phi.dtype.name,
                "nodata": nodata,
                "driver": "COG",
                "compress": "deflate",
                "resampling": "nearest",
            }
        )
        # removing incompatible options
        prof_dst.pop("blockysize", None)
        prof_dst.pop("tiled", None)
        prof_dst.pop("interleave", None)
        with rio.open(out_file, "w", **prof_dst) as dst:
            dst.write(phi, 1)
    else:
        if magnitude_only:
            mag = np.abs(arr_out)
            nodata = 0
            mag[np.isnan(mag)] = nodata
            prof_dst.update(
                {
                    "dtype": mag.dtype.name,
                    "nodata": nodata,
                    "driver": "COG",
                    "compress": "deflate",
                    "resampling": "nearest",
                }
            )
            # removing incompatible options
            prof_dst.pop("blockysize", None)
            prof_dst.pop("tiled", None)
            prof_dst.pop("interleave", None)
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

    open_args = dict(lock=False, chunks="auto", engine="rasterio", cache=True)

    # TODO use rioxarray.open_rasterio instead
    ds_prm = xr.open_dataset(file_prm, **open_args)
    ds_sec = xr.open_dataset(file_sec, **open_args)

    # accessing dask arrays
    prm = ds_prm["band_data"][0].data
    sec = ds_sec["band_data"][0].data

    process_args = dict(
        dimaz=box_az,
        dimrg=box_rg,
        depth=(box_az, box_rg),
    )

    coh = da.map_overlap(boxcar, prm * sec.conj(), **process_args, dtype="complex64")

    coh /= np.sqrt(
        da.map_overlap(
            boxcar,
            np.nan_to_num((prm * prm.conj()).real),
            **process_args,
            dtype="float32",
        )
    )
    coh /= np.sqrt(
        da.map_overlap(
            boxcar,
            np.nan_to_num((sec * sec.conj()).real),
            **process_args,
            dtype="float32",
        )
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
    # del da_coh


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
    kernel,
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
                arr_s *= np.exp(1j * pdb_s)

                # project slave LUT into master grid
                az_s2p, rg_s2p = coregister(arr_p, az_p2g, rg_p2g, az_s2g, rg_s2g)

                # warp raster secondary and deramping phase
                arr_s = align(arr_s, az_s2p, rg_s2p, kernel)
                pdb_s = align(pdb_s, az_s2p, rg_s2p, kernel)

                # reramp slave
                arr_s *= np.exp(-1j * pdb_s)

                # compute topographic phases
                rg_p = np.zeros(arr_p.shape[0])[:, None] + np.arange(0, arr_p.shape[1])
                pht_p = prm.phi_topo(rg_p).reshape(*arr_p.shape)
                pht_s = sec.phi_topo(rg_s2p.ravel()).reshape(*arr_p.shape)
                pha_topo = np.exp(-1j * (pht_p - pht_s)).astype(np.complex64)

                lut_da = _make_da_from_dem(np.stack((az_p2g, rg_p2g)), dem_profile)
                lut_da.rio.to_raster(f"{dir_out}/lut_{burst_idx}.tif", Tiled=True)
                luts.append(f"{dir_out}/lut_{burst_idx}.tif")

                arr_s *= pha_topo

                first_line = (burst_idx - min_burst) * prm.lines_per_burst
                ds_prm.write(
                    arr_p, 1, window=Window(0, first_line, nrg, prm.lines_per_burst)
                )
                ds_sec.write(
                    arr_s,
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

    darr = xr.DataArray(
        data=arr,
        dims=("band", "y", "x"),
    )
    darr = darr.chunk(chunks="auto")
    darr.rio.write_crs(dem_prof["crs"], inplace=True)
    darr.rio.write_transform(dem_prof["transform"], inplace=True)
    darr.attrs["_FillValue"] = np.nan
    return darr


def _merge_luts(files_lut, file_out, lines_per_burst, overlap, offset=4):

    log.info("Merging LUT")
    off = 0
    H = int(overlap / 2)
    naz = lines_per_burst
    to_merge = []
    for i, file_lut in enumerate(files_lut):
        # lut = riox.open_rasterio(file_lut, chunks=True, lock=False)
        lut = riox.open_rasterio(file_lut)  # , chunks=True)#, lock=False)
        # lut = riox.open_rasterio(file_lut, cache=False)
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
    merged.rio.to_raster(file_out)  # , windowed=False, tiled=True)


def _child_process(func, args):
    # convenience function to make code prettier
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as e:
        res = e.submit(func, *args).result()
    return res
