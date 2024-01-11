from pyroSAR.snap.auxil import Workflow, gpt, groupbyWorkers
from pyroSAR import identify

import os
from pathlib import Path
import rasterio as rio
import rasterio.shutil
import numpy as np
from rasterio import merge, mask
from rasterio.io import MemoryFile
from scipy.ndimage import binary_erosion
from eo_tools.auxils import get_burst_geometry
from datetime import datetime
import calendar
from dateutil.parser import parser

from .auxils import remove

import logging

log = logging.getLogger(__name__)


def process_InSAR(
    file_mst,
    file_slv,
    outputs_prefix,
    tmp_dir,
    aoi_name=None,
    shp=None,
    pol="full",
    coh_only=False,
    intensity=True,
    clear_tmp_files=True,
    erosion_width=15,
    resume=False
    # apply_ESD=False -- maybe for later
):
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
        file_mst, target_subswaths=["IW1", "IW2", "IW3"], polarization="VV"
    )
    gdf_burst_slv = get_burst_geometry(
        file_slv, target_subswaths=["IW1", "IW2", "IW3"], polarization="VV"
    )

    # find what subswaths and bursts intersect AOI
    if shp is not None:
        gdf_burst_mst = gdf_burst_mst[gdf_burst_mst.intersects(shp)]
        gdf_burst_slv = gdf_burst_slv[gdf_burst_slv.intersects(shp)]

    # identify corresponding subswaths
    sel_subsw_mst = gdf_burst_mst["subswath"]
    sel_subsw_slv = gdf_burst_slv["subswath"]
    unique_subswaths = np.unique(np.concatenate((sel_subsw_mst, sel_subsw_slv)))

    # check that polarization is correct
    info_mst = identify(file_mst)
    if isinstance(pol, str):
        if pol == "full":
            pol = info_mst.polarizations
        else:
            if pol in info_mst.polarizations:
                pol = [pol]
            else:
                raise RuntimeError(
                    "polarization {} does not exists in the source product".format(pol)
                )
    elif isinstance(pol, list):
        pol = [x for x in pol if x in info_mst.polarizations]
    else:
        raise RuntimeError("polarizations must be of type str or list")

    # do a check on orbits
    info_slv = identify(file_slv)
    meta_mst = info_mst.scanMetadata()
    meta_slv = info_slv.scanMetadata()
    orbnum = meta_mst["orbitNumber_rel"]
    if meta_slv["orbitNumber_rel"] != orbnum:
        raise ValueError("Images from two different orbits")

    # parse dates
    datestr_mst = meta_mst["start"]
    datestr_slv = meta_slv["start"]
    date_mst = datetime.strptime(datestr_mst, "%Y%m%dT%H%M%S")
    date_slv = datetime.strptime(datestr_slv, "%Y%m%dT%H%M%S")

    calendar_mst = (
        f"{date_mst.strftime('%d')}{calendar.month_abbr[date_mst.month]}{date_mst.year}"
    )
    calendar_slv = (
        f"{date_slv.strftime('%d')}{calendar.month_abbr[date_slv.month]}{date_slv.year}"
    )

    id_mst = date_mst.strftime("%Y-%m-%d-%H%M%S")
    id_slv = date_slv.strftime("%Y-%m-%d-%H%M%S")

    # check availability of orbit state vector file
    log.info("---- Looking for available orbit files")
    orbit_type = "Sentinel Precise (Auto Download)"
    match = info_mst.getOSV(osvType="POE", returnMatch=True)  # , osvdir=osvPath)
    match2 = info_slv.getOSV(osvType="POE", returnMatch=True)  # , osvdir=osvPath)
    if match is None or match2 is None:
        log.info("-- Precise orbits not available, using restituted")
        info_mst.getOSV(osvType="RES")  # , osvdir=osvPath)
        info_slv.getOSV(osvType="RES")  # , osvdir=osvPath)
        orbit_type = "Sentinel Restituted (Auto Download)"

    if coh_only:
        substr = "coh"
    else:
        substr = "ifg"
    out_dirs = []
    for p in pol:
        tmp_names = []
        for subswath in unique_subswaths:
            log.info(f"---- Processing subswath {subswath} in {p} polarization")

            # identify bursts to process
            bursts_slv = gdf_burst_slv[gdf_burst_slv["subswath"] == subswath][
                "burst"
            ].values
            burst_slv_min = bursts_slv.min()
            burst_slv_max = bursts_slv.max()
            bursts_mst = gdf_burst_mst[gdf_burst_mst["subswath"] == subswath][
                "burst"
            ].values
            burst_mst_min = bursts_mst.min()
            burst_mst_max = bursts_mst.max()

            tmp_name = f"{subswath}_{p}_{id_mst}_{id_slv}{aoi_substr}"
            tmp_names.append(tmp_name)

            name_coreg = f"{tmp_dir}/{tmp_name}_coreg"
            path_coreg = f"{name_coreg}.dim"
            if not (os.path.exists(path_coreg) and resume):
                log.info("-- TOPS coregistration")
                TOPS_coregistration(
                    file_mst=file_mst,
                    file_slv=file_slv,
                    file_out=path_coreg,
                    tmp_dir=tmp_dir,
                    subswath=subswath,
                    pol=p,
                    orbit_type=orbit_type,
                    burst_mst_min=burst_mst_min,
                    burst_mst_max=burst_mst_max,
                    burst_slv_min=burst_slv_min,
                    burst_slv_max=burst_slv_max,
                )

            name_insar = f"{tmp_dir}/{tmp_name}_{substr}"
            path_insar = f"{name_insar}.dim"
            if not (os.path.exists(path_insar) and resume):
                log.info("-- InSAR processing")
                insar_processing(
                    file_in=path_coreg,
                    file_out=path_insar,
                    tmp_dir=tmp_dir,
                    coh_only=coh_only,
                )

            name_int = f"{tmp_dir}/{tmp_name}_{substr}_int"
            path_int = f"{name_int}.dim"
            if intensity:
                if not (os.path.exists(path_int) and resume):
                    log.info("-- Computing & merging intensities")
                    img_files = Path(f"{tmp_dir}/{tmp_name}_coreg.data").glob("*.img")
                    basenames = list(set([f.stem[2:] for f in img_files]))
                    if len(basenames) == 2:
                        name1 = basenames[0]
                        name2 = basenames[1]
                    else:
                        raise ValueError("Intensity: exactly 2 bands needed.")
                    _merge_intensity(
                        file_coreg=path_coreg,
                        file_insar=path_insar,
                        file_out=path_int,
                        tmp_dir=tmp_dir,
                        coh_only=coh_only,
                        coreg_name1=name1,
                        coreg_name2=name2,
                        substr=substr,
                        subswath=subswath,
                        pol=p,
                        calendar_mst=calendar_mst,
                        calendar_slv=calendar_slv,
                    )

            name_tc = f"{tmp_dir}/{tmp_name}_{substr}_tc"
            path_tc = f"{name_tc}.tif"
            if not os.path.exists(path_tc):  # and resume:
                log.info("-- Terrain correction (geocoding)")
                output_complex = not coh_only
                if intensity:
                    file_in_tc = path_int  # f"{tmp_dir}/{tmp_name}_{substr}_int.dim"
                else:
                    file_in_tc = path_insar  # f"{tmp_dir}/{tmp_name}_{substr}.dim"
                geocoding(
                    file_in=file_in_tc,
                    file_out=path_tc,
                    tmp_dir=tmp_dir,
                    output_complex=output_complex,
                )

            log.info(f"-- Removing dark edges after terrain correction")
            # file_to_open = f"{tmp_dir}/{tmp_name}_{substr}_tc"
            path_edge = f"{name_tc}_edge.tif"
            rasterio.shutil.copy(path_tc, path_edge)
            with rio.open(path_edge, "r+") as src:
                prof = src.profile
                prof.update({"driver": "GTiff", "nodata": 0})
                struct = np.ones((erosion_width, erosion_width))
                for i in range(1, prof["count"] + 1):
                    band_src = src.read(i)
                    msk_src = band_src != 0
                    msk_dst = binary_erosion(msk_src, struct)
                    band_dst = band_src * msk_dst
                    src.write(band_dst, i)

        log.info(f"---- Merging and cropping subswaths {unique_subswaths}")
        to_merge = [
            rio.open(f"{tmp_dir}/{tmp_name}_{substr}_tc_edge.tif")
            for tmp_name in tmp_names
        ]
        arr_merge, trans_merge = merge.merge(to_merge)
        with rio.open(path_edge) as src:
            prof = src.profile.copy()
            prof.update(
                {
                    "height": arr_merge.shape[1],
                    "width": arr_merge.shape[2],
                    "transform": trans_merge,
                    "nodata": 0,
                }
            )

        # crop without writing intermediate file
        if shp is not None:
            with MemoryFile() as memfile:
                with memfile.open(**prof) as mem:
                    # Populate the input file with numpy array
                    mem.write(arr_merge)
                    arr_crop, trans_crop = mask.mask(mem, [shp], crop=True)
                    prof_out = mem.profile.copy()
                    prof_out.update(
                        {
                            "transform": trans_crop,
                            "width": arr_crop.shape[2],
                            "height": arr_crop.shape[1],
                            "count": 1,
                        }
                    )
        else:
            prof_out = prof.copy()
            prof_out.update({"count": 1})

        log.info("---- Writing COG files")

        # Using COG driver
        prof_out.update({"driver": "COG", "compress": "deflate"})
        del prof_out["blockysize"]
        del prof_out["tiled"]
        del prof_out["interleave"]

        if not coh_only and intensity:
            cog_substrings = ["phi", "coh", "int_mst", "int_slv"]
            offidx = 2
        elif coh_only and intensity:
            cog_substrings = ["coh", "int_mst", "int_slv"]
            offidx = 0
        elif not coh_only and not intensity:
            cog_substrings = ["phi", "coh"]
            offidx = 2
        elif coh_only and not intensity:
            cog_substrings = ["coh"]
            offidx = 0

        if shp is not None:
            arr_out = arr_crop
        else:
            arr_out = arr_merge

        out_dir = f"{outputs_prefix}/S1_InSAR_{p}_{id_mst}__{id_slv}{aoi_substr}"
        out_dirs.append(out_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        for sub in cog_substrings:
            if sub == "phi":
                out_path = f"{out_dir}/{sub}.tif"
                with rio.open(out_path, "w", **prof_out) as dst:
                    dst.write(np.angle(arr_out[0] + 1j * arr_out[1]), 1)
            if sub == "coh":
                out_path = f"{out_dir}/{sub}.tif"
                with rio.open(out_path, "w", **prof_out) as dst:
                    dst.write(arr_out[offidx], 1)
            if sub == "int_mst":
                out_path = f"{out_dir}/{sub}.tif"
                with rio.open(out_path, "w", **prof_out) as dst:
                    band = arr_out[1 + offidx]
                    dst.update_tags(mean_value=band[band != 0].mean())
                    dst.write(band, 1)
            if sub == "int_slv":
                out_path = f"{out_dir}/{sub}.tif"
                with rio.open(out_path, "w", **prof_out) as dst:
                    band = arr_out[2 + offidx]
                    dst.update_tags(mean_value=band[band != 0].mean())
                    dst.write(band, 1)

        if clear_tmp_files:
            log.info(
                # "clear_tmp_files: This feature will be implemented in a future version."
                "---- Removing temporary files."
            )
            # tmp_name = f"{subswath}_{p}_{id_mst}_{id_slv}{aoi_substr}"
            for tmp_name in tmp_names:
                name_coreg = f"{tmp_dir}/{tmp_name}_coreg"
                name_insar = f"{tmp_dir}/{tmp_name}_{substr}"
                name_int = f"{tmp_dir}/{tmp_name}_{substr}_int"
                dimfiles_to_remove = [name_coreg, name_insar, name_int]
                for name in dimfiles_to_remove:
                    remove(f"{name}.dim")
                    remove(f"{name}.data")
            
                    name_tc = f"{tmp_dir}/{tmp_name}_{substr}_tc"
                    path_tc = f"{name_tc}.tif"
                    path_edge = f"{name_tc}_edge.tif"
                    remove(path_tc)
                    remove(path_edge)

            remove(f"{tmp_dir}/graph_coreg.xml")
            remove(f"{tmp_dir}/graph_insar.xml")
            remove(f"{tmp_dir}/graph_int.xml")
            remove(f"{tmp_dir}/graph_tc.xml")
    return out_dirs


def TOPS_coregistration(
    file_mst,
    file_slv,
    file_out,
    tmp_dir,
    subswath,
    pol,
    orbit_type,
    burst_mst_min=1,
    burst_mst_max=9,
    burst_slv_min=1,
    burst_slv_max=9,
):
    """Helper function to compute TOPS coregistration"""
    graph_coreg_path = "../graph/S1-TOPSAR-Coregistration.xml"
    wfl_coreg = Workflow(graph_coreg_path)
    wfl_coreg["Read"].parameters["file"] = file_mst
    wfl_coreg["Read(2)"].parameters["file"] = file_slv

    wfl_coreg["TOPSAR-Split"].parameters["subswath"] = subswath
    wfl_coreg["TOPSAR-Split(2)"].parameters["subswath"] = subswath

    wfl_coreg["TOPSAR-Split"].parameters["selectedPolarisations"] = pol
    wfl_coreg["TOPSAR-Split(2)"].parameters["selectedPolarisations"] = pol

    wfl_coreg["TOPSAR-Split"].parameters["firstBurstIndex"] = burst_mst_min
    wfl_coreg["TOPSAR-Split"].parameters["lastBurstIndex"] = burst_mst_max

    wfl_coreg["TOPSAR-Split(2)"].parameters["firstBurstIndex"] = burst_slv_min
    wfl_coreg["TOPSAR-Split(2)"].parameters["lastBurstIndex"] = burst_slv_max

    wfl_coreg["Apply-Orbit-File"].parameters["orbitType"] = orbit_type
    wfl_coreg["Apply-Orbit-File(2)"].parameters["orbitType"] = orbit_type

    wfl_coreg["TOPSAR-Deburst"].parameters["selectedPolarisations"] = pol

    wfl_coreg["Write"].parameters["file"] = file_out
    wfl_coreg.write(f"{tmp_dir}/graph_coreg.xml")
    grp = groupbyWorkers(f"{tmp_dir}/graph_coreg.xml", n=1)
    gpt(f"{tmp_dir}/graph_coreg.xml", groups=grp, tmpdir=tmp_dir)


def insar_processing(file_in, file_out, tmp_dir, coh_only=False):
    """Helper function to compute InSAR phase and / or coherence"""
    graph_coh_path = "../graph/S1-TOPSAR-Coherence.xml"
    graph_ifg_path = "../graph/S1-TOPSAR-Interferogram.xml"
    if coh_only:
        wfl_insar = Workflow(graph_coh_path)
    else:
        wfl_insar = Workflow(graph_ifg_path)
    wfl_insar["Read"].parameters["file"] = file_in
    wfl_insar["Write"].parameters["file"] = file_out
    wfl_insar.write(f"{tmp_dir}/graph_insar.xml")
    gpt(f"{tmp_dir}/graph_insar.xml", tmpdir=tmp_dir)


# convenience function
def _merge_intensity(
    file_coreg,
    file_insar,
    file_out,
    tmp_dir,
    coh_only,
    coreg_name1,
    coreg_name2,
    substr,
    subswath,
    pol,
    calendar_mst,
    calendar_slv,
):
    """Helper function to compute intensities of coregistered master and slave and merge it with InSAR outputs"""
    graph_int_path = "../graph/S1-MasterSlaveIntensity.xml"
    wfl_int = Workflow(graph_int_path)
    wfl_int["Read"].parameters["file"] = file_coreg
    wfl_int["Read(2)"].parameters["file"] = file_insar

    # required to avoid merging virtual bands
    if coh_only:
        wfl_int["BandSelect"].parameters["sourceBands"] = [
            f"coh_{subswath}_{pol}_{calendar_mst}_{calendar_slv}"
        ]
    else:
        wfl_int["BandSelect"].parameters["sourceBands"] = [
            f"i_{substr}_{subswath}_{pol}_{calendar_mst}_{calendar_slv}",
            f"q_{substr}_{subswath}_{pol}_{calendar_mst}_{calendar_slv}",
            f"coh_{subswath}_{pol}_{calendar_mst}_{calendar_slv}",
        ]

    math = wfl_int["BandMaths"]
    exp = math.parameters["targetBands"][0]
    exp["name"] = f"Intensity_{coreg_name1}"
    exp["expression"] = f"sq(i_{coreg_name1}) + sq(q_{coreg_name1})"
    math2 = wfl_int["BandMaths(2)"]
    exp2 = math2.parameters["targetBands"][0]
    exp2["name"] = f"Intensity_{coreg_name2}"
    exp2["expression"] = f"sq(i_{coreg_name2}) + sq(q_{coreg_name2})"
    wfl_int["Write"].parameters["file"] = file_out
    wfl_int.write(f"{tmp_dir}/graph_int.xml")
    gpt(f"{tmp_dir}/graph_int.xml", tmpdir=tmp_dir)


def geocoding(file_in, file_out, tmp_dir, output_complex=False):
    """Helper function to geocode the outputs (performs Range Doppler Terrain Correction)"""
    graph_tc_path = "../graph/S1-TOPSAR-RD-TerrainCorrection.xml"
    wfl_tc = Workflow(graph_tc_path)

    wfl_tc["Read"].parameters["file"] = file_in
    wfl_tc["Write"].parameters["file"] = file_out
    if output_complex:
        wfl_tc["Terrain-Correction"].parameters["outputComplex"] = "true"
    else:
        wfl_tc["Terrain-Correction"].parameters["outputComplex"] = "false"
    wfl_tc.write(f"{tmp_dir}/graph_tc.xml")
    grp = groupbyWorkers(f"{tmp_dir}/graph_tc.xml", n=1)
    gpt(f"{tmp_dir}/graph_tc.xml", groups=grp, tmpdir=tmp_dir)
