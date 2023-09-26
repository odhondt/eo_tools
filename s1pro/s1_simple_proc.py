from pyroSAR.snap.auxil import Workflow, gpt, groupbyWorkers
from pyroSAR import identify

import os
import glob
from pathlib import Path
import rasterio as rio
import numpy as np
from rasterio import merge, mask
from rasterio.io import MemoryFile
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from scipy.ndimage import binary_erosion
from s1pro.auxils import get_burst_geometry
from datetime import datetime
import calendar

from .auxils import remove

import logging

log = logging.getLogger(__name__)


def S1_insar_proc(
    file_mst,
    file_slv,
    out_dir,
    tmp_dir,
    shp=None,
    pol="full",
    coh_only=False,
    intensity=True,
    clear_tmp_files=True,
    erosion_width=15,
    resume=False
    # apply_ESD=False -- maybe for later
):
    # detailed debug info
    # logging.basicConfig(level=logging.DEBUG)

    # if apply_ESD:
    #     raise NotImplementedError("method not implemented")
    # else:
    graph_int_path = "../graph/MasterSlaveIntensity.xml"

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

    # do a check on orbits and slice
    info_slv = identify(file_slv)
    meta_mst = info_mst.scanMetadata()
    meta_slv = info_slv.scanMetadata()
    slnum = meta_mst["sliceNumber"]
    orbnum = meta_mst["orbitNumber_rel"]
    if meta_slv["sliceNumber"] != slnum:
        raise ValueError("Images from two different slices")
    if meta_slv["orbitNumber_rel"] != orbnum:
        raise ValueError("Images from two different orbits")

    # parse dates
    datestr_mst = meta_mst["start"]
    datestr_slv = meta_slv["start"]
    date_mst = datetime.strptime(datestr_mst, "%Y%m%dT%H%M%S")
    date_slv = datetime.strptime(datestr_slv, "%Y%m%dT%H%M%S")
    calendar_mst = f"{date_mst.day}{calendar.month_abbr[date_mst.month]}{date_mst.year}"
    calendar_slv = f"{date_slv.day}{calendar.month_abbr[date_slv.month]}{date_slv.year}"

    # check availability of orbit state vector file
    orbit_type = "Sentinel Precise (Auto Download)"
    match = info_mst.getOSV(osvType="POE", returnMatch=True)  # , osvdir=osvPath)
    match2 = info_slv.getOSV(osvType="POE", returnMatch=True)  # , osvdir=osvPath)
    if match is None or match2 is None:
        print("Precise orbits not available, using restituted")
        info_mst.getOSV(osvType="RES")  # , osvdir=osvPath)
        info_slv.getOSV(osvType="RES")  # , osvdir=osvPath)
        orbit_type = "Sentinel Restituted (Auto Download)"

    if coh_only:
        substr = "coh"
    else:
        substr = "ifg"

    for p in pol:
        tmp_names = []
        for subswath in unique_subswaths:
            log.info(f"Processing subswath {subswath} in {p} polarization.")

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

            tmp_name = f"{subswath}_{p}_{calendar_mst}_{calendar_slv}_slice_{slnum}"
            tmp_names.append(tmp_name)

            if not os.path.exists(f"{tmp_dir}/{tmp_name}_coreg.dim") and resume:
                log.info("TOPS coregistration")
                TOPS_coregistration(
                    file_mst=file_mst,
                    file_slv=file_slv,
                    file_out=f"{tmp_dir}/{tmp_name}_coreg",
                    tmp_dir=tmp_dir,
                    subswath=subswath,
                    pol=p,
                    orbit_type=orbit_type,
                    burst_mst_min=burst_mst_min,
                    burst_mst_max=burst_mst_max,
                    burst_slv_min=burst_slv_min,
                    burst_slv_max=burst_slv_max,
                )

            # InSAR processing
            if not os.path.exists(f"{tmp_dir}/{tmp_name}_{substr}.dim") and resume:
                log.info("InSAR processing")
                insar_processing(
                    file_in=f"{tmp_dir}/{tmp_name}_coreg.dim",
                    file_out=f"{tmp_dir}/{tmp_name}_{substr}",
                    tmp_dir=tmp_dir,
                    coh_only=coh_only,
                )

            if intensity:
                if (
                    not os.path.exists(f"{tmp_dir}/{tmp_name}_{substr}_int.dim")
                    and resume
                ):
                    log.info("Computing intensities")
                    path_coreg = f"{tmp_dir}/{tmp_name}_coreg.data/"
                    img_files = Path(path_coreg).glob("*.img")
                    basenames = list(set([f.stem[2:] for f in img_files]))
                    if len(basenames) == 2:
                        name1 = basenames[0]
                        name2 = basenames[1]
                    else:
                        raise ValueError("Intensity: exactly 2 bands needed.")
                    wfl_int = Workflow(graph_int_path)
                    wfl_int["Read"].parameters[
                        "file"
                    ] = f"{tmp_dir}/{tmp_name}_coreg.dim"
                    wfl_int["Read(2)"].parameters[
                        "file"
                    ] = f"{tmp_dir}/{tmp_name}_{substr}.dim"

                    # required to avoid merging virtual bands
                    if coh_only:
                        wfl_int["BandSelect"].parameters["sourceBands"] = [
                            f"coh_{subswath}_{p}_{calendar_mst}_{calendar_slv}"
                        ]
                    else:
                        wfl_int["BandSelect"].parameters["sourceBands"] = [
                            f"i_{substr}_{subswath}_{p}_{calendar_mst}_{calendar_slv}",
                            f"q_{substr}_{subswath}_{p}_{calendar_mst}_{calendar_slv}",
                            f"coh_{subswath}_{p}_{calendar_mst}_{calendar_slv}",
                        ]

                    math = wfl_int["BandMaths"]
                    exp = math.parameters["targetBands"][0]
                    exp["name"] = f"Intensity_{name1}"
                    exp["expression"] = f"sq(i_{name1}) + sq(q_{name1})"
                    math2 = wfl_int["BandMaths(2)"]
                    exp2 = math2.parameters["targetBands"][0]
                    exp2["name"] = f"Intensity_{name2}"
                    exp2["expression"] = f"sq(i_{name2}) + sq(q_{name2})"
                    wfl_int["Write"].parameters[
                        "file"
                    ] = f"{tmp_dir}/{tmp_name}_{substr}_int"
                    wfl_int.write(f"{tmp_dir}/graph_int.xml")
                    gpt(f"{tmp_dir}/graph_int.xml", tmpdir=tmp_dir)

            # Terrain correction
            tc_path = f"{tmp_dir}/{tmp_name}_{substr}_tc.tif"
            if not os.path.exists(tc_path) and resume:
                log.info("Terrain correction (geocoding)")
                output_complex = not coh_only
                if intensity:
                    file_in_tc = f"{tmp_dir}/{tmp_name}_{substr}_int.dim"
                else:
                    file_in_tc = f"{tmp_dir}/{tmp_name}_{substr}.dim"
                file_out_tc = f"{tmp_dir}/{tmp_name}_{substr}_tc.tif"
                geocoding(
                    file_in=file_in_tc,
                    file_out=file_out_tc,
                    tmp_dir=tmp_dir,
                    output_complex=output_complex,
                )

            log.info(f"Removing dark edges after terrain correction")
            file_to_open = f"{tmp_dir}/{tmp_name}_{substr}_tc"
            rio.shutil.copy(f"{file_to_open}.tif", f"{file_to_open}_edge.tif")
            with rio.open(f"{file_to_open}_edge.tif", "r+") as src:
                prof = src.profile
                prof.update({"driver": "GTiff", "nodata": 0})
                struct = np.ones((erosion_width, erosion_width))
                for i in range(1, prof["count"] + 1):
                    band_src = src.read(i)
                    msk_src = band_src != 0
                    msk_dst = binary_erosion(msk_src, struct)
                    band_dst = band_src * msk_dst
                    src.write(band_dst, i)

        log.info(f"Merging and cropping subswaths {unique_subswaths}")
        to_merge = [
            rio.open(f"{tmp_dir}/{tmp_name}_{substr}_tc_edge.tif")
            for tmp_name in tmp_names
        ]
        arr_merge, trans_merge = merge.merge(to_merge)
        with rio.open(f"{file_to_open}_edge.tif") as src:
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
                            "count": 1
                            # "photometric": "RGB",  # keeps same setting as tiff from SNAP
                        }
                    )
        else:
            prof_out = prof.copy()
            prof_out.update(
                {
                    "count": 1
                    # "photometric": "RGB",  # keeps same setting as tiff from SNAP
                }
            )

        log.info("write COG files")
        cog_prof = cog_profiles.get("deflate")

        if not coh_only and intensity:
            cog_substrings = ["phi", "coh", "mst_int", "slv_int"]
            offidx = 2
        elif coh_only and intensity:
            cog_substrings = ["coh", "mst_int", "slv_int"]
            offidx = 0
        elif not coh_only and not intensity:
            cog_substrings = ["phi", "coh"]
            offidx = 2
        elif coh_only and not intensity:
            cog_substrings = ["coh"]
            offidx = 0

        if shp is not None:
            arr_out = arr_crop
            postfix = "_crop"
        else:
            arr_out = arr_merge
            postfix = ""

        for sub in cog_substrings:
            if sub == "phi":
                out_name = (
                    f"{sub}_{p}_{calendar_mst}_{calendar_slv}_slice{slnum}{postfix}"
                )
                with rio.open(f"{tmp_dir}/{out_name}.tif", "w", **prof_out) as dst:
                    dst.write(np.angle(arr_out[0] + 1j * arr_out[1]), 1)
                    cog_translate(
                        dst, f"{out_dir}/{out_name}.tif", cog_prof, quiet=True
                    )
            if sub == "coh":
                out_name = (
                    f"{sub}_{p}_{calendar_mst}_{calendar_slv}_slice{slnum}{postfix}"
                )
                with rio.open(f"{tmp_dir}/{out_name}.tif", "w", **prof_out) as dst:
                    dst.write(arr_out[offidx], 1)
                    cog_translate(
                        dst, f"{out_dir}/{out_name}.tif", cog_prof, quiet=True
                    )
            if sub == "mst_int":
                out_name = f"int_{p}_{calendar_mst}_slice{slnum}{postfix}"
                with rio.open(f"{tmp_dir}/{out_name}.tif", "w", **prof_out) as dst:
                    dst.write(arr_out[1 + offidx], 1)
                    cog_translate(
                        dst, f"{out_dir}/{out_name}.tif", cog_prof, quiet=True
                    )
            if sub == "slv_int":
                out_name = f"int_{p}_{calendar_slv}_slice{slnum}{postfix}"
                with rio.open(f"{tmp_dir}/{out_name}.tif", "w", **prof_out) as dst:
                    dst.write(arr_out[2 + offidx], 1)
                    cog_translate(
                        dst, f"{out_dir}/{out_name}.tif", cog_prof, quiet=True
                    )

        if clear_tmp_files:
            os.remove(f"{tmp_dir}/graph_coreg.xml")
            if intensity:
                os.remove(f"{tmp_dir}/graph_int.xml")
            os.remove(f"{tmp_dir}/graph_{substr}.xml")
            os.remove(f"{tmp_dir}/graph_tc.xml")
            files = glob.glob(f"{tmp_dir}/*.data") + glob.glob(f"{tmp_dir}/*.dim")
            for fi in files:
                remove(fi)

            for tmp_name in tmp_names:
                os.remove(f"{tmp_dir}/{tmp_name}_{substr}_tc.tif")
                os.remove(f"{tmp_dir}/{tmp_name}_{substr}_tc_border.tif")


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

    # wfl_coreg["Write"].parameters["file"] = f"{tmp_dir}/{tmp_name}_coreg"
    wfl_coreg["Write"].parameters["file"] = file_out
    wfl_coreg.write(f"{tmp_dir}/graph_coreg.xml")
    grp = groupbyWorkers(f"{tmp_dir}/graph_coreg.xml", n=1)
    gpt(f"{tmp_dir}/graph_coreg.xml", groups=grp, tmpdir=tmp_dir)


def insar_processing(file_in, file_out, tmp_dir, coh_only=False):
    graph_coh_path = "../graph/TOPSAR-Coherence.xml"
    graph_ifg_path = "../graph/TOPSAR-Interferogram.xml"
    if coh_only:
        wfl_insar = Workflow(graph_coh_path)
    else:
        wfl_insar = Workflow(graph_ifg_path)
    wfl_insar["Read"].parameters["file"] = file_in
    wfl_insar["Write"].parameters["file"] = file_out
    wfl_insar.write(f"{tmp_dir}/graph_insar.xml")
    gpt(f"{tmp_dir}/graph_insar.xml", tmpdir=tmp_dir)


def merge_intensity():
    pass


def geocoding(file_in, file_out, tmp_dir, output_complex=False):
    graph_tc_path = "../graph/TOPSAR-RD-TerrainCorrection.xml"
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


# TODO:
# - add some parameters
# - subswaths as a parameter
# - ESD (optional)
# - break into functions to be reused by other processors if possible
# - slice assembly (post-process with rio)
# - goldstein filter (optional)
# - gpt options (?)
