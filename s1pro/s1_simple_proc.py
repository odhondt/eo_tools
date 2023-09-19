from pyroSAR.snap.auxil import Workflow, gpt, groupbyWorkers
from pyroSAR import identify

import os
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


def S1_simple_proc(
    file_mst,
    file_slv,
    out_dir,
    tmp_dir,
    shp=None,
    pol="full",
    clear_tmp_files=True,
    erosion_width=15,
):
    graph_path = "../graph/TOPSAR_coh_geocode_IW_to_geotiff.xml"

    # retrieve burst geometries
    gdf_burst_mst = get_burst_geometry(
        file_mst, target_subswaths=["IW1", "IW2", "IW3"], polarization="VV"
    )
    gdf_burst_slv = get_burst_geometry(
        file_slv, target_subswaths=["IW1", "IW2", "IW3"], polarization="VV"
    )

    # find what subswaths and bursts intersect AOI
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

    info_slv = identify(file_slv)
    meta_mst = info_mst.scanMetadata()
    meta_slv = info_slv.scanMetadata()
    slnum = meta_mst["sliceNumber"]
    orbnum = meta_mst["orbitNumber_rel"]
    if meta_slv["sliceNumber"] != slnum:
        raise ValueError("Images from two different slices")
    if meta_slv["orbitNumber_rel"] != orbnum:
        raise ValueError("Images from two different orbits")
    datestr_mst = meta_mst["start"]
    datestr_slv = meta_slv["start"]
    date_mst = datetime.strptime(datestr_mst, "%Y%m%dT%H%M%S")
    date_slv = datetime.strptime(datestr_slv, "%Y%m%dT%H%M%S")
    calendar_mst = f"{date_mst.day}{calendar.month_abbr[date_mst.month]}{date_mst.year}"
    calendar_slv = f"{date_slv.day}{calendar.month_abbr[date_slv.month]}{date_slv.year}"

    tmp_names = []
    for p in pol:
        for subswath in unique_subswaths:
            # setting graph parameters
            wfl = Workflow(graph_path)
            wfl["Read"].parameters["file"] = file_mst
            wfl["Read(2)"].parameters["file"] = file_slv

            print(f"Processing subswath {subswath} in {p} polarization.")
            wfl["TOPSAR-Split"].parameters["subswath"] = subswath
            wfl["TOPSAR-Split(2)"].parameters["subswath"] = subswath

            wfl["TOPSAR-Split"].parameters["selectedPolarisations"] = p
            wfl["TOPSAR-Split(2)"].parameters["selectedPolarisations"] = p

            bursts_mst = gdf_burst_mst[gdf_burst_mst["subswath"] == subswath][
                "burst"
            ].values
            burst_mst_min = bursts_mst.min()
            burst_mst_max = bursts_mst.max()
            wfl["TOPSAR-Split"].parameters["firstBurstIndex"] = burst_mst_min
            wfl["TOPSAR-Split"].parameters["lastBurstIndex"] = burst_mst_max

            bursts_slv = gdf_burst_slv[gdf_burst_slv["subswath"] == subswath][
                "burst"
            ].values
            burst_slv_min = bursts_slv.min()
            burst_slv_max = bursts_slv.max()
            wfl["TOPSAR-Split(2)"].parameters["firstBurstIndex"] = burst_slv_min
            wfl["TOPSAR-Split(2)"].parameters["lastBurstIndex"] = burst_slv_max
            wfl["TOPSAR-Deburst"].parameters["selectedPolarisations"] = p

            tmp_name = f"coh_{subswath}_{p}_{calendar_mst}_{calendar_slv}_slice_{slnum}"
            tmp_names.append(tmp_name)
            wfl["Write"].parameters["file"] = f"{tmp_dir}/{tmp_name}_geo.tif"
            wfl.write(f"{tmp_dir}/graph_coh.xml")
            grp = groupbyWorkers(f"{tmp_dir}/graph_coh.xml", n=1)
            gpt(f"{tmp_dir}/graph_coh.xml", groups=grp, tmpdir=tmp_dir)

            print(f"Removing dark edges after terrain correction")
            with rio.open(f"{tmp_dir}/{tmp_name}_geo.tif", "r") as src:
                prof = src.profile.copy()
                prof.update({"driver": "GTiff", "nodata": 0})
                struct = np.ones((erosion_width, erosion_width))
                with rio.open(
                    f"{tmp_dir}/{tmp_name}_geo_border.tif", "w", **prof
                ) as dst:
                    for i in range(1, prof["count"] + 1):
                        band_src = src.read(i)
                        msk_src = band_src != 0
                        msk_dst = binary_erosion(msk_src, struct)
                        band_dst = band_src * msk_dst
                        dst.write(band_dst, i)

        print("Merging and cropping current subswath")
        to_merge = [
            rio.open(f"{tmp_dir}/{tmp_name}_geo_border.tif") for tmp_name in tmp_names
        ]
        arr_merge, trans_merge = merge.merge(to_merge)
        with rio.open(f"{tmp_dir}/{tmp_names[0]}_geo_border.tif") as src:
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
                        }
                    )
        else:
            prof_out = prof

        # write as COG
        if shp is not None:
            out_name = f"coh_{p}_{calendar_mst}_{calendar_slv}_slice{slnum}_crop"
        else:
            out_name = f"coh_{p}_{calendar_mst}_{calendar_slv}_slice{slnum}"
        with MemoryFile() as memfile:
            with memfile.open(**prof_out) as mem:
                if shp is not None:
                    mem.write(arr_crop)
                else:
                    mem.write(arr_merge)
                cog_prof = cog_profiles.get("deflate")
                cog_translate(
                    mem,
                    f"{out_dir}/{out_name}.tif",
                    cog_prof,
                    in_memory=True,
                    quiet=True,
                )

        if clear_tmp_files:
            for tmp_name in tmp_names:
                os.remove(f"{tmp_dir}/{tmp_name}_geo.tif")
                os.remove(f"{tmp_dir}/{tmp_name}_geo_border.tif")
                os.remove(f"{tmp_dir}/graph_coh.xml")


# TODO:
# - optional crop
# - remove temp files
# - name files with dates (check if band name is in first geotiff)
# - gpt options (?)
# - subswaths as a parameter
# - interferogram
# - add some parameters
# - precise orbits option
# - ESD (optional)
