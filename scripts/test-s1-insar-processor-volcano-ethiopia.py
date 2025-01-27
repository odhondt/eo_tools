# %%
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("numexpr").setLevel(logging.WARNING)

# Uncomment the next block to test conda imports

# import sys
# sys.path.remove("/eo_tools")
# sys.path.append("/eo_tools") # workaround to include eo_tools_dev
# import eo_tools
# print(f"EO-Tools imported from:")
# print(f"{eo_tools.__file__=}")

import os
import geopandas as gpd

# from eodag import EODataAccessGateway
from eo_tools.S1.process import process_insar
from eo_tools.auxils import remove
from eo_tools_dev.util import show_cog, serve_map, palette_phi
from math import pi
import folium
from folium import LayerControl

# credentials need to be stored in the following file (see EODAG docs)
# confpath = "/data/eodag_config.yml"
# dag = EODataAccessGateway(user_conf_file_path=confpath)
# make sure cop_dataspace will be used
# dag.set_preferred_provider("cop_dataspace")
# dag.set_preferred_provider("geodes")

# %%
data_dir = "/data/S1"

ids = [
    "S1A_IW_SLC__1SDV_20241229T030940_20241229T031008_057201_0708EF_00A3.SAFE",
    "S1A_IW_SLC__1SDV_20250110T030939_20250110T031007_057376_070FDB_456B.SAFE",
]
# primary_dir = f"{data_dir}/{ids[0]}.zip"
primary_dir = f"{data_dir}/{ids[0]}"
# secondary_dir = f"{data_dir}/{ids[1]}.zip"
secondary_dir = f"{data_dir}/{ids[1]}"
output_dir = "/data/res/volcano-fentale-ethiopia"

# %%
# load a geometry
# file_aoi = "/eo_tools/data/Morocco_small.geojson"
# file_aoi = "/eo_tools/data/Morocco_tiny.geojson"
# file_aoi = "/eo_tools/data/Morocco_AOI.geojson"
import shapely.wkt

shp = shapely.wkt.loads(
    "POLYGON ((40.112 9.154999999999999, 39.748 9.154999999999999, 39.748 8.795, 40.112 8.795, 40.112 9.154999999999999))"
)
search_criteria = {
    "productType": "S1_SAR_SLC",
    "start": "2024-12-01",
    "end": "2025-01-15",
    "geom": shp,
    "provider": "geodes",
    # "provider": "cop_dataspace"
}

# uncomment if files are not already on disk
# results = dag.search(**search_criteria)
# results, _ = dag.search(**search_criteria)
# print(results)
# to_dl = [it for it in results if it.properties["id"] in ids]

# print(to_dl)
# dag.download_all(to_dl, output_dir="/data/S1/", extract=False)

# %%
import time

start_time = time.time()
out_dir_prev = f"{output_dir}/S1_InSAR_2023-09-04-063730__2023-09-16-063730"

if os.path.isdir(out_dir_prev):
    remove(out_dir_prev)

process_args = dict(
    dir_prm=primary_dir,
    dir_sec=secondary_dir,
    output_dir=output_dir,
    aoi_name=None,
    # shp=shp,
    pol="vv",
    subswaths=["IW1", "IW2", "IW3"],
    write_coherence=True,
    write_interferogram=True,
    write_primary_amplitude=False,
    write_secondary_amplitude=False,
    apply_fast_esd=True,
    dem_name="cop-dem-glo-30",
    dem_upsampling=1.8,
    dem_force_download=False,
    dem_buffer_arc_sec=40,
    boxcar_coherence=[3, 3],
    filter_ifg=True,
    multilook=[1, 4],
    warp_kernel="bicubic",
    cal_type="beta",
    clip_to_shape=True,
)

out_dir = process_insar(**process_args)

print("Process finished --- %s seconds ---" % (time.time() - start_time))

# %%
from eo_tools.S1.process import (
    apply_to_patterns_for_single,
    goldstein,
    geocode_and_merge_iw,
)

# apply Goldstein filter
# apply_to_patterns_for_single(
#     goldstein,
#     out_dir=f"{out_dir}/sar",
#     file_in_prefix="ifg",
#     file_out_prefix="ifggold",
#     alpha=0.5,
#     overlap=14,
# )

# from eo_tools.S1.process import geocode_and_merge_iw
# from pathlib import Path
# geo_dir = Path(out_dir).parent
geocode_and_merge_iw(out_dir, shp=None, var_names=["ifggold"], clip_to_shape=False)


# %%
# display result
out_dir = f"{output_dir}/S1_InSAR_2024-12-29-030940__2025-01-10-030939"

m = folium.Map()
_ = show_cog(
    f"{out_dir}/phigold_vv.tif", m, rescale=f"{-pi},{pi}", colormap=palette_phi()
)
_ = show_cog(f"{out_dir}/phi_vv.tif", m, rescale=f"{-pi},{pi}", colormap=palette_phi())
_ = show_cog(f"{out_dir}/coh_vv.tif", m, rescale=f"0,1")
LayerControl().add_to(m)

# open in a browser
serve_map(m)
# m
# %%
