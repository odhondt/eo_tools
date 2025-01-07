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

# %%
data_dir = "/data/S1"

ids = [
    "S1A_IW_SLC__1SDV_20230904T063730_20230904T063757_050174_0609E3_DAA1",
    "S1A_IW_SLC__1SDV_20230916T063730_20230916T063757_050349_060FCD_6814",
]
primary_dir = f"{data_dir}/{ids[0]}.zip"
secondary_dir = f"{data_dir}/{ids[1]}.zip"
outputs_prefix = "/data/res/test-full-processor"

# %%
# load a geometry
# file_aoi = "/eo_tools/data/Morocco_small.geojson"
file_aoi = "/eo_tools/data/Morocco_tiny.geojson"
# file_aoi = "/eo_tools/data/Morocco_AOI.geojson"
shp = gpd.read_file(file_aoi).geometry[0]

# search_criteria = {
#     "productType": "S1_SAR_SLC",
#     "start": "2023-09-03",
#     "end": "2023-09-17",
#     "geom": shp,
# }

# uncomment if files are not already on disk
# results, _ = dag.search(**search_criteria)
# to_dl = [it for it in results if it.properties["id"] in ids]
# dag.download_all(to_dl, outputs_prefix="/data/S1/", extract=True)

# %%

out_dir_prev = f"{outputs_prefix}/S1_InSAR_2023-09-04-063730__2023-09-16-063730"

if os.path.isdir(out_dir_prev):
    remove(out_dir_prev)

process_args = dict(
    dir_prm=primary_dir,
    dir_sec=secondary_dir,
    outputs_prefix=outputs_prefix,
    aoi_name=None,
    shp=shp,
    pol="vv",
    # subswaths=["IW1", "IW2"],
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


# %%
# compare with reference data processed with SNAP
out_dir = "/data/res/test-full-processor/S1_InSAR_2023-09-04-063730__2023-09-16-063730"
ref_dir = "/data/reference/S1_InSAR_VV_2023-09-04-063730__2023-09-16-063730_Morocco"

m = folium.Map()
_ = show_cog(f"{ref_dir}/phi.tif", m, rescale=f"{-pi},{pi}", colormap=palette_phi())
_ = show_cog(f"{out_dir}/phi_vv.tif", m, rescale=f"{-pi},{pi}", colormap=palette_phi())
_ = show_cog(f"{ref_dir}/coh.tif", m, rescale=f"0,1")
_ = show_cog(f"{out_dir}/coh_vv.tif", m, rescale=f"0,1")
LayerControl().add_to(m)

# open in a browser
# serve_map(m)
m
# %%
