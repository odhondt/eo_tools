# %%
import logging
logging.basicConfig(level=logging.INFO)
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("numexpr").setLevel(logging.WARNING)

# Uncomment the next block to test conda imports

# import sys
# sys.path.remove("/eo_tools")
# sys.path.append("/eo_tools") # workaround to include eo_tools_dev
# import eo_tools
# print(f"EO-Tools imported from:")
# print(f"{eo_tools.__file__=}")


import folium
from folium import LayerControl

from eo_tools_dev.util import show_cog, serve_map
from eo_tools.S1.process import process_slc
import geopandas as gpd
from eodag import EODataAccessGateway

# credentials need to be stored in the following file (see EODAG docs)
confpath = "/data/eodag_config.yml"
dag = EODataAccessGateway(user_conf_file_path=confpath)
# make sure cop_dataspace will be used 
dag.set_preferred_provider("cop_dataspace")
logging.basicConfig(level=logging.INFO)

# %%
# change to your custom locations
data_dir = "/data/S1"

ids = [
 "S1A_IW_SLC__1SDV_20230904T063730_20230904T063757_050174_0609E3_DAA1", 
]
slc_dir = f"{data_dir}/{ids[0]}.SAFE"
outputs_prefix="/data/res/test-slc-processor-sigma"
outputs_prefix_2="/data/res/test-slc-processor-beta"

# %%
# load a geometry
file_aoi = "/eo_tools/data/Morocco_AOI.geojson"
# file_aoi = "/eo_tools/data/Morocco_tiny.geojson"
shp = gpd.read_file(file_aoi).geometry[0]

search_criteria = {
    "productType": "S1_SAR_SLC",
    "start": "2023-09-03",
    "end": "2023-09-17",
    "geom": shp
}

# results, _ = dag.search(**search_criteria)
# to_dl = [it for it in results if it.properties["id"] in ids]
# print(f"{len(to_dl)} products to download")
# dag.download_all(to_dl, outputs_prefix="/data/S1/", extract=True)


# %%

out_dir_sigma = process_slc(
    dir_slc=slc_dir,
    outputs_prefix=outputs_prefix,
    aoi_name=None,
    shp=shp,
    pol="full",
    # pol="vv",
    subswaths=["IW1", "IW2", "IW3"],
    dem_upsampling=1.8,
    dem_force_download=False,
    dem_buffer_arc_sec=40,
    multilook=[1, 4],
    warp_kernel="bicubic",
    cal_type="sigma",
    # clip_to_shape=True,
)

# %%

out_dir_beta = process_slc(
    dir_slc=slc_dir,
    outputs_prefix=outputs_prefix_2,
    aoi_name=None,
    shp=shp,
    pol="full",
    # pol="vv",
    subswaths=["IW1", "IW2", "IW3"],
    dem_upsampling=1.8,
    dem_force_download=False,
    dem_buffer_arc_sec=40,
    multilook=[1, 4],
    warp_kernel="bicubic",
    cal_type="beta",
    # clip_to_shape=True,
)

# %%

m = folium.Map()
_ = show_cog(f"{out_dir_beta}/amp_vv.tif", m, rescale=f"0, 1")
_ = show_cog(f"{out_dir_sigma}/amp_vv.tif", m, rescale=f"0, 1")
_ = show_cog(f"{out_dir_beta}/amp_vh.tif", m, rescale=f"0, 1")
_ = show_cog(f"{out_dir_sigma}/amp_vh.tif", m, rescale=f"0, 1")
LayerControl().add_to(m)

# open in a browser
serve_map(m)
