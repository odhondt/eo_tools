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
    # Morocco
    #  "S1A_IW_SLC__1SDV_20230904T063730_20230904T063757_050174_0609E3_DAA1",
    # Etna
    "S1A_IW_SLC__1SDV_20181228T050448_20181228T050515_025221_02C9BE_1E22",
]
slc_path = f"{data_dir}/{ids[0]}.zip"
output_dir_1 = "/data/res/test-slc-processor-rtc-nasadem"
output_dir_2 = "/data/res/test-slc-processor-rtc-alos"
output_dir_3 = "/data/res/test-slc-processor-rtc-glo30"
output_dir_4 = "/data/res/test-slc-processor-rtc-glo90"

# %%
# load a geometry
# aoi_file = "/eo_tools/data/Morocco_AOI.geojson"
aoi_file = "/eo_tools/data/Etna.geojson"
# aoi_file = "/eo_tools/data/Morocco_tiny.geojson"
# aoi_file = "/eo_tools/data/Morocco_small.geojson"
shp = gpd.read_file(aoi_file).geometry[0]

search_criteria = {
    "productType": "S1_SAR_SLC",
    "start": "2023-09-03",
    "end": "2023-09-17",
    "geom": shp,
}

# results = dag.search(**search_criteria)
# to_dl = [it for it in results if it.properties["id"] in ids]
# print(f"{len(to_dl)} products to download")
# dag.download_all(to_dl, output_dir="/data/S1/", extract=True)

# %%

out_dir_nasadem = process_slc(
    slc_path=slc_path,
    output_dir=output_dir_1,
    aoi_name=None,
    shp=shp,
    pol="vv",
    subswaths=["IW1", "IW2", "IW3"],
    dem_name="nasadem",
    dem_upsampling=1.8,
    dem_force_download=True,
    dem_buffer_arc_sec=40,
    multilook=[1, 4],
    warp_kernel="bicubic",
    cal_type="terrain",
    clip_to_shape=True,
)
# %%

out_dir_alos = process_slc(
    slc_path=slc_path,
    output_dir=output_dir_2,
    aoi_name=None,
    shp=shp,
    pol="vv",
    subswaths=["IW1", "IW2", "IW3"],
    dem_name="alos-dem",
    dem_upsampling=1.8,
    dem_force_download=True,
    dem_buffer_arc_sec=40,
    multilook=[1, 4],
    warp_kernel="bicubic",
    cal_type="terrain",
    clip_to_shape=True,
)

# %%

out_dir_glo30 = process_slc(
    slc_path=slc_path,
    output_dir=output_dir_3,
    aoi_name=None,
    shp=shp,
    pol="vv",
    subswaths=["IW1", "IW2", "IW3"],
    dem_name="cop-dem-glo-30",
    dem_upsampling=1.8,
    dem_force_download=True,
    dem_buffer_arc_sec=40,
    multilook=[1, 4],
    warp_kernel="bicubic",
    cal_type="terrain",
    clip_to_shape=True,
)

# %%

out_dir_glo90 = process_slc(
    slc_path=slc_path,
    output_dir=output_dir_4,
    aoi_name=None,
    shp=shp,
    pol="vv",
    subswaths=["IW1", "IW2", "IW3"],
    dem_name="cop-dem-glo-90",
    # adapted upsampling to resolution
    dem_upsampling=5.4,
    dem_force_download=True,
    dem_buffer_arc_sec=40,
    multilook=[1, 4],
    warp_kernel="bicubic",
    cal_type="terrain",
    clip_to_shape=True,
)

# %%

m = folium.Map()
_ = show_cog(f"{out_dir_nasadem}/amp_vv.tif", m, rescale=f"0, 1")
_ = show_cog(f"{out_dir_alos}/amp_vv.tif", m, rescale=f"0, 1")
_ = show_cog(f"{out_dir_glo30}/amp_vv.tif", m, rescale=f"0, 1")
_ = show_cog(f"{out_dir_glo90}/amp_vv.tif", m, rescale=f"0, 1")
LayerControl().add_to(m)

# open in a browser
# m
serve_map(m)
