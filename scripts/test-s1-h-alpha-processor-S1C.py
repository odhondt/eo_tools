# %%
import logging

logging.basicConfig(level=logging.INFO)
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
import os
from eo_tools.auxils import remove

from eo_tools_dev.util import show_cog, serve_map
from eo_tools.S1.process import process_h_alpha_dual
import geopandas as gpd
from eodag import EODataAccessGateway
from math import sqrt

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
    "S1C_IW_SLC__1SDV_20250401T174601_20250401T174629_001703_002E2A_4DC9",
]
slc_path = f"{data_dir}/{ids[0]}.zip"
output_dir = "/data/res/test-h-alpha-processor"

# %%
# load a geometry
aoi_file = "/eo_tools/data/Andorra_tiny.geojson"
shp = gpd.read_file(aoi_file).geometry[0]

# search_criteria = {
#     "productType": "S1_SAR_SLC",
#     "start": "2023-09-03",
#     "end": "2023-09-17",
#     "geom": shp,
# }

# results = dag.search(**search_criteria)
# to_dl = [it for it in results if it.properties["id"] in ids]
# print(f"{len(to_dl)} products to download")
# dag.download_all(to_dl, output_dir="/data/S1/", extract=True)

# %%


out_dir_ha = process_h_alpha_dual(
    slc_path=slc_path,
    output_dir=output_dir,
    aoi_name="beta_cal",
    shp=shp,
    subswaths=["IW1"],
    write_vv_amplitude=True,
    write_vh_amplitude=True,
    # subswaths=["IW1", "IW2", "IW3"],
    dem_upsampling=1.8,
    dem_force_download=False,
    dem_buffer_arc_sec=40,
    multilook=[1, 4],
    warp_kernel="bicubic",
    cal_type="beta",
    clip_to_shape=False,
)

# %%

m = folium.Map()
_ = show_cog(f"{out_dir_ha}/amp_vv.tif", m, rescale=f"0, 1")
_ = show_cog(f"{out_dir_ha}/amp_vh.tif", m, rescale=f"0, {1./sqrt(2)}")
_ = show_cog(f"{out_dir_ha}/alpha.tif", m, rescale=f"0, 90")
_ = show_cog(f"{out_dir_ha}/H.tif", m, rescale=f"0, 1")
LayerControl().add_to(m)

# open in a browser
serve_map(m)