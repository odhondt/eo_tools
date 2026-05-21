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
    "S1A_IW_SLC__1SDV_20241205T172346_20241205T172413_056860_06FB65_2BBF",
]
slc_path = f"{data_dir}/{ids[0]}.SAFE"
output_dir = "/data/res/test-slc-geocoding"

out_dir = process_slc(
    slc_path=slc_path,
    output_dir=output_dir,
    aoi_name=None,
    pol="vv",
    subswaths=["IW1"],
    dem_name="cop-dem-glo-30",
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
_ = show_cog(f"{out_dir}/amp_vv.tif", m, rescale=f"0, 1")
LayerControl().add_to(m)

# open in a browser
serve_map(m)
