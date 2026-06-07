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
import geopandas as gpd
from folium import LayerControl

from eo_tools.S1.process import process_slc
from eo_tools_dev.util import serve_map, show_cog

# %%
# change to your custom locations
data_dir = "/data/S1/partial_dls"

ids = [
    "S1A_IW_SLC__1SDV_20181228T050448_20181228T050515_025221_02C9BE_1E22",
]
slc_path = f"{data_dir}/{ids[0]}.partial.SAFE"
output_dir_1 = "/data/res/test-slc-processor-rtc-nasadem-partial"
output_dir_2 = "/data/res/test-slc-processor-rtc-alos-partial"
output_dir_3 = "/data/res/test-slc-processor-rtc-glo30-partial"
output_dir_4 = "/data/res/test-slc-processor-rtc-glo90-partial"

# %%
# load a geometry. For partial products, process_slc will use the AOI stored in
# partial_aoi.geojson and ignore this shp argument.
aoi_file = "/eo_tools/data/Etna.geojson"
shp = gpd.read_file(aoi_file).geometry[0]

# %%

common_args = dict(
    slc_path=slc_path,
    aoi_name=None,
    shp=shp,
    pol="vv",
    subswaths=["IW1", "IW2", "IW3"],
    dem_upsampling=1.8,
    dem_force_download=True,
    dem_buffer_arc_sec=40,
    multilook=[1, 4],
    warp_kernel="bicubic",
    cal_type="terrain",
    clip_to_shape=True,
)

out_dir_nasadem = process_slc(
    output_dir=output_dir_1,
    dem_name="nasadem",
    **common_args,
)

# %%

out_dir_alos = process_slc(
    output_dir=output_dir_2,
    dem_name="alos-dem",
    **common_args,
)

# %%

out_dir_glo30 = process_slc(
    output_dir=output_dir_3,
    dem_name="cop-dem-glo-30",
    **common_args,
)

# %%

out_dir_glo90 = process_slc(
    output_dir=output_dir_4,
    dem_name="cop-dem-glo-90",
    # adapted upsampling to resolution
    dem_upsampling=5.4,
    **{k: v for k, v in common_args.items() if k != "dem_upsampling"},
)

# %%

m = folium.Map()
_ = show_cog(f"{out_dir_nasadem}/amp_vv.tif", m, rescale="0, 1")
_ = show_cog(f"{out_dir_alos}/amp_vv.tif", m, rescale="0, 1")
_ = show_cog(f"{out_dir_glo30}/amp_vv.tif", m, rescale="0, 1")
_ = show_cog(f"{out_dir_glo90}/amp_vv.tif", m, rescale="0, 1")
LayerControl().add_to(m)

# open in a browser
serve_map(m)
