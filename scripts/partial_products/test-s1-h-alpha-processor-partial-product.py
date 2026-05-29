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

from eo_tools.S1.process import process_h_alpha_dual
from eo_tools_dev.util import serve_map, show_cog

# %%
# change to your custom locations
data_dir = "/data/S1/partial_dls"

ids = [
    "S1A_IW_SLC__1SDV_20230904T063730_20230904T063757_050174_0609E3_DAA1",
]
slc_path = f"{data_dir}/{ids[0]}.partial.SAFE"
output_dir = "/data/res/test-h-alpha-processor-partial"

# %%
# load a geometry. For partial products, process_h_alpha_dual will use the AOI
# stored in partial_aoi.geojson and ignore this shp argument.
aoi_file = "/eo_tools/data/Morocco_tiny.geojson"
shp = gpd.read_file(aoi_file).geometry[0]

# %%

out_dir_ha = process_h_alpha_dual(
    slc_path=slc_path,
    output_dir=output_dir,
    aoi_name=None,
    shp=shp,
    subswaths=["IW1", "IW2", "IW3"],
    dem_upsampling=1.8,
    dem_force_download=False,
    dem_buffer_arc_sec=40,
    multilook=[1, 4],
    warp_kernel="bicubic",
    cal_type="beta",
    clip_to_shape=True,
)

# %%

m = folium.Map()
_ = show_cog(f"{out_dir_ha}/amp_vv.tif", m, rescale="0, 1")
_ = show_cog(f"{out_dir_ha}/amp_vh.tif", m, rescale="0, 1")
_ = show_cog(f"{out_dir_ha}/alpha.tif", m, rescale="0, 90")
_ = show_cog(f"{out_dir_ha}/H.tif", m, rescale="0, 1")
LayerControl().add_to(m)

# open in a browser
serve_map(m)
