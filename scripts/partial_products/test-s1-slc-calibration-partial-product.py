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
    "S1A_IW_SLC__1SDV_20230904T063730_20230904T063757_050174_0609E3_DAA1",
]
slc_path = f"{data_dir}/{ids[0]}.partial.SAFE"
output_dir = "/data/res/test-slc-processor-sigma-partial"
output_dir_2 = "/data/res/test-slc-processor-beta-partial"
output_dir_3 = "/data/res/test-slc-processor-rtc-partial"
output_dir_4 = "/data/res/test-slc-processor-orbit-bary-partial"

# %%
# load a geometry. For partial products, process_slc will use the AOI stored in
# partial_aoi.geojson and ignore this shp argument.
aoi_file = "/eo_tools/data/Morocco_tiny.geojson"
shp = gpd.read_file(aoi_file).geometry[0]

# %%

common_args = dict(
    slc_path=slc_path,
    aoi_name=None,
    shp=shp,
    pol="full",
    subswaths=["IW1", "IW2", "IW3"],
    dem_upsampling=1.8,
    dem_force_download=False,
    dem_buffer_arc_sec=40,
    multilook=[1, 4],
    warp_kernel="bicubic",
    clip_to_shape=True,
)

out_dir_sigma = process_slc(
    output_dir=output_dir,
    cal_type="sigma",
    **common_args,
)

# %%

out_dir_beta = process_slc(
    output_dir=output_dir_2,
    cal_type="beta",
    **common_args,
)

# %%

out_dir_rtc = process_slc(
    output_dir=output_dir_3,
    cal_type="terrain",
    **common_args,
)

# %%

out_dir_orb = process_slc(
    output_dir=output_dir_4,
    cal_type="beta",
    orbit_interpolator="bary",
    **common_args,
)

# %%

m = folium.Map()
_ = show_cog(f"{out_dir_beta}/amp_vv.tif", m, rescale="0, 1")
_ = show_cog(f"{out_dir_orb}/amp_vv.tif", m, rescale="0, 1")
_ = show_cog(f"{out_dir_sigma}/amp_vv.tif", m, rescale="0, 1")
_ = show_cog(f"{out_dir_rtc}/amp_vv.tif", m, rescale="0, 1")
_ = show_cog(f"{out_dir_beta}/amp_vh.tif", m, rescale="0, 1")
_ = show_cog(f"{out_dir_orb}/amp_vh.tif", m, rescale="0, 1")
_ = show_cog(f"{out_dir_sigma}/amp_vh.tif", m, rescale="0, 1")
_ = show_cog(f"{out_dir_rtc}/amp_vh.tif", m, rescale="0, 1")
LayerControl().add_to(m)

# open in a browser
serve_map(m)
