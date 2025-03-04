# %%
# Uncomment the next block to test conda imports

# import sys
# sys.path.remove("/eo_tools")
# sys.path.append("/eo_tools/") # workaround to include eo_tools_dev
# import eo_tools
# print(f"EO-Tools imported from:")
# print(f"{eo_tools.__file__=}")

import geopandas as gpd
from eodag import EODataAccessGateway
import rioxarray as riox
import numpy as np
import folium
from folium import LayerControl
from eo_tools_dev.util import show_cog
from eo_tools_dev.util import serve_map

# credentials need to be stored in the following file (see EODAG docs)
confpath = "/data/eodag_config.yml"
dag = EODataAccessGateway(user_conf_file_path=confpath)
# make sure cop_dataspace will be used
dag.set_preferred_provider("cop_dataspace")

import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("numexpr").setLevel(logging.WARNING)
log = logging.getLogger(__name__)
# %%
# change to your custom locations
data_dir = "/data/S1"

ids = [
    "S1A_IW_SLC__1SDV_20230904T063730_20230904T063757_050174_0609E3_DAA1",
    "S1A_IW_SLC__1SDV_20230916T063730_20230916T063757_050349_060FCD_6814",
]
primary_path = f"{data_dir}/{ids[0]}.zip"
secondary_path = f"{data_dir}/{ids[1]}.zip"
output_dir = "/data/res/test-change-detection-pipeline"
# %%
# load a geometry
# aoi_file = "/eo_tools/data/Morocco_small.geojson"
aoi_file = "/eo_tools/data/Morocco_tiny.geojson"
# aoi_file = "/eo_tools/data/Morocco_AOI.geojson"
shp = gpd.read_file(aoi_file).geometry[0]

search_criteria = {
    "productType": "S1_SAR_SLC",
    "start": "2023-09-03",
    "end": "2023-09-17",
    "geom": shp,
}

results = dag.search(**search_criteria)
to_dl = [it for it in results if it.properties["id"] in ids]
print(f"{len(to_dl)} products to download")
# dag.download_all(to_dl, output_dir="/data/S1/", extract=False)
# %%
from eo_tools.S1.process import prepare_insar

out_dir = prepare_insar(
    prm_path=primary_path,
    sec_path=secondary_path,
    output_dir=output_dir,
    aoi_name=None,
    shp=shp,
    pol="full",
    subswaths=["IW1", "IW2", "IW3"],
    cal_type="sigma",
    apply_fast_esd=False,
    dem_upsampling=1.8,
    dem_force_download=False,
    dem_buffer_arc_sec=40,
    warp_kernel="bicubic",
)


# %%
def change_detection(amp_prm_file, amp_sec_file, out_file):
    log.info("Smoothing amplitudes")
    amp_prm = riox.open_rasterio(amp_prm_file)[0].rolling(x=7, y=7, center=True).mean()
    amp_sec = riox.open_rasterio(amp_sec_file)[0].rolling(x=7, y=7, center=True).mean()
    log.info("Incoherent changes")
    ch = np.log(amp_prm + 1e-10) - np.log(amp_sec + 1e-10)
    ch.rio.to_raster(out_file)


# %%
from eo_tools.S1.process import coherence, amplitude
from eo_tools.S1.process import apply_to_patterns_for_pair, apply_to_patterns_for_single
from pathlib import Path

out_dir = f"{output_dir}/S1_InSAR_2023-09-04-063730__2023-09-16-063730/sar"
geo_dir = Path(out_dir).parent

# compute interferometric coherence
apply_to_patterns_for_pair(
    coherence,
    out_dir=out_dir,
    prm_file_prefix="slc_prm",
    sec_file_prefix="slc_sec",
    out_file_prefix="coh",
    box_size=[3, 3],
    multilook=[1, 4],
)

# compute primary amplitude
apply_to_patterns_for_single(
    amplitude,
    out_dir=out_dir,
    in_file_prefix="slc_prm",
    out_file_prefix="amp_prm",
    multilook=[2, 8],
)

# compute secondary amplitude
apply_to_patterns_for_single(
    amplitude,
    out_dir=out_dir,
    in_file_prefix="slc_sec",
    out_file_prefix="amp_sec",
    multilook=[2, 8],
)

# compute incoherent changes
apply_to_patterns_for_pair(
    change_detection,
    out_dir=out_dir,
    prm_file_prefix="amp_prm",
    sec_file_prefix="amp_sec",
    out_file_prefix="change",
)

# %%
from eo_tools.S1.process import geocode_and_merge_iw

geo_dir = Path(out_dir).parent
geocode_and_merge_iw(geo_dir, shp=shp, var_names=["coh", "change"], clip_to_shape=False)

# %%
m = folium.Map()
_ = show_cog(f"{geo_dir}/coh_vv.tif", m, rescale="0,1")
_ = show_cog(f"{geo_dir}/coh_vh.tif", m, rescale="0,1")
_ = show_cog(
    f"{geo_dir}/change_vv.tif", m, rescale="-0.25,0.25", colormap_name="rdbu_r"
)
_ = show_cog(
    f"{geo_dir}/change_vh.tif", m, rescale="-0.25,0.25", colormap_name="rdbu_r"
)
LayerControl().add_to(m)
serve_map(m)
