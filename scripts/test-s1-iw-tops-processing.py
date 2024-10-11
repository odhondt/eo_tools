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

import geopandas as gpd
# from eodag import EODataAccessGateway
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
# change to your custom locations
data_dir = "/data/S1"
out_dir = f"/data/res/test-iw-tops-process"

ids = [
 "S1A_IW_SLC__1SDV_20230904T063730_20230904T063757_050174_0609E3_DAA1", 
 "S1A_IW_SLC__1SDV_20230916T063730_20230916T063757_050349_060FCD_6814"
]
primary_dir = f"{data_dir}/{ids[0]}.zip"
secondary_dir = f"{data_dir}/{ids[1]}.zip"

iw = 1 # subswath
pol = "vv" # polarization ("vv"or "vh")
min_burst = 3
max_burst = 6 # Set to None to process all (warning: memory hungry)


# %%
# load a geometry
file_aoi = "/eo_tools/data/Morocco_AOI.geojson"
shp = gpd.read_file(file_aoi).geometry[0]

# search_criteria = {
#     "productType": "S1_SAR_SLC",
#     "start": "2023-09-03",
#     "end": "2023-09-17",
#     "geom": shp
# }

# results, _ = dag.search(**search_criteria)
# to_dl = [it for it in results if it.properties["id"] in ids]
# print(f"{len(to_dl)} products to download")
# dag.download_all(to_dl, outputs_prefix="/data/S1/", extract=False)

# %%
from eo_tools.S1.process import preprocess_insar_iw

# TODO: use downloaded products
preprocess_insar_iw(
    primary_dir,
    secondary_dir,
    out_dir,
    iw=iw,
    pol=pol,
    min_burst=min_burst,
    max_burst=max_burst,
    dem_upsampling=1.8,
    apply_fast_esd=True,
    dem_force_download=False
)

# %%
from eo_tools.S1.process import sar2geo, coherence, amplitude

file_prm = f"{out_dir}/primary.tif"
file_sec = f"{out_dir}/secondary.tif"
file_amp = f"{out_dir}/amp.tif"
file_coh = f"{out_dir}/coh.tif"
file_phi_geo = f"{out_dir}/phi_geo.tif"
file_amp_geo = f"{out_dir}/amp_geo.tif"
file_coh_geo = f"{out_dir}/coh_geo.tif"
file_lut = f"{out_dir}/lut.tif"
# computing amplitude and complex coherence  in the radar geometry
coherence(
    file_prm, file_sec, file_coh, box_size=[3, 10], multilook=[1, 4], magnitude=False
)
amplitude(file_prm, file_amp, multilook=[2, 8])

# combined multilooking and geocoding
# interferometric coherence
sar2geo(
    file_coh,
    file_lut,
    file_coh_geo,
    kernel="bicubic",
    write_phase=False,
    magnitude_only=True,
)

# interferometric phase
sar2geo(
    file_coh,
    file_lut,
    file_phi_geo,
    kernel="bicubic",
    write_phase=True,
    magnitude_only=False,
)

# amplitude of the primary image
sar2geo(
    file_amp,
    file_lut,
    file_amp_geo,
    kernel="bicubic",
    write_phase=False,
    magnitude_only=False,
)

# %%
# out_dir = "/data/res/test-full-processor/S1_InSAR_2023-09-04-063730__2023-09-16-063730"
ref_dir = "/data/reference/S1_InSAR_VV_2023-09-04-063730__2023-09-16-063730_Morocco"

m = folium.Map()
_ = show_cog(f"{ref_dir}/phi.tif", m, rescale=f"{-pi},{pi}", colormap=palette_phi())
_ = show_cog(f"{out_dir}/phi_geo.tif", m, rescale=f"{-pi},{pi}", colormap=palette_phi())
_ = show_cog(f"{ref_dir}/coh.tif", m, rescale=f"0,1")
_ = show_cog(f"{out_dir}/coh_geo.tif", m, rescale=f"0,1")
# _ = show_cog(f"{ref_dir}/amp_prm.tif", m, rescale=f"0,1")
# _ = show_cog(f"{out_dir}/amp_prm.tif", m, rescale=f"0,1")
LayerControl().add_to(m)

# open in a browser
serve_map(m)