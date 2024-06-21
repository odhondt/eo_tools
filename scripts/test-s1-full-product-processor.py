# %%
import logging

logging.basicConfig(level=logging.INFO)

# from folium import LayerControl
import os
import geopandas as gpd
from eodag import EODataAccessGateway
from eo_tools.S1.process import process_insar
from eo_tools.auxils import remove

# credentials need to be stored in the following file (see EODAG docs)
confpath = "/data/eodag_config.yml"
dag = EODataAccessGateway(user_conf_file_path=confpath)
# make sure cop_dataspace will be used
dag.set_preferred_provider("cop_dataspace")
logging.basicConfig(level=logging.INFO)

# %%
data_dir = "/data/S1"

ids = [
    "S1A_IW_SLC__1SDV_20230904T063730_20230904T063757_050174_0609E3_DAA1",
    "S1A_IW_SLC__1SDV_20230916T063730_20230916T063757_050349_060FCD_6814",
]
primary_dir = f"{data_dir}/{ids[0]}.SAFE"
secondary_dir = f"{data_dir}/{ids[1]}.SAFE"
outputs_prefix = "/data/res/test-full-processor"

# %%
# load a geometry
file_aoi = "/eo_tools/data/Morocco_AOI.geojson"
shp = gpd.read_file(file_aoi).geometry[0]

search_criteria = {
    "productType": "S1_SAR_SLC",
    "start": "2023-09-03",
    "end": "2023-09-17",
    "geom": shp,
}

results, _ = dag.search(**search_criteria)
to_dl = [it for it in results if it.properties["id"] in ids]
dag.download_all(to_dl, outputs_prefix="/data/S1/", extract=False)

# %%

out_dir_prev = f"{outputs_prefix}/S1_InSAR_2023-09-04-063730__2023-09-16-063730"

if os.path.isdir(out_dir_prev):
    remove(out_dir_prev)

out_dir = process_insar(
    dir_prm=primary_dir,
    dir_sec=secondary_dir,
    outputs_prefix=outputs_prefix,
    aoi_name=None,
    shp=shp,
    pol="vv",
    write_coherence=True,
    write_interferogram=True,
    write_primary_amplitude=True,
    write_secondary_amplitude=True,
    apply_fast_esd=True,
    subswaths=["IW1", "IW2", "IW3"],
    dem_upsampling=1.8,
    dem_force_download=False,
    dem_buffer_arc_sec=40,
    boxcar_coherence=[3, 10],
    filter_ifg=True,
    multilook=[1, 4],
    warp_kernel="bicubic",
    clip_to_shape=True,
)

# %%
