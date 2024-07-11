# %%
import logging

logging.basicConfig(level=logging.INFO)

# Uncomment these lines to test conda imports
# import sys
# sys.path.remove("/eo_tools")
# import eo_tools
# print(f"EO-Tools imported from:")
# print(f"{eo_tools.__file__=}")

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
file_aoi = "/eo_tools/data/Morocco_tiny.geojson"
# file_aoi = "/eo_tools/data/Morocco_AOI.geojson"
shp = gpd.read_file(file_aoi).geometry[0]

search_criteria = {
    "productType": "S1_SAR_SLC",
    "start": "2023-09-03",
    "end": "2023-09-17",
    "geom": shp,
}

# uncomment if files are not already on disk
# results, _ = dag.search(**search_criteria)
# to_dl = [it for it in results if it.properties["id"] in ids]
# dag.download_all(to_dl, outputs_prefix="/data/S1/", extract=True)

# %%

out_dir_prev = f"{outputs_prefix}/S1_InSAR_2023-09-04-063730__2023-09-16-063730"

if os.path.isdir(out_dir_prev):
    remove(out_dir_prev)

process_args = dict(
    dir_prm=primary_dir,
    dir_sec=secondary_dir,
    outputs_prefix=outputs_prefix,
    aoi_name=None,
    shp=shp,
    pol="vv",
    subswaths=["IW1", "IW2", "IW3"],
    write_coherence=True,
    write_interferogram=True,
    write_primary_amplitude=True,
    write_secondary_amplitude=True,
    apply_fast_esd=True,
    dem_upsampling=1.8,
    dem_force_download=False,
    dem_buffer_arc_sec=40,
    boxcar_coherence=[3, 10],
    filter_ifg=True,
    multilook=[1, 4],
    warp_kernel="bicubic",
    clip_to_shape=True,
)

out_dir = process_insar(
   **process_args 
)

# %%
# Optional test
check_outputs = True
if check_outputs:
    import rioxarray as riox
    from xarray.testing import assert_allclose
    log = logging.getLogger(__name__)
    test_args = dict(
        dir_prm=primary_dir,
        dir_sec=secondary_dir,
        outputs_prefix=outputs_prefix,
        aoi_name=None,
        shp=shp,
        pol="vv",
        subswaths=["IW1", "IW2", "IW3"],
        write_coherence=True,
        write_interferogram=True,
        write_primary_amplitude=True,
        write_secondary_amplitude=True,
        apply_fast_esd=True,
        dem_upsampling=1.8,
        dem_force_download=False,
        dem_buffer_arc_sec=40,
        boxcar_coherence=[3, 10],
        filter_ifg=True,
        multilook=[1, 4],
        warp_kernel="bicubic",
        clip_to_shape=True,
    )

    if test_args != process_args or file_aoi != "/eo_tools/data/Morocco_tiny.geojson":
        raise ValueError("Wrong setup for output checking")
    else:
        log.info("Checking output validity")
        dir_ref = "/eo_tools/data/test-full-processor/S1_InSAR_2023-09-04-063730__2023-09-16-063730"
        for var in ["phi", "coh", "amp_prm", "amp_sec"]:
            da_ref = riox.open_rasterio(f"{dir_ref}/{var}_vv.tif", masked=True)
            da_out = riox.open_rasterio(f"{out_dir_prev}/{var}_vv.tif", masked=True)
            try:
                assert_allclose(da_ref, da_out)
                log.info(f"{var} outputs are OK")
            except:
                raise RuntimeError(f"Variable {var} deviates from reference")