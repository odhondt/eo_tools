# %%
"""Run the full VV/VH InSAR processor and display its outputs without SNAP data."""

import logging
from datetime import datetime
from math import pi
from pathlib import Path
from time import perf_counter

import folium
import geopandas as gpd
from folium import LayerControl

from eo_tools.S1.process import process_insar
from eo_tools_dev.util import palette_phi, serve_map, show_cog

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# %%
data_dir = Path("/data/S1")
ids = [
    "S1A_IW_SLC__1SDV_20230904T063730_20230904T063757_050174_0609E3_DAA1",
    "S1A_IW_SLC__1SDV_20230916T063730_20230916T063757_050349_060FCD_6814",
]
primary_path = data_dir / f"{ids[0]}.zip"
secondary_path = data_dir / f"{ids[1]}.zip"
# Use fresh outputs for each run, retaining previous runs for inspection.
output_dir = Path("/data/res") / (
    f"test-full-processor-dual-pol-{datetime.now():%Y%m%d-%H%M%S-%f}"
)
polarizations = ["vv", "vh"]

# %%
aoi_file = "/eo_tools/data/Morocco_tiny.geojson"
# aoi_file = "/eo_tools/data/Morocco_AOI.geojson"
shp = gpd.read_file(aoi_file).geometry[0]

process_args = dict(
    prm_path=str(primary_path),
    sec_path=str(secondary_path),
    output_dir=str(output_dir),
    aoi_name=None,
    shp=shp,
    pol=polarizations,
    subswaths=["IW1", "IW2", "IW3"],
    write_coherence=True,
    write_interferogram=True,
    write_primary_amplitude=True,
    write_secondary_amplitude=True,
    apply_fast_esd=True,
    dem_upsampling=1.8,
    dem_force_download=False,
    dem_buffer_arc_sec=40,
    boxcar_coherence=[3, 3],
    filter_ifg=True,
    multilook=[1, 4],
    warp_kernel="bicubic",
    warp_kernel_slc="lanczos",
    cal_type="beta",
    clip_to_shape=True,
    orb_dir="/data/S1_orbits/",
)
started = perf_counter()
out_dir = process_insar(**process_args)
elapsed = perf_counter() - started

# %%
# Confirm that both channels produced every requested final output.
for pol in polarizations:
    for variable in ["phi", "coh", "amp_prm", "amp_sec"]:
        path = out_dir / f"{variable}_{pol}.tif"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {pol.upper()} output: {path}")
logging.info("Dual-polarization InSAR outputs: %s", out_dir)
print(f"Processing elapsed time: {elapsed:.1f} s ({elapsed / 60:.2f} min)", flush=True)

# %%
# Inspect VV and VH phase/coherence layers independently; no reference data needed.
m = folium.Map()
for pol in polarizations:
    show_cog(
        str(out_dir / f"phi_{pol}.tif"),
        m,
        rescale=f"{-pi},{pi}",
        colormap=palette_phi(),
    )
    show_cog(str(out_dir / f"coh_{pol}.tif"), m, rescale="0,1")
LayerControl().add_to(m)
serve_map(m)
