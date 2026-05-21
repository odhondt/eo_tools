# %%
# Uncomment the next block to test conda imports

# import sys
# sys.path.remove("/eo_tools")
# sys.path.append("/eo_tools/") # workaround to include eo_tools_dev
# import eo_tools
# print(f"EO-Tools imported from:")
# print(f"{eo_tools.__file__=}")

import logging
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import rioxarray as riox
import xarray as xr
from folium import LayerControl
from math import pi
from matplotlib import colormaps
from matplotlib.colors import hsv_to_rgb

from eo_tools.S1.process import (
    amplitude,
    apply_to_patterns_for_pair,
    apply_to_patterns_for_single,
    coherence,
    geocode_and_merge_iw,
    prepare_insar,
    prepare_slc,
)
from eo_tools_dev.util import palette_phi, serve_map, show_cog

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("numexpr").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# %%
# Same products and AOI as test-s1-insar-processor-S1-C-D.py.
data_dir = "/data/S1"

ids = [
    "S1C_IW_SLC__1SDV_20260420T174607_20260420T174635_007303_00ECDF_FA97",
    "S1D_IW_SLC__1SDV_20260421T174617_20260421T174645_002448_00405E_4FF4",
]
primary_path = f"{data_dir}/{ids[0]}.zip"
secondary_path = f"{data_dir}/{ids[1]}.zip"
output_dir = "/data/res/test-s1c-s1d-custom-pipeline"

# %%
# load a geometry
aoi_file = "/eo_tools/data/Andorra_tiny.geojson"
shp = gpd.read_file(aoi_file).geometry[0]

# # %%
# out_dir = prepare_insar(
#     prm_path=primary_path,
#     sec_path=secondary_path,
#     output_dir=output_dir,
#     aoi_name=None,
#     shp=shp,
#     pol="full",
#     subswaths=["IW1", "IW2", "IW3"],
#     apply_fast_esd=True,
#     dem_name="cop-dem-glo-30",
#     dem_upsampling=1.8,
#     dem_force_download=False,
#     dem_buffer_arc_sec=40,
#     warp_kernel="bicubic",
#     cal_type="beta",
# )
#
#
# # %%
# def change_detection(amp_prm_file, amp_sec_file, out_file):
#     log.info("Smoothing amplitudes")
#     amp_prm = riox.open_rasterio(amp_prm_file)[0].rolling(x=7, y=7, center=True).mean()
#     amp_sec = riox.open_rasterio(amp_sec_file)[0].rolling(x=7, y=7, center=True).mean()
#     log.info("Incoherent changes")
#     ch = np.log(amp_prm + 1e-10) - np.log(amp_sec + 1e-10)
#     ch.rio.to_raster(out_file)
#
#
# # %%
# geo_dir = Path(out_dir).parent
#
# # compute interferometric coherence
# apply_to_patterns_for_pair(
#     coherence,
#     out_dir=out_dir,
#     prm_file_prefix="slc_prm",
#     sec_file_prefix="slc_sec",
#     out_file_prefix="coh",
#     box_size=[3, 3],
#     multilook=[1, 4],
# )
#
# # compute primary amplitude
# apply_to_patterns_for_single(
#     amplitude,
#     out_dir=out_dir,
#     in_file_prefix="slc_prm",
#     out_file_prefix="amp_prm",
#     multilook=[2, 8],
# )
#
# # compute secondary amplitude
# apply_to_patterns_for_single(
#     amplitude,
#     out_dir=out_dir,
#     in_file_prefix="slc_sec",
#     out_file_prefix="amp_sec",
#     multilook=[2, 8],
# )
#
# # compute incoherent changes
# apply_to_patterns_for_pair(
#     change_detection,
#     out_dir=out_dir,
#     prm_file_prefix="amp_prm",
#     sec_file_prefix="amp_sec",
#     out_file_prefix="change",
# )
#
# # %%
# geocode_and_merge_iw(
#     geo_dir,
#     shp=shp,
#     pol="full",
#     subswaths=["IW1", "IW2", "IW3"],
#     var_names=["coh", "change", "amp_prm", "amp_sec"],
#     clip_to_shape=False,
# )
#

# # %%
# # Prepare the first SLC product with terrain flattening and extract full-pol amplitudes.
# slc_out_dir = prepare_slc(
#     slc_path=primary_path,
#     output_dir=output_dir,
#     aoi_name="terrain",
#     shp=shp,
#     pol="full",
#     subswaths=["IW1", "IW2", "IW3"],
#     cal_type="terrain",
#     dem_name="cop-dem-glo-30",
#     dem_upsampling=1.8,
#     dem_force_download=False,
#     dem_buffer_arc_sec=40,
# )
# slc_geo_dir = Path(slc_out_dir).parent
#
# apply_to_patterns_for_single(
#     amplitude,
#     out_dir=slc_out_dir,
#     in_file_prefix="slc",
#     out_file_prefix="amp_terrain",
#     multilook=[2, 8],
# )
#
# geocode_and_merge_iw(
#     slc_geo_dir,
#     shp=shp,
#     pol="full",
#     subswaths=["IW1", "IW2", "IW3"],
#     var_names=["amp_terrain"],
#     clip_to_shape=False,
# )
#

# %%
# Hardcoded output directories so the visualization can be run independently.
geo_dir = Path(output_dir) / "S1_InSAR_2026-04-20-174607__2026-04-21-174617"
slc_geo_dir = Path(output_dir) / "S1_2026-04-20-174607_terrain"

# %%
def percentile_stretch(img, valid=None, percentiles=(2, 98), gamma=1.0):
    img = np.asarray(img, dtype="float32")
    if valid is None:
        valid = np.isfinite(img) & (img != 0)

    out = np.zeros(img.shape, dtype="float32")
    if not np.any(valid):
        return out

    vmin, vmax = np.nanpercentile(img[valid], percentiles)
    if vmax <= vmin:
        return out

    out[valid] = np.clip((img[valid] - vmin) / (vmax - vmin), 0, 1)
    if gamma != 1.0:
        out[valid] = out[valid] ** gamma
    return out


def write_polsar_rgb(in_dir, amp_prefix, out_file):
    in_dir = Path(in_dir)
    vv = riox.open_rasterio(in_dir / f"{amp_prefix}_vv.tif").squeeze("band", drop=True)
    vh = riox.open_rasterio(in_dir / f"{amp_prefix}_vh.tif").squeeze("band", drop=True)

    vv_data = vv.values.astype("float32")
    vh_data = vh.values.astype("float32")
    valid = np.isfinite(vv_data) & np.isfinite(vh_data) & (vv_data > 0) & (vh_data > 0)

    ratio = np.zeros_like(vv_data, dtype="float32")
    ratio[valid] = vh_data[valid] / (vv_data[valid] + 1e-30)

    red = percentile_stretch(np.log1p(vv_data), valid, gamma=0.85)
    green = percentile_stretch(np.log1p(vh_data), valid, gamma=0.85)
    blue = percentile_stretch(ratio, valid, percentiles=(2, 99), gamma=0.75)
    rgb = (255 * np.stack([red, green, blue])).round().astype("uint8")

    arrout = xr.DataArray(
        rgb,
        coords={"band": [1, 2, 3], **{dim: vv.coords[dim] for dim in vv.dims}},
        dims=("band", *vv.dims),
        name="polsar_rgb",
    )
    arrout.rio.write_crs(vv.rio.crs, inplace=True)
    arrout.rio.write_transform(vv.rio.transform(), inplace=True)
    arrout.rio.write_nodata(0, inplace=True)

    out_path = in_dir / out_file
    arrout.rio.to_raster(out_path, driver="COG")
    return out_path


def write_rgb_cog(template, rgb, out_path, name):
    arrout = xr.DataArray(
        rgb,
        coords={"band": [1, 2, 3], **{dim: template.coords[dim] for dim in template.dims}},
        dims=("band", *template.dims),
        name=name,
    )
    arrout.rio.write_crs(template.rio.crs, inplace=True)
    arrout.rio.write_transform(template.rio.transform(), inplace=True)
    arrout.rio.write_nodata(0, inplace=True)
    arrout.rio.to_raster(out_path, driver="COG")
    return out_path


def write_dual_coh_rgb(in_dir, out_file="coh_dual_rgb.tif", swap_red_blue=False):
    in_dir = Path(in_dir)
    vv = riox.open_rasterio(in_dir / "coh_vv.tif").squeeze("band", drop=True)
    vh = riox.open_rasterio(in_dir / "coh_vh.tif").squeeze("band", drop=True)

    vv_data = np.clip(vv.values.astype("float32"), 0, 1)
    vh_data = np.clip(vh.values.astype("float32"), 0, 1)
    valid = np.isfinite(vv_data) & np.isfinite(vh_data) & ((vv_data > 0) | (vh_data > 0))

    red = percentile_stretch(vv_data, valid, percentiles=(0, 100), gamma=1.0)
    green = percentile_stretch(vh_data, valid, percentiles=(0, 100), gamma=1.0)
    blue = percentile_stretch(np.abs(vv_data - vh_data), valid, percentiles=(1, 99), gamma=1.0)
    channels = [red, green, blue]
    if swap_red_blue:
        channels = [blue, green, red]

    rgb = (255 * np.stack(channels)).round().astype("uint8")
    return write_rgb_cog(vv, rgb, in_dir / out_file, Path(out_file).stem)


def write_dual_coh_rg(in_dir, out_file="coh_dual_rg.tif"):
    in_dir = Path(in_dir)
    vv = riox.open_rasterio(in_dir / "coh_vv.tif").squeeze("band", drop=True)
    vh = riox.open_rasterio(in_dir / "coh_vh.tif").squeeze("band", drop=True)

    vv_data = np.clip(vv.values.astype("float32"), 0, 1)
    vh_data = np.clip(vh.values.astype("float32"), 0, 1)
    valid = np.isfinite(vv_data) & np.isfinite(vh_data) & ((vv_data > 0) | (vh_data > 0))

    red = percentile_stretch(vv_data, valid, percentiles=(0, 100), gamma=1.0)
    green = percentile_stretch(vh_data, valid, percentiles=(0, 100), gamma=1.0)
    blue = np.zeros_like(red, dtype="float32")
    rgb = (255 * np.stack([red, green, blue])).round().astype("uint8")
    return write_rgb_cog(vv, rgb, in_dir / out_file, Path(out_file).stem)


def write_dual_coh_hsv(
    in_dir,
    out_file="coh_dual_hsv.tif",
    hue_span=0.95,
    hue_rotation=0.0,
    saturation_gain=2.5,
):
    in_dir = Path(in_dir)
    vv = riox.open_rasterio(in_dir / "coh_vv.tif").squeeze("band", drop=True)
    vh = riox.open_rasterio(in_dir / "coh_vh.tif").squeeze("band", drop=True)

    vv_data = np.clip(vv.values.astype("float32"), 0, 1)
    vh_data = np.clip(vh.values.astype("float32"), 0, 1)
    valid = np.isfinite(vv_data) & np.isfinite(vh_data) & ((vv_data > 0) | (vh_data > 0))

    total = vv_data + vh_data + 1e-30
    hue = np.zeros_like(vv_data, dtype="float32")
    saturation = np.zeros_like(vv_data, dtype="float32")
    value = np.zeros_like(vv_data, dtype="float32")

    # hue_span controls VV-to-VH color separation; hue_rotation shifts the palette.
    hue[valid] = (hue_rotation + hue_span * vh_data[valid] / total[valid]) % 1.0
    saturation[valid] = np.clip(
        saturation_gain * np.abs(vv_data[valid] - vh_data[valid]) / total[valid],
        0,
        1,
    )
    value[valid] = np.maximum(vv_data[valid], vh_data[valid])
    value = percentile_stretch(value, valid, percentiles=(1, 99), gamma=0.75)

    rgb = (255 * hsv_to_rgb(np.dstack([hue, saturation, value]))).round().astype("uint8")
    rgb = np.rollaxis(rgb, -1)
    return write_rgb_cog(vv, rgb, in_dir / out_file, "coh_dual_hsv")


def write_colormap_rgb(in_file, out_file, cmap_name="RdYlGn_r", vmin=0.0, vmax=1.0, gamma=0.85):
    src = riox.open_rasterio(in_file).squeeze("band", drop=True)
    data = src.values.astype("float32")
    valid = np.isfinite(data) & (data != 0)

    norm = np.zeros(data.shape, dtype="float32")
    norm[valid] = np.clip((data[valid] - vmin) / (vmax - vmin), 0, 1)
    if gamma != 1.0:
        norm[valid] = norm[valid] ** gamma

    rgba = colormaps[cmap_name](norm)
    rgb = (255 * np.moveaxis(rgba[..., :3], -1, 0)).round().astype("uint8")
    rgb[:, ~valid] = 0
    return write_rgb_cog(src, rgb, out_file, Path(out_file).stem)


def write_mean_coh_colormap(in_dir, out_file, cmap_name="RdYlGn_r"):
    in_dir = Path(in_dir)
    vv = riox.open_rasterio(in_dir / "coh_vv.tif").squeeze("band", drop=True)
    vh = riox.open_rasterio(in_dir / "coh_vh.tif").squeeze("band", drop=True)

    vv_data = np.clip(vv.values.astype("float32"), 0, 1)
    vh_data = np.clip(vh.values.astype("float32"), 0, 1)
    valid = np.isfinite(vv_data) & np.isfinite(vh_data) & ((vv_data > 0) | (vh_data > 0))
    mean_coh = np.zeros_like(vv_data, dtype="float32")
    mean_coh[valid] = 0.5 * (vv_data[valid] + vh_data[valid])

    rgba = colormaps[cmap_name](mean_coh)
    rgb = (255 * np.moveaxis(rgba[..., :3], -1, 0)).round().astype("uint8")
    rgb[:, ~valid] = 0
    return write_rgb_cog(vv, rgb, out_file, Path(out_file).stem)


def write_coh_difference_colormap_with_avg_brightness(
    in_dir,
    out_file,
    cmap_name="RdYlGn_r",
    stretch_percentile=99,
    brightness_gamma=0.65,
):
    in_dir = Path(in_dir)
    vv = riox.open_rasterio(in_dir / "coh_vv.tif").squeeze("band", drop=True)
    vh = riox.open_rasterio(in_dir / "coh_vh.tif").squeeze("band", drop=True)

    vv_data = np.clip(vv.values.astype("float32"), 0, 1)
    vh_data = np.clip(vh.values.astype("float32"), 0, 1)
    valid = np.isfinite(vv_data) & np.isfinite(vh_data) & ((vv_data > 0) | (vh_data > 0))

    avg = np.zeros_like(vv_data, dtype="float32")
    diff = np.zeros_like(vv_data, dtype="float32")
    avg[valid] = 0.5 * (vv_data[valid] + vh_data[valid])
    diff[valid] = vv_data[valid] - vh_data[valid]

    max_abs = np.nanpercentile(np.abs(diff[valid]), stretch_percentile)
    if max_abs == 0:
        max_abs = 1

    norm = np.zeros(diff.shape, dtype="float32")
    norm[valid] = np.clip((diff[valid] + max_abs) / (2 * max_abs), 0, 1)
    brightness = percentile_stretch(avg, valid, percentiles=(1, 99), gamma=brightness_gamma)

    rgb = colormaps[cmap_name](norm)[..., :3].astype("float32")
    rgb *= brightness[..., None]
    rgb[~valid] = 0
    rgb = (255 * np.moveaxis(rgb, -1, 0)).round().astype("uint8")
    return write_rgb_cog(vv, rgb, out_file, Path(out_file).stem)


def write_coh_avgdiff_images(in_dir):
    in_dir = Path(in_dir)
    vv = riox.open_rasterio(in_dir / "coh_vv.tif").squeeze("band", drop=True)
    vh = riox.open_rasterio(in_dir / "coh_vh.tif").squeeze("band", drop=True)

    vv_data = np.clip(vv.values.astype("float32"), 0, 1)
    vh_data = np.clip(vh.values.astype("float32"), 0, 1)
    valid = np.isfinite(vv_data) & np.isfinite(vh_data) & ((vv_data > 0) | (vh_data > 0))

    avg = np.zeros_like(vv_data, dtype="float32")
    diff = np.zeros_like(vv_data, dtype="float32")
    avg[valid] = 0.5 * (vv_data[valid] + vh_data[valid])
    diff[valid] = vv_data[valid] - vh_data[valid]

    brightness = percentile_stretch(avg, valid, percentiles=(1, 99), gamma=0.65)
    max_abs = np.nanpercentile(np.abs(diff[valid]), 99)
    if max_abs == 0:
        max_abs = 1

    signed_strength = np.zeros_like(diff, dtype="float32")
    signed_strength[valid] = np.clip(diff[valid] / max_abs, -1, 1)
    unsigned_strength = np.abs(signed_strength)

    green = np.array([0.10, 0.68, 0.30], dtype="float32")
    orange = np.array([1.00, 0.46, 0.05], dtype="float32")

    pos = np.clip(signed_strength, 0, 1)
    neg = np.clip(-signed_strength, 0, 1)
    signed_rgb = np.stack(
        [
            1 - neg * (1 - green[0]),
            1 - pos * (1 - orange[1]) - neg * (1 - green[1]),
            1 - pos * (1 - orange[2]) - neg * (1 - green[2]),
        ],
        axis=-1,
    )
    signed_rgb = np.clip(signed_rgb, 0, 1) * brightness[..., None]

    unsigned_rgb = np.stack(
        [
            np.ones_like(unsigned_strength),
            1 - unsigned_strength * (1 - orange[1]),
            1 - unsigned_strength * (1 - orange[2]),
        ],
        axis=-1,
    )
    unsigned_rgb *= brightness[..., None]

    signed_rgb[~valid] = 0
    unsigned_rgb[~valid] = 0

    signed_out = in_dir / "coh_avg_signed_diff_green_orange.tif"
    unsigned_out = in_dir / "coh_avg_unsigned_diff_orange.tif"
    signed_rgb = (255 * np.moveaxis(signed_rgb, -1, 0)).round().astype("uint8")
    unsigned_rgb = (255 * np.moveaxis(unsigned_rgb, -1, 0)).round().astype("uint8")
    write_rgb_cog(vv, signed_rgb, signed_out, signed_out.stem)
    write_rgb_cog(vv, unsigned_rgb, unsigned_out, unsigned_out.stem)
    return signed_out, unsigned_out


def write_coh_difference_colormap(
    in_dir, out_file, cmap_name="RdYlGn_r", stretch_percentile=99
):
    in_dir = Path(in_dir)
    vv = riox.open_rasterio(in_dir / "coh_vv.tif").squeeze("band", drop=True)
    vh = riox.open_rasterio(in_dir / "coh_vh.tif").squeeze("band", drop=True)

    vv_data = vv.values.astype("float32")
    vh_data = vh.values.astype("float32")
    valid = np.isfinite(vv_data) & np.isfinite(vh_data) & ((vv_data != 0) | (vh_data != 0))

    diff = np.zeros_like(vv_data, dtype="float32")
    diff[valid] = vv_data[valid] - vh_data[valid]
    max_abs = np.nanpercentile(np.abs(diff[valid]), stretch_percentile)
    if max_abs == 0:
        max_abs = 1

    # Symmetric normalization keeps sign: negative -> green, zero -> yellow, positive -> orange/red.
    norm = np.zeros(diff.shape, dtype="float32")
    norm[valid] = np.clip((diff[valid] + max_abs) / (2 * max_abs), 0, 1)

    rgba = colormaps[cmap_name](norm)
    rgb = (255 * np.moveaxis(rgba[..., :3], -1, 0)).round().astype("uint8")
    rgb[:, ~valid] = 0
    return write_rgb_cog(vv, rgb, out_file, Path(out_file).stem)


polsar_rgb_file = slc_geo_dir / "polsar_rgb_terrain.tif"
polsar_rgb_prm_file = geo_dir / "polsar_rgb_amp_prm.tif"
coh_dual_rgb_file = geo_dir / "coh_dual_rgb.tif"
coh_dual_rg_file = write_dual_coh_rg(geo_dir)
coh_dual_rgb_swapped_file = write_dual_coh_rgb(
    geo_dir, "coh_dual_rgb_swap_rb.tif", swap_red_blue=True
)
coh_dual_hsv_file = geo_dir / "coh_dual_hsv.tif"

coh_vv_rdylgn_file = write_colormap_rgb(
    geo_dir / "coh_vv.tif", geo_dir / "coh_vv_RdYlGn_r.tif", "RdYlGn_r"
)
coh_vh_rdylgn_file = write_colormap_rgb(
    geo_dir / "coh_vh.tif", geo_dir / "coh_vh_RdYlGn_r.tif", "RdYlGn_r"
)
coh_mean_rdylgn_file = write_mean_coh_colormap(
    geo_dir, geo_dir / "coh_mean_RdYlGn_r.tif", "RdYlGn_r"
)
coh_mean_rdylbu_file = write_mean_coh_colormap(
    geo_dir, geo_dir / "coh_mean_RdYlBu_r.tif", "RdYlBu_r"
)
coh_diff_rdylgn_file = write_coh_difference_colormap(
    geo_dir, geo_dir / "coh_diff_vv_minus_vh_RdYlGn_r.tif", "RdYlGn_r"
)
coh_diff_avg_brightness_file = write_coh_difference_colormap_with_avg_brightness(
    geo_dir, geo_dir / "coh_diff_vv_minus_vh_RdYlGn_r_avg_brightness.tif", "RdYlGn_r"
)
coh_avg_signed_file, coh_avg_unsigned_file = write_coh_avgdiff_images(geo_dir)

# %%
m = folium.Map()
# _ = show_cog(f"{geo_dir}/coh_vv.tif", m, rescale="0,1")
# _ = show_cog(
#     f"{geo_dir}/change_vv.tif", m, rescale="-0.25,0.25", colormap_name="rdbu_r"
# )
_ = show_cog(f"{slc_geo_dir}/amp_terrain_vv.tif", m, rescale="0,1")
_ = show_cog(f"{slc_geo_dir}/amp_terrain_vh.tif", m, rescale="0,1")
_ = show_cog(str(polsar_rgb_file), m, rescale="0,255")
_ = show_cog(str(polsar_rgb_prm_file), m, rescale="0,255")
_ = show_cog(str(coh_dual_rgb_file), m, rescale="0,255")
_ = show_cog(str(coh_dual_rg_file), m, rescale="0,255")
_ = show_cog(str(coh_dual_rgb_swapped_file), m, rescale="0,255")
_ = show_cog(str(coh_dual_hsv_file), m, rescale="0,255")
_ = show_cog(str(coh_vv_rdylgn_file), m, rescale="0,255")
_ = show_cog(str(coh_vh_rdylgn_file), m, rescale="0,255")
_ = show_cog(str(coh_mean_rdylgn_file), m, rescale="0,255")
_ = show_cog(str(coh_mean_rdylbu_file), m, rescale="0,255")
_ = show_cog(str(coh_diff_rdylgn_file), m, rescale="0,255")
_ = show_cog(str(coh_diff_avg_brightness_file), m, rescale="0,255")
_ = show_cog(str(coh_avg_signed_file), m, rescale="0,255")
_ = show_cog(str(coh_avg_unsigned_file), m, rescale="0,255")

# Uncomment if the custom pipeline is extended to write phase products.
# _ = show_cog(
#     f"{geo_dir}/phi_vv.tif",
#     m,
#     rescale=f"{-pi},{pi}",
#     colormap=palette_phi(),
# )

LayerControl().add_to(m)
serve_map(m)
