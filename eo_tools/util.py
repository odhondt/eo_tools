import folium

from shapely import intersection_all
from shapely.geometry import mapping

import os
import numpy as np
import rasterio
import geopandas as gpd
import json
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_hex
from eo_tools.S2 import make_s2_rgb, make_s2_color

from localtileserver import get_folium_tile_layer
from localtileserver.client import TileClient
import httpx


# check for geometrical overlap
def has_overlap(geom1, geom2, tolerance=0.01):
    intersection = geom1.intersection(geom2)
    return intersection.area / min(geom1.area, geom2.area) >= (1 - tolerance)


# make groups of almost fully overlapping products
def group_by_overlap(dfg):
    init_list = list(dfg.index)
    groups = []
    while len(init_list) > 0:
        geom_a = dfg.geometry.loc[init_list[0]]
        grp = []
        grp.append(init_list.pop(0))
        i = 0
        while i < len(init_list):
            geom_b = dfg.geometry.loc[init_list[i]]
            if has_overlap(geom_a, geom_b):
                grp.append(init_list.pop(i))
            else:
                i += 1
        groups.append(grp)
    return groups


# TODO: color styling
def explore_products(products, aoi=None):
    # Convert results to geodataframe
    gj = products.as_geojson_object()
    gdf = gpd.read_file(json.dumps(gj))

    ll = gdf.total_bounds.reshape(2, 2)
    # folium inverts coordinates
    bbox = list(map(lambda x: [x[1], x[0]], ll))

    m = folium.Map()
    if aoi is not None:
        folium_aoi = folium.GeoJson(
            data=mapping(aoi),
            style_function=lambda x: {
                "fillColor": "none",
                "color": "black",
            },
            name="Area of Interest",
        )
    folium.Tooltip(f"Area of Interest").add_to(folium_aoi)
    folium_aoi.add_to(m)

    by_orbit = gdf.groupby("relativeOrbitNumber")
    group_indices = []
    for idx in by_orbit.groups:
        dfg = by_orbit.get_group(idx).explode(index_parts=False)

        # group very overlapping products
        group_indices = group_by_overlap(dfg)
        for i, g in enumerate(group_indices):
            sel = gdf.loc[g]
            geom = mapping(sel.unary_union)

            orbit_conf = sel.iloc[0]["orbitDirection"]
            orbit_num = sel.iloc[0]["relativeOrbitNumber"]
            folium_products = folium.GeoJson(
                geom, name=f"Orbit {orbit_num}, group {i+1}"
            ).add_to(m)
            # folium_products = folium.GeoJson(geom).add_to(m)
            date_ts = sel["startTimeFromAscendingNode"].dt.strftime("%Y-%m-%d %X")
            date_str = "<br>".join(
                date_ts.index.astype(str).str.cat(date_ts.values, sep=" / ")
            )
            folium.Tooltip(
                f"<b>Configuration:</b><br>{orbit_conf}, Orbit: {orbit_num}<br><b>Product list (index / date):</b><br>{date_str}"
            ).add_to(folium_products)
    folium.LayerControl().add_to(m)
    m.fit_bounds(bbox)
    return m


def ttcog_get_stats(url):
    titiler_endpoint = "http://localhost:8085"
    r = httpx.get(
        f"{titiler_endpoint}/cog/statistics",
        params={
            "url": url,
        },
    ).json()
    return r


def ttcog_get_info(url):
    titiler_endpoint = "http://localhost:8085"
    r = httpx.get(
        f"{titiler_endpoint}/cog/info",
        params={
            "url": url,
        },
    ).json()
    return r


def ttcog_get_tilejson(url, **kwargs):
    titiler_endpoint = "http://localhost:8085"
    r = httpx.get(
        f"{titiler_endpoint}/cog/tilejson.json", params={"url": url, **kwargs}
    ).json()
    return r


def show_insar_phi(input_path):
    # def visualize_insar_phase(input_path):
    """Visualize interferometric phase on a map with a cyclic colormap (similar to SNAP).

    Args:
        input_path (str): Directory with InSAR products (phi.tif) or GeoTiff input file (preferably COG).

    Returns:
        folium.Map: raster visualization on an interactive map
    """
    if os.path.isdir(input_path):
        print("dir")
        file_in = f"{input_path}/phi.tif"
    elif os.path.isfile(input_path):
        print("file")
        file_in = input_path
    else:
        raise FileExistsError(f"Problem reading file.")
    # palette used by SNAP for insar phase
    palette = [
        [110, 60, 170],
        [210, 60, 160],
        [255, 110, 70],
        [200, 200, 50],
        [80, 245, 100],
        [25, 200, 180],
        [60, 130, 220],
        [100, 70, 190],
    ]

    palette_norm = [np.array(it) / 255 for it in palette]
    interp_cmap = LinearSegmentedColormap.from_list("cubehelix_cycle", palette_norm)
    cmap_hex = list(map(to_hex, interp_cmap(np.linspace(0, 1, 256))))

    info = ttcog_get_info(file_in)
    bounds = info["bounds"]

    stats = ttcog_get_stats(file_in)

    # col_idx = (65535 * np.arange(0,256) / 255).astype(int).tolist()
    col_idx = range(0,256)
    tjson = ttcog_get_tilejson(
        file_in,
        rescale=f"{-np.pi},{np.pi}",
        resampling="nearest", # make sure COG has been made with nearest too
        colormap=json.dumps({x: y for x, y in zip(col_idx, cmap_hex)}),
    )

    m = folium.Map(
        location=((bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2),
        zoom_start=8,
    )

    folium.TileLayer(tiles=tjson["tiles"][0], attr="InSAR Coherence").add_to(m)

    return m


def show_insar_coh(input_path):
    """Visualize coherence on a map.

    Args:
        input_path (str): Directory with InSAR products (coh.tif) or GeoTiff input file (preferably COG).

    Returns:
        folium.Map: raster visualization on an interactive map
    """

    if os.path.isdir(input_path):
        file_in = f"{input_path}/coh.tif"
    elif os.path.isfile(input_path):
        file_in = input_path
    else:
        raise FileExistsError("Problem reading file or file does not exist.")

    info = ttcog_get_info(file_in)
    bounds = info["bounds"]
    tjson = ttcog_get_tilejson(file_in, rescale="0,1")

    m = folium.Map(
        location=((bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2),
        zoom_start=8,
    )

    folium.TileLayer(tiles=tjson["tiles"][0], attr="InSAR Coherence").add_to(m)
    return m


# TODO: add dB
# def visualize_sar_intensity(input_path, master=True, vmin=None, vmax=None):
def show_sar_int(input_path, master=True, vmin=None, vmax=None):
    """Visualize intensity on a map.

    Args:
        input path (str): Directory with InSAR products (int_mst.tif or int_slv.tif) or GeoTiff input file (preferably COG).
        master (bool): Read master file if True, slave file otherwise. This has no effect if input_path points to a file.
        vmin (float): minimum clipping value (0 if not set)
        vmax (float): maximum clipping value (2.5*mean(raster) if not set)

    Returns:
        folium.Map: raster visualization on an interactive map
    """
    if os.path.isdir(input_path):
        if master:
            file_in = f"{input_path}/int_mst.tif"
        else:
            file_in = f"{input_path}/int_slv.tif"
    elif os.path.isfile(input_path):
        file_in = input_path
    else:
        raise FileExistsError("Problem reading file or file does not exist.")
    try:
        with rasterio.open(file_in) as src:
            mean_val = src.tags()["mean_value"]
    except:
        raise Exception("File not found or no 'mean_value' tag.")
    client = TileClient(file_in)

    if vmin is None:
        vmin_ = 0
    else:
        vmin_ = vmin

    if vmax is None:
        vmax_ = 2.5 * float(mean_val)
    else:
        vmax_ = vmax

    t = get_folium_tile_layer(client, vmin=vmin_, vmax=vmax_)

    m = folium.Map(location=client.center(), zoom_start=client.default_zoom)
    t.add_to(m)
    return m


def visualize_s2_rgb(input_dir, force_create=False):
    """Visualize Sentinel-2 RGB color image on a map

    Args:
        input_dir (str): directory that has RGB (B4, B3, B2) files.
        force_create (bool, optional): Force create RGB.tif event if file already exists. Defaults to False.

    Returns:
        folium.Map: raster visualization on an interactive map
    """
    rgb_path = f"{input_dir}/RGB.tif"
    if not os.path.exists(rgb_path) or force_create:
        print("RGB.tif not found (or force_create==True). Creating the file.")
        make_s2_rgb(input_dir)

    client = TileClient(rgb_path)
    t = get_folium_tile_layer(client)

    m = folium.Map(location=client.center(), zoom_start=client.default_zoom)
    t.add_to(m)
    return m


def visualize_s2_color(input_dir, name="RGB", force_create=False):
    """Visualize Sentinel-2 color image on a map

    Args:
        input_dir (str): directory that contains GeoTIFF band files.
        name (str, optional): Name of the pre-defined color representation. Possible choices are 'RGB', 'CIR', 'SWIR', 'agri', 'geol', 'bathy'. Defaults to "RGB".
        force_create (bool, optional): Force create RGB.tif event if file already exists. Defaults to False.

    Returns:
        folium.Map: raster visualization on an interactive map
    """
    im_path = f"{input_dir}/{name}.tif"
    if not os.path.exists(im_path) or force_create:
        print(f"{name}.tif not found (or force_create==True). Creating the file.")
        make_s2_color(input_dir, name)

    client = TileClient(im_path)
    t = get_folium_tile_layer(client)

    m = folium.Map(location=client.center(), zoom_start=client.default_zoom)
    t.add_to(m)
    return m


# TODO: Viz single band (any raster)
# TODO: Other band combinations ( NDVI, ...)
# TODO: Viz InSAR composite (HSV)
