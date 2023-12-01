import folium

from shapely import intersection_all
from shapely.geometry import mapping

import numpy as np
import rasterio
import geopandas as gpd
import json
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_hex

from localtileserver import get_folium_tile_layer
from localtileserver.client import TileClient


def visualize_S1_products_for_insar(product_geodataframe, aoi_geojson=None):
    """Visualize Sentinel-1 products on a map to choose pairs that are relevant for InSAR.

    Args:
        product_geodataframe (geopandas GeoDataFrame): products returned by sentinelsat (see example notebook).
        aoi_geojson (dict, optional): Area of interest to display on the map. Defaults to None.

    Returns:
        folium Map: interactive map
    """
    map_ = folium.Map()
    grouped = product_geodataframe.groupby(["relativeorbitnumber", "slicenumber"])
    groups = list(grouped.groups.keys())
    ll = product_geodataframe.total_bounds.reshape(2, 2)
    # folium inverts coordinates
    bbox = list(map(lambda x: [x[1], x[0]], ll))
    for g in groups:
        # convert Multipolygons to Polygons
        dfg = grouped.get_group(g).explode(index_parts=False).to_crs("EPSG:4236")
        # find the intersection of all geometries
        inter = intersection_all(dfg.geometry)

        # a collection is required for folium styling props
        gj = {}
        gj["type"] = "FeatureCollection"
        gj["features"] = []
        item = mapping(inter)
        item["properties"] = {}
        if all(dfg.orbitdirection == "ASCENDING"):
            item["properties"]["color"] = "cadetblue"
            orbit_conf = "Ascending"
        elif all(dfg.orbitdirection == "DESCENDING"):
            item["properties"]["color"] = "orange"
            orbit_conf = "Descending"
        else:
            # for safety -- not sure if this may happen
            item["properties"]["color"] = "red"
            orbit_conf = "Mix of ascending and descending"
        gj["features"].append(item)
        folium_products = folium.GeoJson(
            data=gj,
            style_function=lambda x: {
                "fillColor": "lightblue",
                "color": x["properties"]["color"],
            },
        )

        date_ts = dfg["beginposition"].dt.strftime("%Y-%m-%d %X")
        date_str = "<br>".join(
            date_ts.index.astype(str).str.cat(date_ts.values, sep=" / ")
        )
        folium.Tooltip(
            f"<b>Configuration:</b><br>{orbit_conf}, Orbit: {g[0]}, Slice: {g[1]}<br><b>Product list (index / date):</b><br>{date_str}"
        ).add_to(folium_products)
        folium_products.add_to(map_)

    if aoi_geojson is not None:
        folium_aoi = folium.GeoJson(
            data=aoi_geojson,
            style_function=lambda x: {
                "fillColor": "none",
                "color": "grey",
            },
        )
        folium.Tooltip(f"Area of Interest").add_to(folium_aoi)
        folium_aoi.add_to(map_)
    map_.fit_bounds(bbox)
    return map_


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
        )
    folium.Tooltip(f"Area of Interest").add_to(folium_aoi)
    folium_aoi.add_to(m)

    by_orbit = gdf.groupby("relativeOrbitNumber")
    group_indices = []
    for idx in by_orbit.groups:
        dfg = by_orbit.get_group(idx).explode(index_parts=False)

        # group very overlapping products
        group_indices = group_by_overlap(dfg)
        for g in group_indices:
            sel = gdf.loc[g]
            geom = mapping(sel.unary_union)

            folium_products = folium.GeoJson(geom).add_to(m)
            date_ts = sel["startTimeFromAscendingNode"].dt.strftime("%Y-%m-%d %X")
            date_str = "<br>".join(
                date_ts.index.astype(str).str.cat(date_ts.values, sep=" / ")
            )
            orbit_conf = sel.iloc[0]["orbitDirection"]
            orbit_num = sel.iloc[0]["relativeOrbitNumber"]
            folium.Tooltip(
                f"<b>Configuration:</b><br>{orbit_conf}, Orbit: {orbit_num}<br><b>Product list (index / date):</b><br>{date_str}"
            ).add_to(folium_products)

    m.fit_bounds(bbox)
    return m


def visualize_insar_phase(file_in):
    """Visualize interferometric phase on a map with a cyclic colormap (similar to SNAP).

    Args:
        file_in (str): GeoTiff input file (preferably COG)

    Returns:
        folium.Map: raster visualization on an interactive map
    """

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

    client = TileClient(file_in)
    t = get_folium_tile_layer(client, palette=cmap_hex)

    m = folium.Map(location=client.center(), zoom_start=client.default_zoom)
    t.add_to(m)
    return m


def visualize_insar_coh(file_in):
    """Visualize coherence on a map.

    Args:
        file_in (str): GeoTiff input file (preferably COG)

    Returns:
        folium.Map: raster visualization on an interactive map
    """

    client = TileClient(file_in)
    t = get_folium_tile_layer(client, vmin=0.0, vmax=1.0)

    m = folium.Map(location=client.center(), zoom_start=client.default_zoom)
    t.add_to(m)
    return m


def visualize_sar_intensity(file_in, vmin=None, vmax=None):
    """Visualize intensity on a map.

    Args:
        file_in (str): GeoTiff input file (preferably COG)

    Returns:
        folium.Map: raster visualization on an interactive map
    """

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
