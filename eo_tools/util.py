import folium

from shapely import intersection_all
from shapely.geometry import mapping


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
