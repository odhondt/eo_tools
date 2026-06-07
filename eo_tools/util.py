import folium

from shapely.geometry import mapping

import geopandas as gpd
import json


def explore_products(products, aoi=None):
    """Visualize product footprints on a map.
    Products with almost 100% overlap are automatically grouped.

    Args:
        products (eodag products): List of products produced by EODAG search.
        aoi (shapely geometry, optional): Adds AOI to the map. Defaults to None.

    Returns:
        folium.Map: Interactive map.

    Note:
        Hover on the map to see the product characteristics. Overlapping products are grouped for better visibility. Indices can be used to select products to download. For instance for InSAR pairs, select two products with nearly identical footprints.
    """
    # Convert results to geodataframe
    if isinstance(products, gpd.GeoDataFrame):
        gdf = products
    else:
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
        group_indices = _group_by_overlap(dfg)
        for i, g in enumerate(group_indices):
            sel = gdf.loc[g]
            geom = mapping(sel.unary_union)

            orbit_conf = sel.iloc[0]["orbitDirection"]
            orbit_num = sel.iloc[0]["relativeOrbitNumber"]
            folium_products = folium.GeoJson(
                geom, name=f"Orbit {orbit_num}, group {i+1} ({len(sel)})"
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


# make groups of almost fully overlapping products
def _group_by_overlap(dfg):
    init_list = list(dfg.index)
    groups = []
    while len(init_list) > 0:
        geom_a = dfg.geometry.loc[init_list[0]]
        grp = []
        grp.append(init_list.pop(0))
        i = 0
        while i < len(init_list):
            geom_b = dfg.geometry.loc[init_list[i]]
            if _has_overlap(geom_a, geom_b):
                grp.append(init_list.pop(i))
            else:
                i += 1
        groups.append(grp)
    return groups


# check for geometrical overlap
def _has_overlap(geom1, geom2, tolerance=0.01):
    intersection = geom1.intersection(geom2)
    return intersection.area / min(geom1.area, geom2.area) >= (1 - tolerance)
