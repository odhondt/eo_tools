# %%
from folium import Map, GeoJson, Tooltip, LayerControl, Popup
import json
import zipfile
from pathlib import Path
from shapely.geometry import Polygon, mapping
from eo_tools_dev.util import serve_map

# change to you directory
data_dir = "/data/S1"
aoi_dir = Path(__file__).resolve().parent / "../data"


def extract_footprint(manifest):
    footprint_coords = None
    with manifest.open() as file:
        for line in file:
            if "<gml:coordinates>" in line:
                # Extract coordinates between the tags
                start = line.find("<gml:coordinates>") + len("<gml:coordinates>")
                end = line.find("</gml:coordinates>")
                footprint_coords = line[start:end].strip()
                break

    # Convert the coordinates to a valid Shapely Polygon
    if footprint_coords:
        coord_pairs = footprint_coords.split()
        coordinates = [
            (float(c2), float(c1))
            for c1, c2 in (pair.split(",") for pair in coord_pairs)
        ]
        return Polygon(coordinates)
    else:
        raise RuntimeError("No footprint found!")


def add_geojson_files_to_map(m, folder):
    folder = Path(folder)
    if not folder.exists():
        return

    for geojson_file in sorted(folder.glob("*.geojson")):
        with geojson_file.open() as file:
            geojson_data = json.load(file)

        gj = GeoJson(
            geojson_data,
            name=geojson_file.name,
            style_function=lambda feature: {
                "fillColor": "black",
                "color": "black",
                "weight": 2,
                "fillOpacity": 0.0,
            },
        )
        gj.add_child(Popup(f"Filename: {geojson_file.name}"))
        gj.add_to(m)


def add_partial_products_to_map(m, folder):
    folder = Path(folder)
    if not folder.exists():
        return

    for f in list(folder.glob("*.SAFE")):
        pth = zipfile.Path(f) if Path(f).suffix == ".zip" else Path(f)
        manifest = list(pth.glob("**/manifest.safe"))[0]
        footprint = extract_footprint(manifest)

        gj = GeoJson(
            mapping(footprint),
            name=f"partial {f.stem}",
            style_function=lambda feature: {
                "fillColor": "pink",
                "color": "hotpink",
                "weight": 2,
                "fillOpacity": 0.15,
            },
        )
        gj.add_child(Popup(f"Product ID: partial {f.stem}<br>Path: {str(f)}"))
        gj.add_to(m)


m = Map()
for f in list(Path(data_dir).glob("S1*")):
    pth = zipfile.Path(f) if Path(f).suffix == ".zip" else Path(f)
    manifest = list(pth.glob("**/manifest.safe"))[0]
    footprint = extract_footprint(manifest)

    gj = GeoJson(mapping(footprint), name=f.stem)
    gj.add_child(Popup(f"Product ID: {f.stem}<br>Path: {str(f)}"))
    gj.add_to(m)

add_partial_products_to_map(m, Path(data_dir) / "partial_dls")
add_geojson_files_to_map(m, aoi_dir)

LayerControl().add_to(m)
serve_map(m)

# %%
