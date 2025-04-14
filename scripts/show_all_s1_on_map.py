# %%
from folium import Map, GeoJson, Tooltip, LayerControl, Popup
import zipfile
from pathlib import Path
from shapely.geometry import Polygon, mapping
from eo_tools_dev.util import serve_map

# change to you directory
data_dir = "/data/S1"


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


m = Map()
for f in list(Path(data_dir).glob("S1*")):
    pth = zipfile.Path(f) if Path(f).suffix == ".zip" else Path(f)
    manifest = list(pth.glob("**/manifest.safe"))[0]
    footprint = extract_footprint(manifest)

    gj = GeoJson(mapping(footprint), name=f.stem)
    gj.add_child(Popup(f.stem))
    gj.add_to(m)

LayerControl().add_to(m)
serve_map(m)

# %%
