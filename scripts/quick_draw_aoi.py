# %%
import folium
import folium.plugins
from eo_tools_dev.util import serve_map

m = folium.Map()
folium.plugins.Draw(
    export=True,
    draw_options={
        "polyline": False,
        "circlemarker": False,
        "marker": False,
    },
).add_to(m)

# open in a browser
serve_map(m)