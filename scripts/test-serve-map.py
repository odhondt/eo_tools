import tempfile

import folium
import numpy as np
import rasterio
from affine import Affine

from eo_tools_dev.util import serve_map, show_cog


# Keep the file in the shared data directory so a containerized TiTiler can access it.
with tempfile.TemporaryDirectory(dir="/data/tmp") as tmp_dir:
    tiff_path = f"{tmp_dir}/trig-rgb.tif"
    height, width = 512, 512
    y, x = np.mgrid[0:height, 0:width]
    red = ((np.sin(x / 10) + 1) * 127.5).astype(np.uint8)
    green = ((np.sin(y / 12) + 1) * 127.5).astype(np.uint8)
    blue = ((np.sin((x + y) / 15) + 1) * 127.5).astype(np.uint8)

    with rasterio.open(
        tiff_path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=Affine.translation(1.0, 47.8)
        * Affine.rotation(12)
        * Affine.scale(0.005, -0.005),
    ) as dataset:
        dataset.write(red, 1)
        dataset.write(green, 2)
        dataset.write(blue, 3)

    map_object = folium.Map(location=[46.6, 2.3], zoom_start=5)
    show_cog(tiff_path, folium_map=map_object, rescale="0,255")
    serve_map(map_object)
