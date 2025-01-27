import folium
import tempfile
from http.server import SimpleHTTPRequestHandler, HTTPServer
import webbrowser
import os
import numpy as np
import json
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_hex
from eo_tools.S2 import make_s2_rgb, make_s2_color
import httpx


def ttcog_get_stats(url, **kwargs):
    if "port" in kwargs.keys():
        port = kwargs["port"]
    else:
        port = 8085
    titiler_endpoint = f"http://localhost:{port}"
    try:
        r = httpx.get(
            f"{titiler_endpoint}/cog/statistics",
            params={"url": url, **kwargs},
        ).json()
    except:
        raise RuntimeError(
            "Server not running. Please run `uvicorn titiler.application.main:app --host 127.0.0.1 --port=8085` in a terminal to use this function."
        )
    return r


def ttcog_get_info(url, port=8085):
    titiler_endpoint = f"http://localhost:{port}"
    try:
        r = httpx.get(
            f"{titiler_endpoint}/cog/info",
            params={
                "url": url,
            },
        ).json()
    except:
        raise RuntimeError(
            "Server not running. Please run `uvicorn titiler.application.main:app --host 127.0.0.1 --port=8085` in a terminal to use this function."
        )
    return r


def ttcog_get_tilejson(url, **kwargs):
    if "port" in kwargs.keys():
        port = kwargs["port"]
    else:
        port = 8085
    titiler_endpoint = f"http://localhost:{port}"
    try:
        r = httpx.get(
            f"{titiler_endpoint}/cog/WebMercatorQuad/tilejson.json",
            params={"url": url, **kwargs},
        ).json()
    except:
        raise RuntimeError(
            "Server not running. Please run `uvicorn titiler.application.main:app --host 127.0.0.1 --port=8085` in a terminal to use this function."
        )
    return r


def palette_phi():
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
    return json.dumps({x: y for x, y in zip(range(256), cmap_hex)})


def show_insar_phi(input_path, port=8085):
    # def visualize_insar_phase(input_path):
    """Visualize interferometric phase on a map with a cyclic colormap (similar to SNAP).

    Args:
        input_path (str): Directory with InSAR products (phi.tif) or GeoTiff input file (preferably COG).

    Returns:
        folium.Map: raster visualization on an interactive map
    """
    if os.path.isdir(input_path):
        in_file = f"{input_path}/phi.tif"
    elif os.path.isfile(input_path):
        in_file = input_path
    else:
        raise FileExistsError(f"Problem reading file.")

    if not os.path.isfile(in_file):
        raise FileExistsError("Problem reading file or file does not exist.")

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

    info = ttcog_get_info(in_file, port=port)
    bounds = info["bounds"]

    eps = np.random.uniform(1e-8, 1e-9)
    tjson = ttcog_get_tilejson(
        in_file,
        port=port,
        rescale=f"{-np.pi+eps},{np.pi}",
        resampling="nearest",  # make sure COG has been made with 'nearest'
        colormap=json.dumps({x: y for x, y in zip(range(256), cmap_hex)}),
    )

    m = folium.Map(
        location=((bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2),
        zoom_start=8,
    )

    folium.TileLayer(tiles=tjson["tiles"][0], attr="InSAR phase").add_to(m)
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    return m


def show_insar_coh(input_path, port=8085):
    """Visualize coherence on a map.

    Args:
        input_path (str): Directory with InSAR products (coh.tif) or GeoTiff input file (preferably COG).

    Returns:
        folium.Map: raster visualization on an interactive map
    """

    if os.path.isdir(input_path):
        in_file = f"{input_path}/coh.tif"
    elif os.path.isfile(input_path):
        in_file = input_path
    else:
        raise FileExistsError("Problem reading file or file does not exist.")

    if not os.path.isfile(in_file):
        raise FileExistsError("Problem reading file or file does not exist.")
    info = ttcog_get_info(in_file, port)
    bounds = info["bounds"]
    eps = np.random.uniform(1e-8, 1e-9)
    tjson = ttcog_get_tilejson(in_file, port=port, rescale=f"{0+eps},1")

    m = folium.Map(
        location=((bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2),
        zoom_start=8,
    )

    folium.TileLayer(tiles=tjson["tiles"][0], attr="InSAR Coherence").add_to(m)
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    return m


def show_sar_int(input_path, master=True, vmin=None, vmax=None, dB=False, port=8085):
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
            in_file = f"{input_path}/int_mst.tif"
        else:
            in_file = f"{input_path}/int_slv.tif"
    elif os.path.isfile(input_path):
        in_file = input_path
    else:
        raise FileExistsError("Problem reading file or file does not exist.")

    if not os.path.isfile(in_file):
        raise FileExistsError("Problem reading file or file does not exist.")

    if not dB:
        stats = ttcog_get_stats(in_file, port=port)["b1"]
    else:
        stats = ttcog_get_stats(in_file, port=port, expression="10*log10(b1)")[
            "10*log10(b1)"
        ]

    if vmin is None:
        vmin_ = float(stats["percentile_2"])
    else:
        vmin_ = vmin

    if vmax is None:
        vmax_ = float(stats["percentile_98"])
    else:
        vmax_ = vmax
    info = ttcog_get_info(in_file, port=port)
    bounds = info["bounds"]
    eps = np.random.uniform(1e-8, 1e-9)
    if dB:
        tjson = ttcog_get_tilejson(
            in_file,
            port=port,
            rescale=f"{vmin_+eps},{vmax_}",
            expression="10*log10(b1)",
        )
    else:
        tjson = ttcog_get_tilejson(in_file, port=port, rescale=f"{vmin_+eps},{vmax_}")

    m = folium.Map(
        location=((bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2),
        # zoom_start=8,
    )

    tt = folium.TileLayer(tiles=tjson["tiles"][0], attr="SAR Intensity")
    tt.add_to(m)

    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    return m


def show_s2_rgb(input_dir, force_create=False, port=8085):
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

    info = ttcog_get_info(rgb_path, port=port)
    bounds = info["bounds"]
    tjson = ttcog_get_tilejson(rgb_path, port=port)

    m = folium.Map(
        location=((bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2),
    )

    folium.TileLayer(tiles=tjson["tiles"][0], attr="RGB raster").add_to(m)
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    return m


def show_s2_color(input_dir, name="RGB", force_create=False, port=8085):
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

    info = ttcog_get_info(im_path, port=port)
    bounds = info["bounds"]
    tjson = ttcog_get_tilejson(im_path, port=port)

    m = folium.Map(
        location=((bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2),
    )

    folium.TileLayer(tiles=tjson["tiles"][0], attr="RGB raster").add_to(m)
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    return m


def show_cog(url, folium_map=None, port=8085, **kwargs):
    """Low-level function to show local or remote raster COG on a folium map.

    Args:
        url (str): remote url or local path
        kwargs: extra-arguments to be passed to the TiTiler cog endpoint

    Returns:
        folium.Map: raster visualization on an interactive map
    """
    # workaround to enforce GDAL not using VSI cache (otherwise preview may not be updated)
    if "rescale" in kwargs:
        low, high = kwargs["rescale"].split(",")
        low = float(low)
        high = float(high)
        eps = np.random.uniform(1e-8, 1e-9)  # Generating a small random number
        kwargs["rescale"] = f"{low + eps}, {high}"
    else:
        raise ValueError(
            "Missing 'rescale' argument. Please provide as follows: rescale='low_value, high_value'"
        )

    info = ttcog_get_info(url, port=port)
    bounds = info["bounds"]
    tjson = ttcog_get_tilejson(url, port=port, **kwargs)

    if folium_map is None:
        m = folium.Map(
            location=((bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2),
        )
    else:
        m = folium_map

    folium.TileLayer(
        tiles=tjson["tiles"][0], attr="COG", overlay=True, name=url
    ).add_to(m)
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    return m


def serve_map(map_object, port=8000):
    """Opens folium map in a browser by starting a local server.

    Args:
        map_object (folim.Map): map to display
        port (int, optional): port on localhost. Defaults to 8000.
    """
    # temporary file to store the html map
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
        map_filename = tmp_file.name
        map_dir = os.path.dirname(map_filename)  # Directory of the temp file
        map_object.save(map_filename)

    def start_server():
        os.chdir(map_dir)

        handler = SimpleHTTPRequestHandler
        httpd = HTTPServer(("", port), handler)
        print(
            f"Serving map on http://localhost:{port}/{os.path.basename(map_filename)}"
        )

        # Open the map in the default web browser
        webbrowser.open(f"http://localhost:{port}/{os.path.basename(map_filename)}")

        # hit ctrl-c to serve
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped by user.")
        finally:
            # delete the temporary file after server stops
            os.remove(map_filename)
            httpd.server_close()
            print("Temporary file deleted and server closed.")

    start_server()


# TODO: Other band combinations ( NDVI, ...)
# TODO: Viz InSAR composite (HSV)
