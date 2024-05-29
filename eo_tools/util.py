import folium

from shapely.geometry import mapping

import os
import numpy as np
import geopandas as gpd
import json
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_hex
from eo_tools.S2 import make_s2_rgb, make_s2_color
import httpx

import subprocess
import signal
import os
import time
import socket


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


class TileServerManager:
    """Start / stop the Titiler app."""

    _ENV_VARIABLE = "TILE_SERVER_PID"

    @classmethod
    def start(cls, port=8085, timeout=30):
        # Check if the server is already running
        if not cls._get_server_pid():
            try:
                # Check if the port is already in use
                if cls._is_port_in_use(port):
                    raise RuntimeError(
                        f"Port {port} is already in use by another application."
                    )

                # Start the server
                env = os.environ.copy()
                process = subprocess.Popen(
                    [
                        "uvicorn",
                        "titiler.application.main:app",
                        "--host",
                        "127.0.0.1",
                        f"--port={port}",
                        "--log-level",
                        "info",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid,
                    shell=False,
                    env=env,
                )

                start_time = time.time()
                while time.time() - start_time < timeout:
                    if process.poll() is not None:
                        # If the process has exited, raise an error
                        raise RuntimeError(
                            f"Failed to start server on port {port}. Error: {process.stderr.read().decode()}"
                        )

                    if cls._is_port_in_use(port):
                        # Server is up and running
                        cls._set_server_pid(process.pid)
                        print(f"Server started with PID: {process.pid} on port {port}")
                        return
                    time.sleep(1)

                # Timeout reached, server didn't start
                process.terminate()
                raise RuntimeError(
                    f"Timeout reached. Server failed to start on port {port} within {timeout} seconds."
                )

            except Exception as e:
                print(f"Error starting server: {e}")
                # re-raise for pytest
                raise
        else:
            raise RuntimeError(f"Server is already running on port {port}.")

    @classmethod
    def stop(cls):
        # Get the server PID
        pid = cls._get_server_pid()
        if pid:
            # Stop the server process using the PID
            try:
                os.kill(int(pid), signal.SIGTERM)
                print(f"Server with PID {pid} stopped.")
            except OSError as e:
                print(f"Error stopping server with PID {pid}: {e}")
            # Clear the PID from the environment variable
            cls._clear_server_pid()
            time.sleep(1)
        else:
            print("Server is not running.")

    @classmethod
    def _get_server_pid(cls):
        # Get the server PID from the environment variable
        return os.getenv(cls._ENV_VARIABLE)

    @classmethod
    def _set_server_pid(cls, pid):
        # Set the server PID in the environment variable
        os.environ[cls._ENV_VARIABLE] = str(pid)

    @classmethod
    def _clear_server_pid(cls):
        # Clear the server PID from the environment variable
        os.environ.pop(cls._ENV_VARIABLE, None)

    @staticmethod
    def _is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) == 0


def ttcog_get_stats(url, **kwargs):
    if "port" in kwargs.keys():
        port = kwargs["port"]
    else:
        port = 8085
    titiler_endpoint = f"http://localhost:{port}"
    r = httpx.get(
        f"{titiler_endpoint}/cog/statistics",
        params={"url": url, **kwargs},
    ).json()
    return r


def ttcog_get_info(url, port=8085):
    titiler_endpoint = f"http://localhost:{port}"
    r = httpx.get(
        f"{titiler_endpoint}/cog/info",
        params={
            "url": url,
        },
    ).json()
    return r


def ttcog_get_tilejson(url, **kwargs):
    if "port" in kwargs.keys():
        port = kwargs["port"]
    else:
        port = 8085
    titiler_endpoint = f"http://localhost:{port}"
    r = httpx.get(
        f"{titiler_endpoint}/cog/tilejson.json", params={"url": url, **kwargs}
    ).json()
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
        print("dir")
        file_in = f"{input_path}/phi.tif"
    elif os.path.isfile(input_path):
        print("file")
        file_in = input_path
    else:
        raise FileExistsError(f"Problem reading file.")

    if not os.path.isfile(file_in):
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

    info = ttcog_get_info(file_in, port=port)
    bounds = info["bounds"]

    eps = np.random.uniform(1e-8, 1e-9)
    tjson = ttcog_get_tilejson(
        file_in,
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
        file_in = f"{input_path}/coh.tif"
    elif os.path.isfile(input_path):
        file_in = input_path
    else:
        raise FileExistsError("Problem reading file or file does not exist.")

    if not os.path.isfile(file_in):
        raise FileExistsError("Problem reading file or file does not exist.")
    info = ttcog_get_info(file_in, port)
    bounds = info["bounds"]
    eps = np.random.uniform(1e-8, 1e-9)
    tjson = ttcog_get_tilejson(file_in, port=port, rescale=f"{0+eps},1")

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
            file_in = f"{input_path}/int_mst.tif"
        else:
            file_in = f"{input_path}/int_slv.tif"
    elif os.path.isfile(input_path):
        file_in = input_path
    else:
        raise FileExistsError("Problem reading file or file does not exist.")

    if not os.path.isfile(file_in):
        raise FileExistsError("Problem reading file or file does not exist.")

    if not dB:
        stats = ttcog_get_stats(file_in, port=port)["b1"]
    else:
        stats = ttcog_get_stats(file_in, port=port, expression="10*log10(b1)")[
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
    info = ttcog_get_info(file_in, port=port)
    bounds = info["bounds"]
    eps = np.random.uniform(1e-8, 1e-9)
    if dB:
        tjson = ttcog_get_tilejson(
            file_in,
            port=port,
            rescale=f"{vmin_+eps},{vmax_}",
            expression="10*log10(b1)",
        )
    else:
        tjson = ttcog_get_tilejson(file_in, port=port, rescale=f"{vmin_+eps},{vmax_}")

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
    # if "port" in kwargs.keys():
    #     port = kwargs["port"]
    # else:
    #     port = 8085
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


# TODO: Other band combinations ( NDVI, ...)
# TODO: Viz InSAR composite (HSV)
