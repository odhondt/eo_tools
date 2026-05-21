from pystac_client.client import Client
import geopandas as gpd
import numpy as np
import json
from dask.diagnostics import ProgressBar
from pathlib import Path
from urllib.parse import urlparse
import os
from datetime import datetime

from rasterio.session import AWSSession
from rasterio.windows import Window
import rasterio
import boto3

from eo_tools.auxils import get_burst_geometry
from eo_tools.S1.core import read_metadata

import logging
log = logging.getLogger(__name__)


def _write_partial_download_info(path, info):
    lines = [f"product_id: {json.dumps(info['product_id'])}", "subsets:"]
    for subswath, pols in info["subsets"].items():
        lines.append(f"  {subswath}:")
        for pol, subset in pols.items():
            lines.append(f"    {pol}:")
            lines.append(f"      file: {json.dumps(subset['file'])}")
            lines.append(f"      min_burst: {subset['min_burst']}")
            lines.append(f"      max_burst: {subset['max_burst']}")
            lines.append(f"      line_start: {subset['line_start']}")
            lines.append(f"      number_of_lines: {subset['number_of_lines']}")
            lines.append(f"      lines_per_burst: {subset['lines_per_burst']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# search with pystac client and store in a dataframe
# TODO: chack arg validity ()
def search_products(**kwargs):
    # Search using STAC api
    catalog = Client.open("https://stac.dataspace.copernicus.eu/v1/")
    search = catalog.search(**kwargs)  # , ids=ids)
    items = list(search.items())
    from shapely.geometry import shape

    start_times = []
    ids = []
    relative_orbits = []
    orbit_states = []
    geometries = []

    # convert to dict only if needed
    # items_ = [it.to_dict() for it in items]

    for it in items:
        it_dict = it.to_dict()
        props = it_dict["properties"]
        start_times.append(datetime.fromisoformat(props["start_datetime"]))
        ids.append(it.id)
        relative_orbits.append(props["sat:relative_orbit"])
        orbit_states.append(props["sat:orbit_state"])
        geometries.append(shape(it.geometry))

    # gdf = gpd.GeoDataFrame(data={"id": ids, "start_time": start_times, "orbit": relative_orbits, "orbit_state": orbit_states, "stac_item": items_}, geometry=geometries)
    # use columns compatible with util.explore_products
    gdf = gpd.GeoDataFrame(
        data={
            "id": ids,
            "startTimeFromAscendingNode": start_times,
            "relativeOrbitNumber": relative_orbits,
            "orbitDirection": orbit_states,
            "stac_item": items, # keep pystac objects
            # "stac_item": items_, # use dict instead
        },
        geometry=geometries,
    )
    return gdf


def download_partial_products(products, shp, out_dir, aws_key, aws_secret):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # rasterio session
    rio_session = AWSSession(
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name="default",
        endpoint_url="eodata.dataspace.copernicus.eu",
    )

    # needed for other (non-tiff) files
    # session = boto3.session.Session()
    s3 = boto3.resource(
        "s3",
        endpoint_url="https://eodata.dataspace.copernicus.eu",
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name="default",
    )
    for it in products.stac_item:
        log.info(f"Downloading {it.id}.")
        # product will be saved in this SAFE-like partial-product subdir
        product_root_dir = f"{it.id}.partial.SAFE"
        source_product_root_dir = f"{it.id}.SAFE"

        # use manifest file to get S3 bucket and prefix
        manifest_url = it.assets["safe_manifest"].href

        # Parse the url
        parsed = urlparse(manifest_url)
        if parsed.scheme != "s3":
            raise ValueError("Product url does not start with s3://")

        # Bucket is the "netloc"
        bucket_name = parsed.netloc

        # Look for subdir prefix
        manifest_path = Path(parsed.path.lstrip("/"))
        prefix = str(manifest_path.parent)

        # create product subdirectories
        subdirs = (
            "annotation",
            "measurement",
            "preview",
            "support",
            "annotation/rfi",
            "annotation/calibration",
            "preview/icons",
        )
        for subdir in subdirs:
            subpath = Path(out_dir) / product_root_dir / subdir
            if not os.path.isdir(subpath):
                os.makedirs(subpath)

        # find annotation files
        bucket = s3.Bucket(bucket_name)
        files = [it.key for it in list(bucket.objects.filter(Prefix=prefix))]

        # download
        for f in files:
            remote_file = f

            # remove all that is before the SAFE dir
            parts = Path(f).parts
            idx = parts.index(source_product_root_dir)
            # keep only the subdir (?)
            local_path = str(
                Path(out_dir)
                / product_root_dir
                / Path(*parts[idx + 1 :])
            )
            # skip raster files
            if Path(remote_file).suffix != ".tiff":
                bucket.download_file(remote_file, local_path)

        # retrieve burst geometries
        gdf_burst = get_burst_geometry(
            str(Path(out_dir) / product_root_dir),
            target_subswaths=["IW1", "IW2", "IW3"],
            polarization="VV",
        )

        # find what subswaths and bursts intersect AOI
        gdf_burst = gdf_burst[gdf_burst.intersects(shp)]

        if gdf_burst.empty:
            raise RuntimeError(
                "The list of bursts to process is empty. Make sure shp intersects the product."
            )

        # identify corresponding subswaths
        sel_subsw = gdf_burst["subswath"]
        unique_subswaths = np.unique(sel_subsw)
        partial_info = {"product_id": it.id, "subsets": {}}
        for pol in ["vv", "vh"]:
            for subswath in unique_subswaths:
                # use metadata to find where to crop
                str_xml = f"**/annotation/*{subswath.lower()}*{pol}*.xml"
                pth_xml = list((Path(out_dir) / product_root_dir).glob(str_xml))[0]
                meta = read_metadata(pth_xml=pth_xml)
                burst_info = meta["product"]["swathTiming"]
                lines_per_burst = int(burst_info["linesPerBurst"])
                burst_indices = gdf_burst[gdf_burst.subswath == subswath].burst
                min_burst = int(burst_indices.min())
                max_burst = int(burst_indices.max())
                line_start = lines_per_burst * (min_burst - 1)
                num_lines = lines_per_burst * (max_burst - min_burst + 1)

                log.info(
                    f"Polarization {pol}, Subswath {subswath}, "
                    f"Bursts {min_burst} to {max_burst}."
                )
                url = it.assets[f"{subswath.lower()}-{pol}"].href
                tiff_name = Path(url).name
                tiff_out = Path(out_dir) / product_root_dir / "measurement" / tiff_name

                with rasterio.Env(session=rio_session, AWS_VIRTUAL_HOSTING=False):
                    with rasterio.open(url) as src:
                        window = Window(0, line_start, src.width, num_lines)
                        profile = {
                            key: value
                            for key, value in src.profile.copy().items()
                            if "gcp" not in key.lower()
                        }
                        profile.update(
                            height=num_lines,
                            width=src.width,
                            transform=src.window_transform(window),
                        )
                        with rasterio.open(tiff_out, "w", **profile) as dst:
                            with ProgressBar():
                                dst.write(src.read(window=window))

                            for namespace in src.tag_namespaces():
                                dst.update_tags(ns=namespace, **src.tags(ns=namespace))
                            dst.update_tags(**src.tags())
                            for band_idx in src.indexes:
                                for namespace in src.tag_namespaces(band_idx):
                                    dst.update_tags(
                                        band_idx,
                                        ns=namespace,
                                        **src.tags(band_idx, ns=namespace),
                                    )
                                dst.update_tags(band_idx, **src.tags(band_idx))

                partial_info["subsets"].setdefault(subswath.lower(), {})[pol] = {
                    "file": f"measurement/{tiff_name}",
                    "min_burst": min_burst,
                    "max_burst": max_burst,
                    "line_start": int(line_start),
                    "number_of_lines": int(num_lines),
                    "lines_per_burst": int(lines_per_burst),
                }

        partial_file = Path(out_dir) / product_root_dir / "partial_download.yml"
        _write_partial_download_info(partial_file, partial_info)
