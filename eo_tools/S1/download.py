from pystac_client.client import Client
import geopandas as gpd
import rioxarray as riox
import xarray as xr
import numpy as np
import json
from dask.diagnostics import ProgressBar
from pathlib import Path
from urllib.parse import urlparse
import os

from rasterio.session import AWSSession
import rasterio
import boto3

from eo_tools.auxils import get_burst_geometry
from eo_tools.S1.core import read_metadata


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
    session = boto3.session.Session()
    s3 = boto3.resource(
        "s3",
        endpoint_url="https://eodata.dataspace.copernicus.eu",
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name="default",
    )
    for it in products:
        # product will be saved in this subdir
        product_root_dir = f"{it.id}.SAFE"

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
            idx = parts.index(product_root_dir)
            # keep only the subdir (?)
            local_path = str(Path(out_dir) / Path(*parts[idx:]))  # .parent
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

        # TODO: loop on polarizations and iw
        iw = "3"
        pol = "vv"

        if gdf_burst.empty:
            raise RuntimeError(
                "The list of bursts to process is empty. Make sure shp intersects the product."
            )

        # identify corresponding subswaths
        sel_subsw = gdf_burst["subswath"]
        unique_subswaths = np.unique(sel_subsw)
        for pol in ["vv", "vh"]:
            for subswath in unique_subswaths:
                # use metadata to find where to crop
                str_xml = f"**/annotation/*{subswath.lower()}*{pol}*.xml"
                pth_xml = list((Path(out_dir) / product_root_dir).glob(str_xml))[0]
                meta = read_metadata(pth_xml=pth_xml)
                burst_info = meta["product"]["swathTiming"]
                lines_per_burst = int(burst_info["linesPerBurst"])
                burst_indices = gdf_burst[gdf_burst.subswath == f"IW{iw}"].burst
                min_burst = burst_indices.min()
                max_burst = burst_indices.max()
                line_start = lines_per_burst * (min_burst - 1)
                num_lines = lines_per_burst * (max_burst - min_burst + 1)
                line_end = line_start + num_lines

                # open raster
                url = it.assets[f"iw{iw}-{pol}"].href
                # !! Important: open as a dataset (band_as_variable=True)
                with rasterio.Env(session=rio_session, AWS_VIRTUAL_HOSTING=False):
                    ds = riox.open_rasterio(url, chunks="auto", band_as_variable=True)

                # Complex not handled (yet) in zarr
                ds["i"] = ds.band_1.real
                ds["q"] = ds.band_1.imag
                del ds["band_1"]

                ds.attrs["min_burst"] = min_burst
                ds.attrs["max_burst"] = max_burst

                # download cropped array
                zarr_name = f"{Path(url).stem}.zarr"
                zarr_out = Path(out_dir) / product_root_dir / "measurement" / zarr_name
                with ProgressBar():
                    ds.isel(y=slice(line_start, line_end)).chunk("auto").to_zarr(
                        zarr_out, mode="w"
                    )
