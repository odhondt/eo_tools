import json
import logging
import os
import warnings
from datetime import datetime as DateTime
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import boto3
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import yaml
from dask.diagnostics import ProgressBar
from pystac_client.client import Client
from rasterio.session import AWSSession
from rasterio.windows import Window
from shapely.geometry import Polygon, mapping, shape

from eo_tools.auxils import get_burst_geometry, remove
from eo_tools.S1.core import read_metadata

log = logging.getLogger(__name__)
PARTIAL_AOI_FILENAME = "partial_aoi.geojson"
S1_SLC_COLLECTION = ["sentinel-1-slc"]

def search_products(
    intersects: Polygon,
    datetime: Any | None = None,
    ids: Sequence[str] | None = None,
) -> gpd.GeoDataFrame:
    """Search CDSE for Sentinel-1 SLC products intersecting one polygon.

    The returned frame keeps the original pystac item objects in the stac_item
    column so downstream download helpers can access assets and metadata without
    repeating the catalog lookup. The collection is fixed to Sentinel-1 SLC
    because other product families are not valid inputs to the partial SLC
    downloader.

    Args:
        intersects: Single polygon AOI products must intersect.
        datetime: Optional STAC datetime/range filter. Required when ids is not
            set.
        ids: Optional product identifier sequence. Required when datetime is
            not set.

    Returns:
        A GeoDataFrame containing product metadata and geometry.

    Raises:
        ValueError: If intersects is not one Polygon, or neither datetime nor
            a non-empty ids sequence is provided.
    """
    intersects = _validate_single_polygon(intersects, "intersects")
    if ids is not None:
        if (
            isinstance(ids, str)
            or not ids
            or not all(isinstance(it, str) for it in ids)
        ):
            raise ValueError("ids must be a non-empty sequence of product ID strings.")
    if datetime is None and ids is None:
        raise ValueError("At least one of datetime or ids must be provided.")

    search_params: dict[str, Any] = {
        "collections": S1_SLC_COLLECTION,
        "intersects": intersects,
    }
    if datetime is not None:
        search_params["datetime"] = datetime
    if ids is not None:
        search_params["ids"] = ids

    # Search using STAC api
    catalog = Client.open("https://stac.dataspace.copernicus.eu/v1/")
    search = catalog.search(**search_params)
    items = list(search.items())
    start_times = []
    result_ids = []
    relative_orbits = []
    orbit_states = []
    geometries = []

    # convert to dict only if needed
    # items_ = [it.to_dict() for it in items]

    for it in items:
        it_dict = it.to_dict()
        props = it_dict["properties"]
        start_times.append(DateTime.fromisoformat(props["start_datetime"]))
        result_ids.append(it.id)
        relative_orbits.append(props["sat:relative_orbit"])
        orbit_states.append(props["sat:orbit_state"])
        geometries.append(shape(it.geometry))

    # gdf = gpd.GeoDataFrame(data={"id": ids, "start_time": start_times, "orbit": relative_orbits, "orbit_state": orbit_states, "stac_item": items_}, geometry=geometries)
    # use columns compatible with util.explore_products
    gdf = gpd.GeoDataFrame(
        data={
            "id": result_ids,
            "startTimeFromAscendingNode": start_times,
            "relativeOrbitNumber": relative_orbits,
            "orbitDirection": orbit_states,
            "stac_item": items,  # keep pystac objects
            # "stac_item": items_, # use dict instead
        },
        geometry=geometries,
    )
    return gdf


def download_partial_products(
    products: gpd.GeoDataFrame,
    shp: Any,
    out_dir: str | Path,
    aws_key: str,
    aws_secret: str,
    pol: str | Sequence[str] = "full",
    force_overwrite: bool = False,
) -> None:
    """Download cropped Sentinel-1 SAFE-like partial products from CDSE S3.

    For each STAC product, the function copies the non-measurement SAFE structure,
    crops the measurement TIFFs to the bursts intersecting shp, and writes a
    partial_download.yml manifest and a required partial_aoi.geojson file that
    records the AOI used to select bursts.

    Args:
        products: Search result table returned by search_products.
        shp: Area of interest used to select intersecting bursts and written to
            each partial product as required GeoJSON metadata.
        out_dir: Output directory that will receive <product>.partial.SAFE.
        aws_key: Copernicus Data Space S3 access key.
        aws_secret: Copernicus Data Space S3 secret key.
        pol: Polarization selection. Accepts "vv", "vh", "full", or a
            sequence containing "vv" and/or "vh". Defaults to "full".
        force_overwrite: When True, remove and re-download an existing partial
            product directory. When False, existing directories are left
            untouched. Defaults to False.

    Returns:
        None.

    Note:
        Existing partial-product directories are not integrity-checked before
        being skipped. Users are responsible for verifying existing partial
        products; when in doubt, use force_overwrite=True to remove and
        re-download them.
    """
    shp = _validate_single_polygon(shp, "shp")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    rio_session = AWSSession(
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name="default",
        endpoint_url="eodata.dataspace.copernicus.eu",
    )

    s3 = boto3.resource(
        "s3",
        endpoint_url="https://eodata.dataspace.copernicus.eu",
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name="default",
    )
    selected_pols = _normalize_polarizations(pol)
    log.info("Selected polarizations: %s", ", ".join(p.upper() for p in selected_pols))
    geometry_pol = selected_pols[0]

    for it in products.stac_item:
        log.info(f"Downloading {it.id}.")
        product_dir, _, source_product_root_dir = _get_partial_product_paths(
            out_dir, it.id
        )
        if product_dir.exists() and not product_dir.is_dir():
            raise FileExistsError(
                f"Cannot create partial product directory because a non-directory "
                f"path already exists: {product_dir}"
            )
        if product_dir.is_dir():
            log.warning("Partial product directory already exists: %s", product_dir)
            if not force_overwrite:
                log.warning(
                    "Skipping %s. Use force_overwrite=True to remove and "
                    "re-download this partial product.",
                    it.id,
                    stacklevel=1,
                )
                continue
            log.warning(
                "force_overwrite=True; removing existing partial product directory "
                "and re-downloading %s.",
                it.id,
                stacklevel=1,
            )
            remove(product_dir)

        bucket_name, prefix = _get_partial_product_source(it.assets["safe_manifest"].href)
        bucket = s3.Bucket(bucket_name)

        log.info("Creating partial product structure for %s", it.id)
        _create_partial_product_subdirs(product_dir)
        log.info("Write annotation files for %s", it.id)
        _download_metadata_files(
            bucket=bucket,
            prefix=prefix,
            product_dir=product_dir,
            source_product_root_dir=source_product_root_dir,
        )

        partial_info, download_jobs = _build_download_list(
            product_dir=product_dir,
            stac_item=it,
            selected_pols=selected_pols,
            geometry_pol=geometry_pol,
            shp=shp,
        )
        for job in download_jobs:
            log.info(
                "Downloading partial raster for %s / %s, burst %s to %s.",
                job["subswath"],
                job["pol"],
                job["min_burst"],
                job["max_burst"],
            )
            _download_partial_raster_files(
                url=job["url"],
                tiff_out=job["tiff_out"],
                line_start=job["line_start"],
                num_lines=job["num_lines"],
                rio_session=rio_session,
            )

        partial_file = product_dir / "partial_download.yml"
        log.info("Create partial download manifest %s", partial_file.name)
        _write_partial_download_info(partial_file, partial_info)
        aoi_file = product_dir / PARTIAL_AOI_FILENAME
        log.info("Create partial product AOI metadata %s", aoi_file.name)
        _write_partial_aoi(aoi_file, shp)


def _write_partial_download_info(path: Path, info: dict[str, Any]) -> None:
    """Write the generated partial-download manifest to disk."""
    with path.open("w", encoding="utf-8") as f:
        f.write("# This file is automatically generated. Do not edit manually.\n")
        yaml.safe_dump(
            info,
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=False,
        )


def _write_partial_aoi(path: Path, shp: Any) -> None:
    """Write the AOI used to build a partial product as GeoJSON metadata."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": mapping(shp),
            }
        ],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)
        f.write("\n")


def _validate_single_polygon(shp: Any, parameter_name: str) -> Polygon:
    """Validate that an AOI is represented by exactly one Polygon."""
    if not isinstance(shp, Polygon):
        raise ValueError(
            f"{parameter_name} must be a single shapely Polygon, not "
            f"{getattr(shp, 'geom_type', type(shp).__name__)}."
        )
    if shp.is_empty:
        raise ValueError(f"{parameter_name} must not be an empty Polygon.")
    return shp


def _normalize_polarizations(pol: str | Sequence[str]) -> list[str]:
    """Normalize a polarization selection to a validated list."""
    allowed = ("vv", "vh")
    if isinstance(pol, str):
        if pol.lower() == "full":
            selected = list(allowed)
        else:
            selected = [pol.lower()]
    elif isinstance(pol, (list, tuple, set)):
        selected = [p.lower() for p in pol]
    else:
        raise ValueError("pol must be a string or an iterable of strings")

    invalid = sorted(set(selected) - set(allowed))
    if invalid:
        raise ValueError(
            f"Invalid polarization(s): {invalid}. Allowed values are 'vv', 'vh', 'full', or ['vv', 'vh']."
        )

    selected = [p for p in allowed if p in set(selected)]
    if not selected:
        raise ValueError("At least one valid polarization must be selected")
    return selected


def _get_partial_product_paths(out_dir: str | Path, product_id: str) -> tuple[Path, str, str]:
    """Return the target directory and SAFE root names for a partial product."""
    out_dir = Path(out_dir)
    product_root_dir = f"{product_id}.partial.SAFE"
    product_dir = out_dir / product_root_dir
    source_product_root_dir = f"{product_id}.SAFE"
    return product_dir, product_root_dir, source_product_root_dir


def _get_partial_product_source(manifest_url: str) -> tuple[str, str]:
    """Extract the S3 bucket and prefix from a Sentinel-1 manifest URL."""
    parsed = urlparse(manifest_url)
    if parsed.scheme != "s3":
        raise ValueError("Product url does not start with s3://")

    bucket_name = parsed.netloc
    manifest_path = Path(parsed.path.lstrip("/"))
    prefix = str(manifest_path.parent)
    return bucket_name, prefix


def _create_partial_product_subdirs(product_dir: Path) -> None:
    """Create the SAFE-like directory structure used by a partial product."""
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
        subpath = product_dir / subdir
        if not os.path.isdir(subpath):
            os.makedirs(subpath)


def _download_metadata_files(
    bucket: Any,
    prefix: str,
    product_dir: Path,
    source_product_root_dir: str,
) -> None:
    """Download annotation and other non-measurement sidecar files into the partial product tree."""
    files = [it.key for it in list(bucket.objects.filter(Prefix=prefix))]

    for remote_file in files:
        # S3 "folder" markers end with "/" and must not be downloaded as files.
        if remote_file.endswith("/"):
            log.debug("Skipping directory marker key %s", remote_file)
            continue
        parts = Path(remote_file).parts
        idx = parts.index(source_product_root_dir)
        relative_parts = parts[idx + 1 :]
        if not relative_parts:
            log.debug("Skipping directory marker key %s", remote_file)
            continue
        local_path = str(product_dir / Path(*relative_parts))
        if Path(remote_file).suffix != ".tiff":
            bucket.download_file(remote_file, local_path)


def _build_download_list(
    product_dir: Path,
    stac_item: Any,
    selected_pols: Sequence[str],
    geometry_pol: str,
    shp: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the partial-product manifest data and per-measurement download list."""
    gdf_burst = get_burst_geometry(
        str(product_dir),
        target_subswaths=["IW1", "IW2", "IW3"],
        polarization=geometry_pol.upper(),
    )

    gdf_burst = gdf_burst[gdf_burst.intersects(shp)]
    if gdf_burst.empty:
        raise RuntimeError(
            "The list of bursts to process is empty. Make sure shp intersects the product."
        )

    selected_subswaths = np.unique(gdf_burst["subswath"])
    partial_info: dict[str, Any] = {
        "product_id": stac_item.id,
        "aoi_file": PARTIAL_AOI_FILENAME,
        "subsets": {},
    }
    download_jobs: list[dict[str, Any]] = []

    for pol in selected_pols:
        for subswath in selected_subswaths:
            str_xml = f"**/annotation/*{subswath.lower()}*{pol}*.xml"
            pth_xml = list(product_dir.glob(str_xml))[0]
            meta = read_metadata(pth_xml=pth_xml)
            burst_info = meta["product"]["swathTiming"]
            lines_per_burst = int(burst_info["linesPerBurst"])
            burst_indices = gdf_burst[gdf_burst.subswath == subswath].burst
            min_burst = int(burst_indices.min())
            max_burst = int(burst_indices.max())
            line_start = lines_per_burst * (min_burst - 1)
            num_lines = lines_per_burst * (max_burst - min_burst + 1)
            url = stac_item.assets[f"{subswath.lower()}-{pol}"].href
            tiff_name = Path(url).name
            partial_info["subsets"].setdefault(subswath.lower(), {})[pol] = {
                "file": f"measurement/{tiff_name}",
                "min_burst": min_burst,
                "max_burst": max_burst,
                "line_start": int(line_start),
                "number_of_lines": int(num_lines),
                "lines_per_burst": int(lines_per_burst),
            }
            download_jobs.append(
                {
                    "url": url,
                    "tiff_out": product_dir / "measurement" / tiff_name,
                    "line_start": int(line_start),
                    "num_lines": int(num_lines),
                    "subswath": subswath.upper(),
                    "pol": pol.upper(),
                    "min_burst": min_burst,
                    "max_burst": max_burst,
                }
            )

    return partial_info, download_jobs


def _download_partial_raster_files(
    url: str,
    tiff_out: Path,
    line_start: int,
    num_lines: int,
    rio_session: AWSSession,
) -> None:
    """Download and write one cropped measurement TIFF for a partial product."""
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
            # These tiling options are not always compatible with cropped
            # outputs, so drop them unless we explicitly re-tile the file.
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
            profile.pop("tiled", None)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=NotGeoreferencedWarning
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
