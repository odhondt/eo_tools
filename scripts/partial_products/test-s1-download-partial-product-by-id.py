"""Re-download Sentinel-1 SLC products as partial SAFE directories by ID."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd

from eo_tools.S1.download import download_partial_products, search_products

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

data_dir = Path("/data/S1/partial_dls/")
creds_file = Path("/data/creds_s3.json")

jobs = [
    (
        "/eo_tools/data/Etna.geojson",
        [
            "S1A_IW_SLC__1SDV_20181228T050448_20181228T050515_025221_02C9BE_1E22",
            "S1B_IW_SLC__1SDV_20181222T050400_20181222T050427_014150_01A4AE_4A45",
        ],
    ),
    (
        "/eo_tools/data/Andorra_tiny.geojson",
        [
            "S1C_IW_SLC__1SDV_20260420T174607_20260420T174635_007303_00ECDF_FA97",
            "S1D_IW_SLC__1SDV_20260421T174617_20260421T174645_002448_00405E_4FF4",
        ],
    ),
    (
        "/eo_tools/data/Berlin.geojson",
        [
            "S1A_IW_SLC__1SDV_20240916T165234_20240916T165301_055693_06CD1B_C701",
            "S1A_IW_SLC__1SDV_20160922T165150_20160922T165217_013168_014EE2_7902",
        ],
    ),
]


def main() -> None:
    with creds_file.open(encoding="utf-8") as f:
        cred = json.load(f)

    # data_dir.mkdir(parents=True, exist_ok=True)

    for aoi_file, product_ids in jobs:
        shp = gpd.read_file(aoi_file).geometry[0]
        products = search_products(intersects=shp, ids=product_ids)

        download_partial_products(
            products,
            shp,
            out_dir=data_dir,
            aws_key=cred["username"],
            aws_secret=cred["password"],
            pol="full",
            force_overwrite=False,
        )


if __name__ == "__main__":
    main()
