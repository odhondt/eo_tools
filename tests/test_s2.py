import pytest
import tempfile
import shutil
from eo_tools.S2 import process_s2_tiles
from glob import glob
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling


@pytest.fixture
def temp_output_dir():
    """Fixture to create a temporary directory and clean it up after the test."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_process_s2_tiles(temp_output_dir):
    # Prepare input data
    shp = gpd.read_file("./data/Bretagne_AOI.geojson").geometry[0]
    s2_tiles = glob("./data/S2/*.zip")

    # Process the S2 tiles
    out_dirs = process_s2_tiles(
        s2_tiles, bands=["B1"], shp=shp, output_dir=temp_output_dir, force_create=True
    )

    # Check output files and their profiles
    assert len(out_dirs) > 0, "No output directories were generated."

    for out_dir in out_dirs:
        with rasterio.open(glob(f"{out_dir}/*.tif")[0]) as src:
            profile = src.profile

            expected_profile = {
                "driver": "GTiff",
                "dtype": "float32",
                "nodata": 0.0,
                "width": 6475,
                "height": 3513,
                "count": 1,
                "crs": rasterio.crs.CRS.from_epsg(4326),
                "blockxsize": 512,
                "blockysize": 512,
                "tiled": True,
                "compress": "deflate",
                "interleave": "band",
            }

            # Check main profile properties
            for key, value in expected_profile.items():
                assert (
                    profile[key] == value
                ), f"Mismatch in {key}: expected {value}, got {profile[key]}"

            # Check the transform (this may need more tolerance due to floating point precision)
            expected_transform = rasterio.Affine(
                0.00011490345024196447,
                0.0,
                -3.3854078086412236,
                0.0,
                -0.00011490345024196447,
                48.326506645022775,
            )
            assert src.transform.almost_equals(
                expected_transform, precision=1e-8
            ), f"Mismatch in transform: expected {expected_transform}, got {src.transform}"
