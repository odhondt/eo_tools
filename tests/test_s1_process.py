import pytest
import tempfile
import os
import numpy as np
import xarray as xr
from eo_tools.S1.process import coherence, process_insar
import geopandas as gpd
import multiprocessing
import rasterio as rio
import rioxarray
from affine import Affine
from tempfile import NamedTemporaryFile
from eo_tools.S1.process import multilook
from eo_tools.S1.process import goldstein
import tempfile
from unittest.mock import patch

multiprocessing.set_start_method("forkserver", force=True)
import warnings
import rasterio

from glob import glob


# TODO create dataArrays instead of datasets
@pytest.fixture
def create_test_data():
    # Create temporary GeoTiff files for testing
    prm_data = np.random.rand(10, 10)
    sec_data = np.random.rand(10, 10)
    prm_ds = xr.DataArray(prm_data, dims=("y", "x")).to_dataset(name="band_data")
    sec_ds = xr.DataArray(sec_data, dims=("y", "x")).to_dataset(name="band_data")

    with tempfile.TemporaryDirectory() as tmpdirname:
        prm_file = os.path.join(tmpdirname, "prm.tif")
        sec_file = os.path.join(tmpdirname, "sec.tif")
        out_file = os.path.join(tmpdirname, "out.tif")
        prm_ds.rio.to_raster(prm_file)
        sec_ds.rio.to_raster(sec_file)
        yield prm_file, sec_file, out_file


def test_coherence(create_test_data):
    prm_file, sec_file, out_file = create_test_data

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=rasterio.errors.NotGeoreferencedWarning
        )
        coherence(prm_file, sec_file, out_file)
    assert os.path.exists(out_file)


# def test_process_insar(tmp_path):

#     data_dir = "./data/S1"
#     aoi_file = "/eo_tools/data/Morocco_tiny.geojson"
#     shp = gpd.read_file(aoi_file).geometry[0]
#     ids = [
#         "S1A_IW_SLC__1SDV_20230904T063730_20230904T063757_050174_0609E3_DAA1",
#         "S1A_IW_SLC__1SDV_20230916T063730_20230916T063757_050349_060FCD_6814",
#     ]
#     primary_path = f"{data_dir}/{ids[0]}.SAFE"
#     secondary_path = f"{data_dir}/{ids[1]}.SAFE"

#     process_args = dict(
#         prm_path=primary_path,
#         sec_path=secondary_path,
#         output_dir=tmp_path,
#         aoi_name=None,
#         shp=shp,
#         pol="vv",
#         subswaths=["IW1", "IW2", "IW3"],
#         write_coherence=True,
#         write_interferogram=True,
#         write_primary_amplitude=True,
#         write_secondary_amplitude=False,
#         apply_fast_esd=True,
#         dem_dir="./data/dem/",
#         dem_upsampling=0.5,
#         dem_force_download=False,
#         dem_buffer_arc_sec=20,
#         boxcar_coherence=[3, 3],
#         filter_ifg=True,
#         multilook=[2, 8],
#         warp_kernel="nearest",
#         clip_to_shape=True,
#     )
#     # ignore zero division inherent to "fake" test data
#     out_dir = process_insar(**process_args)
#     assert out_dir
#     for name in ("coh", "phi", "amp_prm"):
#         assert glob(f"{out_dir}/{name}_vv.tif")
#     for iw in ("iw1", "iw2"):
#         for name in ("slc_prm", "slc_sec", "coh", "amp_prm", "lut"):
#             assert glob(f"{out_dir}/sar/{name}_vv_{iw}.tif")


# Helper function to create a dummy GeoTIFF file
def create_dummy_geotiff(width, height, count=1, dtype="uint16", transform=None):
    """Creates a dummy GeoTIFF file with specified dimensions and transform."""
    if transform is None:
        transform = Affine.translation(0, 0) * Affine.scale(
            1, 1
        )  # Identity transform (rectilinear)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": dtype,
        "transform": transform,
        "crs": None,  # Ensure no CRS is used
    }
    data = np.ones((count, height, width), dtype=dtype)  # Dummy data (all ones)

    with NamedTemporaryFile(suffix=".tif", delete=False) as temp_file:
        with rio.open(temp_file.name, "w", **profile) as dst:
            dst.write(data)

        return temp_file.name


def test_multilook_transform_and_dimensions():
    width, height = 100, 100
    input_file = create_dummy_geotiff(width, height)

    output_file = NamedTemporaryFile(suffix=".tif", delete=False).name

    multilook_factors = [2, 2]

    multilook(input_file, output_file, mlt=multilook_factors)

    with rio.open(input_file) as src:
        assert src.transform.is_rectilinear  # Check that the transform is rectilinear
        assert src.crs is None  # Ensure there's no CRS

    with rio.open(output_file) as src:
        assert src.width == width // multilook_factors[1]
        assert src.height == height // multilook_factors[0]

        assert src.transform.is_rectilinear

        expected_transform = Affine.scale(multilook_factors[1], multilook_factors[0])
        assert src.transform.a == expected_transform.a  # x scale
        assert src.transform.e == expected_transform.e  # y scale

        assert src.crs is None


@pytest.fixture
def create_dummy_ifg():
    """
    Create a temporary complex-valued dummy interferogram for testing.

    Yields:
        str: Path to the temporary input interferogram file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a random complex-valued array to simulate the interferogram
        ifg_data = np.random.rand(2048, 2048) + 1j * np.random.rand(2048, 2048)
        da_ifg = xr.DataArray(ifg_data[None], dims=("band", "y", "x"))
        da_ifg.rio.write_crs("EPSG:4326", inplace=True)

        # Save to a temporary file
        input_file = os.path.join(tmpdir, "dummy_ifg.tif")
        da_ifg.rio.to_raster(input_file)

        yield input_file


@pytest.fixture
def create_dummy_output():
    """
    Create a temporary output file path.

    Yields:
        str: Path to the temporary output file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "output_ifg.tif")
        yield output_file


# Dummy function for process_overlapping_windows to be mocked in the test
def dummy_block_process(chunk, window_size, overlap, func, alpha):
    return chunk  # returns the chunk as-is for testing purposes


def test_goldstein(create_dummy_ifg, create_dummy_output):
    """
    Test the goldstein filter function using a random dummy interferogram.
    """
    input_file = create_dummy_ifg
    output_file = create_dummy_output

    # Mock the process_overlapping_windows to avoid actual processing during the test
    with patch("eo_tools.auxils.block_process", side_effect=dummy_block_process):
        goldstein(input_file, output_file, alpha=0.5, overlap=14)

        # Check if the output file was created
        assert os.path.exists(output_file), "Output file was not created."

        # Check if the output is a valid raster
        da_out = rioxarray.open_rasterio(output_file)
        assert da_out.shape == (1, 2048, 2048), "Output shape is incorrect."
