import pytest
import tempfile
import os
import numpy as np
import xarray as xr
from eo_tools.S1.process import coherence, process_insar
import geopandas as gpd
import multiprocessing
import rasterio as rio
from affine import Affine
from tempfile import NamedTemporaryFile
from eo_tools.S1.process import multilook  # Assume this is th

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
        prm_ds.to_netcdf(prm_file)
        sec_ds.to_netcdf(sec_file)
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
#     file_aoi = "/eo_tools/data/Morocco_tiny.geojson"
#     shp = gpd.read_file(file_aoi).geometry[0]
#     ids = [
#         "S1A_IW_SLC__1SDV_20230904T063730_20230904T063757_050174_0609E3_DAA1",
#         "S1A_IW_SLC__1SDV_20230916T063730_20230916T063757_050349_060FCD_6814",
#     ]
#     primary_dir = f"{data_dir}/{ids[0]}.SAFE"
#     secondary_dir = f"{data_dir}/{ids[1]}.SAFE"

#     process_args = dict(
#         dir_prm=primary_dir,
#         dir_sec=secondary_dir,
#         outputs_prefix=tmp_path,
#         aoi_name=None,
#         shp=shp,
#         pol="vv",
#         subswaths=["IW1", "IW2", "IW3"],
#         write_coherence=True,
#         write_interferogram=True,
#         write_primary_amplitude=True,
#         write_secondary_amplitude=False,
#         apply_fast_esd=True,
#         dir_dem="./data/dem/",
#         dem_upsampling=0.5,
#         dem_force_download=False,
#         dem_buffer_arc_sec=20,
#         boxcar_coherence=[3, 10],
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

    multilook(input_file, output_file, multilook=multilook_factors)

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
