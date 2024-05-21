from unittest.mock import patch, MagicMock
import pytest
import tempfile
import os
import numpy as np
import xarray as xr
from eo_tools.S1.process import coherence, _merge_luts


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
    coherence(prm_file, sec_file, out_file)
    assert os.path.exists(out_file)


def test_merge_luts():
    lines = 1507
    overlap = 165
    luts = ["./data/lut_5.tif", "./data/lut_6.tif"]
    fout = "/tmp/merged.tif"
    _merge_luts(luts, fout, lines, overlap, 4)
    da = xr.open_dataset(fout)["band_data"]
    assert da.dims == ("band", "y", "x")
    assert da.shape == (2, 3328, 6773)
