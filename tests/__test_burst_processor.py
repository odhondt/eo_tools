import pytest
import numpy as np
import tempfile
import os
from unittest.mock import MagicMock, patch
import rasterio as rio
from rasterio.windows import Window
from pathlib import Path

from eo_tools.S1.process import _process_bursts
from eo_tools.S1.core import S1IWSwath

@pytest.fixture
def setup_temp_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directories for measurements and annotations
        dir_tiff = Path(tmpdir) / "measurement"
        dir_tiff.mkdir(parents=True)
        dir_xml = Path(tmpdir) / "annotation"
        dir_xml.mkdir(parents=True)
        dir_cal = dir_xml / "calibration"
        dir_cal.mkdir(parents=True)

        # Create dummy XML and TIFF files
        iw = 1
        pol = "vv"
        dummy_xml = dir_xml / f"dummy_iw{iw}_{pol}.xml"
        dummy_tiff = dir_tiff / f"dummy_iw{iw}_{pol}.tiff"
        dummy_cal_xml = dir_cal / f"calibration_iw{iw}_{pol}.xml"
        
        # Create the dummy files
        dummy_xml.touch()
        dummy_tiff.touch()
        dummy_cal_xml.touch()

        yield tmpdir

@patch('eo_tools.S1.core.S1IWSwath.fetch_dem_burst')
@patch('eo_tools.S1.core.S1IWSwath.geocode_burst')
@patch('eo_tools.S1.core.S1IWSwath.read_burst')
@patch('eo_tools.S1.core.S1IWSwath.deramp_burst')
@patch('rasterio.open')
@patch('eo_tools.S1.process._make_da_from_dem')
def test_process_bursts(
        mock_make_da_from_dem,
        mock_rio_open,
        mock_deramp_burst,
        mock_read_burst,
        mock_geocode_burst,
        mock_fetch_dem_burst,
        setup_temp_files
    ):
    tmpdir = setup_temp_files

    # Mock parameter values
    prm = S1IWSwath(tmpdir)
    sec = MagicMock()
    naz, nrg = 100, 100
    min_burst, max_burst = 1, 1
    dem_upsampling = 1
    dem_buffer_arc_sec = 1
    dem_force_download = False
    order = 1

    # Mock return values
    dem_file = 'dem_file.tif'
    mock_fetch_dem_burst.return_value = dem_file
    az_p2g = np.zeros((100, 100))
    rg_p2g = np.zeros((100, 100))
    dem_profile = {}
    mock_geocode_burst.return_value = (az_p2g, rg_p2g, dem_profile)
    arr_p = np.zeros((100, 100), dtype=np.complex64)
    arr_s = np.zeros((100, 100), dtype=np.complex64)
    mock_read_burst.return_value = arr_p
    pdb_s = np.zeros((100, 100), dtype=np.complex64)
    mock_deramp_burst.return_value = pdb_s
    aligned_arr = np.zeros((100, 100), dtype=np.complex64)
    mock_make_da_from_dem.return_value = MagicMock()

    # Define temporary file paths
    tmp_prm = os.path.join(tmpdir, 'tmp_prm.tif')
    tmp_sec = os.path.join(tmpdir, 'tmp_sec.tif')
    dir_out = tmpdir
    dir_dem = tmpdir

    with patch('rasterio.open', mock_rio_open):
        mock_dataset = MagicMock()
        mock_rio_open.return_value.__enter__.return_value = mock_dataset

        luts = _process_bursts(
            prm,
            sec,
            tmp_prm,
            tmp_sec,
            dir_out,
            dir_dem,
            naz,
            nrg,
            min_burst,
            max_burst,
            dem_upsampling,
            dem_buffer_arc_sec,
            dem_force_download,
            order
        )

    # Assertions
    assert len(luts) == 1
    assert luts[0] == f"{dir_out}/lut_1.tif"

    mock_fetch_dem_burst.assert_called_once_with(prm, 1, dir_dem, buffer_arc_sec=dem_buffer_arc_sec, force_download=dem_force_download)
    mock_geocode_burst.assert_called_once_with(prm, dem_file, 1, dem_upsampling=dem_upsampling)
    mock_read_burst.assert_called_once_with(prm, 1, True)
    mock_deramp_burst.assert_called_once_with(sec, 1)
    mock_make_da_from_dem.assert_called_once_with(np.stack((az_p2g, rg_p2g)), dem_profile)
    mock_dataset.write.assert_called()

    # Additional assertions for inputs and outputs
    align_call_args = mock_make_da_from_dem.call_args_list
    assert np.array_equal(align_call_args[0][0][0], np.stack((az_p2g, rg_p2g)))
    assert align_call_args[0][1] == dem_profile
    mock_rio_open.assert_called_once()
    assert np.array_equal(mock_dataset.write.call_args[0][0], arr_p)
    assert mock_dataset.write.call_args[1]['window'] == Window(0, 0, nrg, prm.lines_per_burst)
