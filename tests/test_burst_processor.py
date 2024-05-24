import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np
from eo_tools.S1.process import _process_bursts


@pytest.fixture
def mock_S1IWSwath():
    with patch("eo_tools.S1.core.S1IWSwath") as mock:
        yield mock


@pytest.fixture
def mock_rio_open():
    with patch("rasterio.open") as mock:
        yield mock


@pytest.fixture
def mock_coregister():
    with patch("eo_tools.S1.process.coregister") as mock:
        yield mock


@pytest.fixture
def mock_align():
    with patch("eo_tools.S1.process.align") as mock:
        yield mock


@pytest.fixture
def mock_make_da_from_dem():
    with patch("eo_tools.S1.process._make_da_from_dem") as mock:
        yield mock


def test_process_bursts(
    mock_S1IWSwath, mock_rio_open, mock_coregister, mock_align, mock_make_da_from_dem
):
    # Setup mock objects and return values
    mock_prm = MagicMock()
    mock_sec = MagicMock()
    mock_S1IWSwath.side_effect = [mock_prm, mock_sec]

    mock_prm.lines_per_burst = 100
    mock_prm.samples_per_burst = 200

    shape_dem = (300, 50)
    shape_burst = (100, 200) 

    mock_rio_open.return_value.__enter__.return_value = MagicMock()

    mock_prm.fetch_dem_burst.return_value = "mock_dem_file"
    mock_prm.geocode_burst.return_value = (
        np.ones(shape_dem, dtype=np.float32),
        np.ones(shape_dem, dtype=np.float32),
        {"some": "profile"},
    )
    mock_sec.geocode_burst.return_value = (
        np.ones(shape_dem, dtype=np.float32),
        np.ones(shape_dem, dtype=np.float32),
        {"some": "profile"},
    )
    mock_prm.read_burst.return_value = np.ones(shape_burst, dtype=np.complex64)
    mock_sec.read_burst.return_value = np.ones(shape_burst, dtype=np.complex64)
    mock_sec.deramp_burst.return_value = np.ones(shape_burst, dtype=np.float32)
    mock_prm.phi_topo.return_value = np.ones(shape_burst, dtype=np.float32)
    mock_sec.phi_topo.return_value = np.ones(shape_burst, dtype=np.float32)

    mock_coregister.return_value = (
        np.ones(shape_burst, dtype=np.float32),
        np.ones(shape_burst, dtype=np.float32),
    )
    mock_align.return_value = np.ones(shape_burst, dtype=np.complex64)
    mock_make_da_from_dem.return_value = MagicMock()

    # Parameters
    iw = 1
    pol = "vh"
    dir_primary = "./data/S1A_IW_SLC__1SDV_20230904T063730_20230904T063757_050174_0609E3_DAA1.SAFE"
    dir_secondary = "./data/S1A_IW_SLC__1SDV_20230916T063730_20230916T063757_050349_060FCD_6814.SAFE"
    dir_out = "/tmp"
    dir_dem = "/tmp"
    tmp_prm = f"{dir_out}/tmp_primary.tif"
    tmp_sec = f"{dir_out}/tmp_secondary.tif"
    prm = mock_S1IWSwath(dir_primary, iw=iw, pol=pol)
    sec = mock_S1IWSwath(dir_secondary, iw=iw, pol=pol)
    min_burst = 5
    max_burst = 6
    naz = prm.lines_per_burst * (max_burst - min_burst + 1)
    nrg = prm.samples_per_burst
    dem_upsampling = 1.8
    dem_buffer_arc_sec = 40
    warp_kernel = "bicubic"
    dem_force_download = True

    # Call the function
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
        warp_kernel,
    )
    # Assertions
    assert len(luts) == max_burst - min_burst + 1
    mock_prm.fetch_dem_burst.assert_called()
    mock_prm.geocode_burst.assert_called()
    mock_sec.geocode_burst.assert_called()
    mock_prm.read_burst.assert_called()
    mock_sec.read_burst.assert_called()
    mock_sec.deramp_burst.assert_called()
    mock_coregister.assert_called()
    mock_align.assert_called()
    mock_make_da_from_dem.assert_called()
    mock_rio_open.assert_called()
