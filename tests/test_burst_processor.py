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

    mock_rio_open.return_value.__enter__.return_value = MagicMock()

    mock_prm.fetch_dem_burst.return_value = "mock_dem_file"
    mock_prm.geocode_burst.return_value = (
        np.ones((100, 200), dtype=np.float32),
        np.ones((100, 200), dtype=np.float32),
        {"some": "profile"},
    )
    mock_sec.geocode_burst.return_value = (
        np.ones((100, 200), dtype=np.float32),
        np.ones((100, 200), dtype=np.float32),
        {"some": "profile"},
    )
    mock_prm.read_burst.return_value = np.ones((100, 200), dtype=np.complex64)
    mock_sec.read_burst.return_value = np.ones((100, 200), dtype=np.complex64)
    mock_sec.deramp_burst.return_value = np.ones((100, 200), dtype=np.float32)
    mock_prm.phi_topo.return_value = np.ones((100, 200), dtype=np.float32)
    mock_sec.phi_topo.return_value = np.ones((100, 200), dtype=np.float32)

    mock_coregister.return_value = (
        np.ones((100, 200), dtype=np.float32),
        np.ones((100, 200), dtype=np.float32),
    )
    mock_align.return_value = np.ones((100, 200), dtype=np.complex64)
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
    order = 3
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
        order,
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
 
    # Verify the flow of data between functions
    # Checking the arguments for geocode_burst calls
    assert mock_prm.geocode_burst.call_count == 2
    assert mock_sec.geocode_burst.call_count == 2
    assert mock_prm.geocode_burst.call_args_list[0] == call('mock_dem_file', burst_idx=5, dem_upsampling=1.8)
    assert mock_sec.geocode_burst.call_args_list[0] == call('mock_dem_file', burst_idx=5, dem_upsampling=1.8)

    # Check the coregister call
    az_p2g, rg_p2g, _ = mock_prm.geocode_burst.return_value
    az_s2g, rg_s2g, _ = mock_sec.geocode_burst.return_value
    arr_p = mock_prm.read_burst.return_value
    coregister_call_args = mock_coregister.call_args_list[0]
    assert np.array_equal(coregister_call_args[0][0], arr_p)
    assert np.array_equal(coregister_call_args[0][1], az_p2g)
    assert np.array_equal(coregister_call_args[0][2], rg_p2g)
    assert np.array_equal(coregister_call_args[0][3], az_s2g)
    assert np.array_equal(coregister_call_args[0][4], rg_s2g)

    # Check the align call for arr_s2p
    arr_s_de = mock_sec.read_burst.return_value * np.exp(1j * mock_sec.deramp_burst.return_value)
    az_s2p, rg_s2p = mock_coregister.return_value
    align_call_args_0 = mock_align.call_args_list[0]
    assert np.array_equal(align_call_args_0[0][0], arr_p)
    assert np.array_equal(align_call_args_0[0][1], arr_s_de)
    assert np.array_equal(align_call_args_0[0][2], az_s2p)
    assert np.array_equal(align_call_args_0[0][3], rg_s2p)
    assert align_call_args_0[1]['order'] == 3

    # Check the align call for pdb_s2p
    pdb_s = mock_sec.deramp_burst.return_value
    align_call_args_1 = mock_align.call_args_list[1]
    assert np.array_equal(align_call_args_1[0][0], arr_p)
    assert np.array_equal(align_call_args_1[0][1], pdb_s)
    assert np.array_equal(align_call_args_1[0][2], az_s2p)
    assert np.array_equal(align_call_args_1[0][3], rg_s2p)
    assert align_call_args_1[1]['order'] == 3

    # Check the _make_da_from_dem call
    make_da_from_dem_call_args = mock_make_da_from_dem.call_args_list[0]
    assert np.array_equal(make_da_from_dem_call_args[0][0], np.stack((az_p2g, rg_p2g)))
    assert make_da_from_dem_call_args[0][1] == {'some': 'profile'}
