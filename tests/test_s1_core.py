import pytest
import os
from eo_tools.S1.core import S1IWSwath
from eo_tools.S1.core import range_doppler
from eo_tools.S1.core import simulate_terrain_backscatter, detect_active_shadow
import hashlib
from shapely.geometry import box
from unittest.mock import patch
import numpy as np
import rioxarray as riox
import rasterio
from rasterio import MemoryFile


@pytest.fixture(scope="module")
def create_swath():
    safe_dir = "./data/S1/S1A_IW_SLC__1SDV_20230904T063730_20230904T063757_050174_0609E3_DAA1.SAFE"
    iw = 1
    pol = "vh"
    return S1IWSwath(safe_dir, iw, pol)


def test_s1iwswath_init(create_swath):
    import os

    swath = create_swath

    assert os.path.isfile(swath.pth_tiff)
    assert swath.start_time == "2023-09-04T06:37:31.072288"
    assert swath.lines_per_burst == 1507
    assert swath.samples_per_burst == 23055
    assert swath.burst_count == 9
    assert swath.beta_nought == 2.370000e02


def test_read_burst_valid_burst(create_swath):
    swath = create_swath

    # Mock the attributes that would be set by real product metadata
    with patch.object(swath, "burst_count", 3), patch.object(
        swath, "lines_per_burst", 1500
    ), patch.object(swath, "pth_tiff", "mocked_path.tiff"), patch(
        "eo_tools.S1.core.read_chunk"
    ) as mock_read_chunk:

        # Arrange: Set up a fake array to return when read_chunk is called
        fake_array = np.ones((1500, 2000), dtype=np.complex64)
        mock_read_chunk.return_value = fake_array

        # Act: Call the `read_burst` method with a valid burst index
        result = swath.read_burst(burst_idx=1, remove_invalid=False)

        # Assert: Check if read_chunk was called correctly and the result matches
        mock_read_chunk.assert_called_once_with("mocked_path.tiff", 0, 1500)
        assert np.array_equal(
            result, fake_array
        ), "The returned burst data does not match the expected output"


def test_read_burst_remove_invalid(create_swath):
    swath = create_swath

    lines_per_burst = 1500

    # Generate firstValidSample and lastValidSample arrays
    first_valid_sample_array = (
        "1 " * lines_per_burst
    )  # First column valid for all lines
    last_valid_sample_array = (
        "1999 " * lines_per_burst
    )  # Last valid sample at column 1999 for all lines

    # Break down the metadata dictionary into manageable parts
    burst_metadata = [
        {
            "firstValidSample": {"#text": first_valid_sample_array.strip()},
            "lastValidSample": {"#text": last_valid_sample_array.strip()},
        },
        {  # Mock burst metadata for burst_idx=2
            "firstValidSample": {"#text": "2 " * lines_per_burst},
            "lastValidSample": {"#text": "1998 " * lines_per_burst},
        },
    ]

    burst_list = {"burst": burst_metadata}

    swath_timing = {"burstList": burst_list}

    product_metadata = {"swathTiming": swath_timing}

    mock_meta = {"product": product_metadata}

    # Mock the attributes and methods that would normally be populated by metadata
    with patch.object(swath, "burst_count", 3), patch.object(
        swath, "lines_per_burst", lines_per_burst
    ), patch.object(swath, "pth_tiff", "mocked_path.tiff"), patch.object(
        swath, "meta", mock_meta
    ), patch(
        "eo_tools.S1.core.read_chunk"
    ) as mock_read_chunk:

        # Create a fake array where the first and last columns are invalid
        fake_array = np.ones((lines_per_burst, 2000), dtype=np.complex64)
        fake_array[:, 0] = np.nan + 1j * np.nan  # Invalid first column
        fake_array[:, -1] = np.nan + 1j * np.nan  # Invalid last column

        mock_read_chunk.return_value = fake_array

        # Act: Call the `read_burst` method with `remove_invalid=True`
        result = swath.read_burst(burst_idx=1, remove_invalid=True)

        # Assert: Check if the invalid pixels in first and last columns are NaN
        assert np.isnan(
            result[:, 0]
        ).all(), "Invalid first column pixels were not set to NaN"
        assert np.isnan(
            result[:, -1]
        ).all(), "Invalid last column pixels were not set to NaN"
        assert not np.isnan(result[:, 1:1999]).any(), "Valid pixels should not be NaN"


# Test invalid burst index
def test_read_burst_invalid_burst(create_swath):
    swath = create_swath

    # Mock the burst count to 3 for testing
    with patch.object(swath, "burst_count", 3):

        # Act & Assert: Test burst_idx out of range (e.g., 0 or larger than burst_count)
        with pytest.raises(ValueError, match=r"Invalid burst index.*"):
            swath.read_burst(burst_idx=0)  # Invalid index

        with pytest.raises(ValueError, match=r"Invalid burst index.*"):
            swath.read_burst(burst_idx=4)  # Invalid index (larger than burst_count)


# Test case for beta calibration using create_swath fixture
def test_calibration_factor_beta(create_swath):
    swath = create_swath

    # Mock the necessary attributes
    with patch.object(swath, "beta_nought", 1.5):
        # Act: Call the calibration_factor method with cal_type "beta"
        result = swath.calibration_factor(cal_type="beta")

        # Assert: The result should match the beta_nought constant
        assert result == 1.5, "Beta calibration factor should be a constant"


# Test case for sigma calibration with interpolation using create_swath fixture
def test_calibration_factor_sigma(create_swath):
    swath = create_swath

    # Mock the necessary attributes
    with patch.object(swath, "lines_per_burst", 2), patch.object(
        swath, "samples_per_burst", 3
    ), patch.object(
        swath,
        "calvec",
        [
            {
                "line": "0",
                "pixel": {"#text": "0 1 2"},
                "sigmaNought": {"#text": "1.0 2.0 3.0"},
            },
            {
                "line": "1",
                "pixel": {"#text": "0 1 2"},
                "sigmaNought": {"#text": "4.0 5.0 6.0"},
            },
        ],
    ):
        # Act: Call the calibration_factor method with cal_type "sigma"
        result = swath.calibration_factor(cal_type="sigma")

        # Assert: Compare the result to the expected interpolation result
        expected_result = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert np.allclose(
            result, expected_result
        ), "Sigma nought calibration factor interpolation failed"


def test_range_doppler():
    xx = np.array([0.0, 5.0])
    yy = np.array([0.0, 0.0])
    zz = np.array([0.0, 5.0])

    positions = np.vstack((np.linspace(-10, 10, 10), np.full(10, 0), np.full(10, 10))).T

    velocities = np.vstack((np.ones(10), np.zeros(10), np.zeros(10))).T

    expected_i_zd = np.array([4.5, 6.75])
    expected_r_zd = np.array([10, 5])

    i_zd, r_zd, _, _, _ = range_doppler(xx, yy, zz, positions, velocities)

    np.testing.assert_allclose(i_zd, expected_i_zd, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(r_zd, expected_r_zd, rtol=1e-5, atol=1e-8)


def test_fetch_dem_filename_uniqueness(create_swath):
    swath = create_swath

    with patch("os.path.exists") as mock_exists, patch(
        "eo_tools.S1.core.retrieve_dem"
    ) as mock_retrieve_dem:

        mock_exists.return_value = False  # Simulate that the DEM doesn't exist

        # Mock burst count
        swath.burst_count = 3

        # Create bounding box from GeoDataFrame geometries
        geom_all = swath.gdf_burst_geom
        geom_sub_nobuf = geom_all[
            (geom_all["burst"] >= 1) & (geom_all["burst"] <= 2)
        ].union_all()

        dir_dem = "/tmp/test-fetch-dem"

        if not os.path.isdir(dir_dem):
            os.mkdir(dir_dem)

        def expected_filename(
            buffer_arc_sec, upscale_factor, dem_name
        ):
            # Apply the buffer in degrees
            geom_sub = geom_sub_nobuf.buffer(buffer_arc_sec / 3600)
            shp = box(*geom_sub.bounds)
            hash_input = (
                f"{shp.wkt}_{upscale_factor}_{dem_name}".encode(
                    "utf-8"
                )
            )
            hash_str = hashlib.md5(hash_input).hexdigest()
            # expected file name
            file_dem_expected = f"{dir_dem}/dem-{hash_str}.tif"

            # generate dummy file that will be opened in the function to test
            with rasterio.open(
                file_dem_expected,
                "w",
                driver="GTiff",
                height=10,
                width=10,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=rasterio.transform.from_origin(0, 0, 1, 1),
            ) as dataset:
                dataset.write(np.zeros((10, 10), dtype="float32"), 1)

            return file_dem_expected

        # generate expected files and their names
        file_dem_1_e = expected_filename(40, 1.0, "nasadem")
        file_dem_2_e = expected_filename(40, 2.0, "nasadem")
        file_dem_3_e = expected_filename(50, 1.0, "nasadem")
        file_dem_4_e = expected_filename(40, 1.0, "alos-dem")

        # Generate DEM with different parameters and capture filenames
        file_dem_1 = swath.fetch_dem(
            min_burst=1,
            max_burst=2,
            dir_dem=dir_dem,
            upscale_factor=1.0,
            buffer_arc_sec=40,
        )
        file_dem_2 = swath.fetch_dem(
            min_burst=1,
            max_burst=2,
            dir_dem=dir_dem,
            upscale_factor=2.0,
            buffer_arc_sec=40,
        )

        # Now change buffer_arc_sec to affect geometry and the resulting filename
        file_dem_3 = swath.fetch_dem(
            min_burst=1,
            max_burst=2,
            dir_dem=dir_dem,
            upscale_factor=1.0,
            buffer_arc_sec=50,
        )

        file_dem_4 = swath.fetch_dem(
            min_burst=1,
            max_burst=2,
            dir_dem=dir_dem,
            upscale_factor=1.0,
            buffer_arc_sec=40,
            dem_name="alos-dem",
        )

        # Ensure unique filenames are generated based on the parameters
        assert (
            file_dem_1 != file_dem_2
        ), "DEM filenames should differ for different upscale factors"
        assert (
            file_dem_1 != file_dem_3
        ), "DEM filenames should differ for different buffer_arc_sec values"
        assert (
            file_dem_2 != file_dem_3
        ), "DEM filenames should differ for different parameters"
        assert (
            file_dem_1 != file_dem_3
        ), "DEM filenames should differ for different dem_name"

        # Verify that the filenames are as expected
        assert (
            file_dem_1 == file_dem_1_e
        ), "Generated DEM filename for upscale_factor=1.0, buffer=40 is incorrect"
        assert (
            file_dem_2 == file_dem_2_e
        ), "Generated DEM filename for upscale_factor=2.0 is incorrect"
        assert (
            file_dem_3 == file_dem_3_e
        ), "Generated DEM filename for buffer_arc_sec=50 is incorrect"
        assert (
            file_dem_4 == file_dem_4_e
        ), "Generated DEM filename for dem_name='alos-dem' is incorrect"


def test_burst_geocoding(create_swath):
    swath = create_swath
    file_dem = swath.fetch_dem_burst(burst_idx=3, force_download=True)
    arr = riox.open_rasterio(file_dem)
    dem_shape = arr[0].shape
    raster_shape = (swath.lines_per_burst, swath.samples_per_burst)
    dem_upsampling = 0.1
    az, rg, gamma_t = swath.geocode_burst(
        file_dem, burst_idx=3, dem_upsampling=dem_upsampling, simulate_terrain=True
    )

    out_shape = (
        int(dem_shape[0] * dem_upsampling),
        int(dem_shape[1] * dem_upsampling),
    )

    assert (
        np.isfinite(az).any() and np.isfinite(rg).any() and np.isfinite(gamma_t).any()
    )
    assert (az.shape == out_shape) and (rg.shape == out_shape)
    assert gamma_t.shape == raster_shape


def test_burst_deramping(create_swath):
    swath = create_swath
    arr = swath.deramp_burst(burst_idx=3)
    assert isinstance(arr, np.ndarray)
    assert np.isfinite(arr).any()
    assert arr.ndim == 2


def test_phi_topo(create_swath):
    swath = create_swath
    rg = np.full((20, 10), fill_value=10000)
    arr = swath.phi_topo(rg)
    assert isinstance(arr, np.ndarray)
    assert np.isfinite(arr).any()
    assert arr.shape == rg.shape


def test_burst_overlap(create_swath):
    swath = create_swath
    with pytest.raises(ValueError, match=r"Invalid burst index.*"):
        _ = swath.compute_burst_overlap(burst_idx=1)
    olap = swath.compute_burst_overlap(burst_idx=3)
    assert isinstance(olap, float)
    assert olap > 0


if __name__ == "__main__":
    pytest.main()
