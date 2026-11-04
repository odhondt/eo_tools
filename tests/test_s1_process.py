import pytest
import tempfile
import os
import numpy as np
from numpy.random import randn
import xarray as xr
from eo_tools.S1.process import (
    coherence,
    process_insar,
    h_alpha_dual,
    eigh_2x2,
    polsar_cov_dual,
)
import geopandas as gpd
from shapely.geometry import box
import multiprocessing
import rasterio as rio
import rioxarray
from affine import Affine
from tempfile import NamedTemporaryFile
from eo_tools.S1.process import multilook
from eo_tools.S1.process import goldstein
import tempfile
from unittest.mock import Mock, patch

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


def test_eig_2x2():
    shape = (100, 100)
    k_vv = np.sqrt(0.5) * (np.random.rand(*shape) + 1j * np.random.rand(*shape))
    k_vh = np.sqrt(0.5) * (np.random.rand(*shape) + 1j * np.random.rand(*shape))

    c11 = np.mean(k_vv * k_vv.conj(), axis=(0, 1)).real
    c22 = np.mean(k_vh * k_vh.conj(), axis=(0, 1)).real
    c12 = np.mean(k_vv * k_vh.conj(), axis=(0, 1))

    l1, l2, v11, v12, v21, v22 = eigh_2x2(c11, c22, c12)

    assert np.all((np.isfinite(it) for it in [l1, l2, v11, v12, v21, v22]))


def test_alpha_ent_basic():
    # let's test alpha is well estimated
    # on a simulated single mechanism target

    # used for conditioning
    eps = 1e-10
    # emat = eps * np.diag([1, 1, 1])
    emat = eps * np.diag([1, 1])

    # centered random vector
    D = 2
    N = 1000
    v = np.sqrt(0.5) * (randn(N, N, D) + 1j * randn(N, N, D))

    # polarimetric mechanism
    alpha_sim = np.pi / 5
    # unitary vector
    u = np.array([np.cos(alpha_sim), np.sin(alpha_sim)])
    # u = np.array([np.cos(alpha_sim), np.sin(alpha_sim), 0])
    l1 = 7.0

    # square root of mean matrix
    Sigma = np.sqrt(l1) * u[None, :] * u[:, None].conj()
    # correlate target vector
    k = np.matmul(v, Sigma.T)

    # # covariance estimate
    M = k[:, :, None, :] * k[:, :, :, None].conj()
    C = np.mean(M, axis=(0, 1)) + emat

    c11 = C[..., 0, 0].real
    c22 = C[..., 1, 1].real
    c12 = C[..., 0, 1]
    l1, l2, v11, _, v21, _ = eigh_2x2(c11, c22, c12)

    eps = 1e-30
    span = l1 + l2

    # Pseudo-probabilities
    pp1 = np.clip(l1 / (span + eps), eps, 1)
    pp2 = np.clip(l2 / (span + eps), eps, 1)

    # Entropy
    H = -pp1 * np.log2(pp1 + eps) - pp2 * np.log2(pp2 + eps)

    alpha1 = np.arccos(np.abs(v11))  # * 180 / np.pi
    alpha2 = np.arccos(np.abs(v21))  # * 180 / np.pi

    alpha = pp1 * alpha1 + pp2 * alpha2

    assert np.allclose(l1, 7, atol=0.1)
    assert np.allclose(l2, 0, atol=0.1)
    assert np.allclose(alpha, alpha_sim)
    assert np.allclose(H, 0)


@pytest.fixture
def create_h_alpha_data():
    # siz = 128
    siz = 64
    vv_data = np.random.rand(siz, siz) + 1j * np.random.rand(siz, siz)
    vh_data = np.random.rand(siz, siz) + 1j * np.random.rand(siz, siz)
    vv_ds = xr.DataArray(vv_data.astype("complex64"), dims=("y", "x"))
    vh_ds = xr.DataArray(vh_data.astype("complex64"), dims=("y", "x"))

    with tempfile.TemporaryDirectory() as tmpdirname:
        vv_file = os.path.join(tmpdirname, "vv.tif")
        vh_file = os.path.join(tmpdirname, "vh.tif")
        h_file = os.path.join(tmpdirname, "H.tif")
        alpha_file = os.path.join(tmpdirname, "alpha.tif")
        # span_file = os.path.join(tmpdirname, "span.tif")
        vv_ds.rio.to_raster(vv_file)
        vh_ds.rio.to_raster(vh_file)
        yield vv_file, vh_file, h_file, alpha_file, vv_ds.shape


def test_h_alpha_dual(create_h_alpha_data):
    import rioxarray as riox

    vv_file, vh_file, h_file, alpha_file, shp = create_h_alpha_data
    h_alpha_dual(vv_file=vv_file, vh_file=vh_file, h_file=h_file, alpha_file=alpha_file)
    alpha = riox.open_rasterio(alpha_file)[0]
    h = riox.open_rasterio(h_file)[0]

    assert alpha.shape == shp
    assert h.shape == shp
    assert alpha.dtype == "float32"
    assert h.dtype == "float32"

@pytest.fixture
def create_polsar_data():
    # siz = 128
    siz = 64
    vv_data = np.random.rand(siz, siz) + 1j * np.random.rand(siz, siz)
    vh_data = np.random.rand(siz, siz) + 1j * np.random.rand(siz, siz)
    vv_ds = xr.DataArray(vv_data.astype("complex64"), dims=("y", "x"))
    vh_ds = xr.DataArray(vh_data.astype("complex64"), dims=("y", "x"))

    with tempfile.TemporaryDirectory() as tmpdirname:
        vv_file = os.path.join(tmpdirname, "vv.tif")
        vh_file = os.path.join(tmpdirname, "vh.tif")
        c11_file = os.path.join(tmpdirname, "c11.tif")
        c12_real_file = os.path.join(tmpdirname, "c12_real.tif")
        c12_imag_file = os.path.join(tmpdirname, "c12_imag.tif")
        c22_file = os.path.join(tmpdirname, "c22.tif")
        vv_ds.rio.to_raster(vv_file)
        vh_ds.rio.to_raster(vh_file)
        yield vv_file, vh_file, c11_file, c22_file, c12_real_file, c12_imag_file, vv_ds.shape

def test_polsar_cov_dual(create_polsar_data):
    import rioxarray as riox

    vv_file, vh_file, c11_file, c22_file, c12_real_file, c12_imag_file, shp = create_polsar_data
    polsar_cov_dual(vv_file=vv_file, vh_file=vh_file, c11_file=c11_file, c22_file=c22_file, c12_real_file=c12_real_file,  c12_imag_file=c12_imag_file)
    c11 = riox.open_rasterio(c11_file)[0]
    c22 = riox.open_rasterio(c22_file)[0]
    c12r = riox.open_rasterio(c12_real_file)[0]
    c12i = riox.open_rasterio(c12_imag_file)[0]

    assert c11.shape == shp
    assert c22.shape == shp
    assert c12r.shape == shp
    assert c12i.shape == shp
    assert c11.dtype == "float32"
    assert c22.dtype == "float32"
    assert c12r.dtype == "float32"
    assert c12i.dtype == "float32"


@pytest.mark.parametrize("kind", ["insar", "slc"])
@pytest.mark.parametrize("pol", ["full", ["vv", "vh"], "VV"])
@pytest.mark.parametrize("skip", [False, True])
def test_prepare_outputs(tmp_path, pol: str | list[str], skip: bool, kind: str) -> None:
    """Preparation batches channels per IW and exposes the shared LUT filenames."""
    from pathlib import Path
    from types import SimpleNamespace

    from eo_tools.S1.process import prepare_insar, prepare_slc

    primary_path, secondary_path = tmp_path / "primary", tmp_path / "secondary"
    primary_path.mkdir()
    secondary_path.mkdir()
    geometry = gpd.GeoDataFrame(
        {"burst": [1, 2], "subswath": ["IW1", "IW2"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    primary = SimpleNamespace(
        polarizations=["VV", "VH"],
        scanMetadata=lambda: {"orbitNumber_rel": 42, "start": "20230904T063730"},
    )
    secondary = SimpleNamespace(
        scanMetadata=lambda: {"orbitNumber_rel": 42, "start": "20230916T063730"}
    )

    def preprocess(*args, **kwargs) -> None:
        output_dir = args[-1]
        for p in kwargs["pol"]:
            roles = ["primary", "secondary"] if kind == "insar" else ["slc"]
            for role in roles:
                (Path(output_dir) / f"{role}_{p}.tif").touch()
        (Path(output_dir) / "lut.tif").touch()

    with (
        patch("eo_tools.S1.process.get_burst_geometry", return_value=geometry),
        patch(
            "eo_tools.S1.process.identify_safe_product",
            side_effect=[primary, secondary] if kind == "insar" else [primary],
        ),
        patch(
            f"eo_tools.S1.process.preprocess_{kind}_iw", side_effect=preprocess
        ) as prepare,
    ):
        prepare_func = prepare_insar if kind == "insar" else prepare_slc
        paths = [str(primary_path)]
        if kind == "insar":
            paths.append(str(secondary_path))
        out_dir = Path(
            prepare_func(
                *paths,
                str(tmp_path),
                pol=pol,
                subswaths=["IW1", "IW2"],
                skip_preprocessing=skip,
            )
        )

    assert prepare.call_count == (0 if skip else 2)
    if skip:
        return
    polarizations = ["vv"] if pol == "VV" else ["vv", "vh"]
    assert [call.kwargs["iw"] for call in prepare.call_args_list] == [1, 2]
    for call in prepare.call_args_list:
        assert call.kwargs["pol"] == polarizations
    expected = {f"lut_iw{iw}.tif" for iw in [1, 2]}
    prefixes = ["slc_prm", "slc_sec"] if kind == "insar" else ["slc"]
    expected.update(
        f"{prefix}_{p}_iw{iw}.tif"
        for iw in [1, 2]
        for p in polarizations
        for prefix in prefixes
    )
    assert {path.name for path in out_dir.iterdir()} == expected


@pytest.mark.parametrize("shared", [True, False])
def test_geocode_iw_lut(tmp_path, shared: bool) -> None:
    """Geocoding requires a shared IW LUT and rejects legacy channel LUTs."""
    from eo_tools.S1.process import geocode_and_merge_iw

    sar_dir = tmp_path / "sar"
    sar_dir.mkdir()
    rows, cols = np.indices((4, 4), dtype=float)
    profile = dict(
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="float32",
        transform=Affine.identity(),
        nodata=np.nan,
    )
    for name, value in [("amp_vv", 2), ("amp_vh", 3), ("H", 4)]:
        with rio.open(sar_dir / f"{name}_iw1.tif", "w", **profile) as dst:
            dst.write(np.full((4, 4), value, dtype=np.float32), 1)
    profile.update(
        count=2, dtype="float64", crs="EPSG:4326", transform=Affine(1, 0, 0, 0, -1, 4)
    )
    names = ["lut_iw1"] if shared else ["lut_vv_iw1", "lut_vh_iw1"]
    for name in names:
        with rio.open(sar_dir / f"{name}.tif", "w", **profile) as dst:
            dst.write(np.stack([rows, cols]))

    if not shared:
        with pytest.raises(FileNotFoundError, match="lut_iw1.tif"):
            geocode_and_merge_iw(str(tmp_path), ["amp"], subswaths=["IW1"])
        return

    geocode_and_merge_iw(
        str(tmp_path), ["amp", "H"], subswaths=["IW1"], warp_kernel="nearest"
    )

    for name, value in [("amp_vv", 2), ("amp_vh", 3), ("H", 4)]:
        with rio.open(tmp_path / f"{name}.tif") as result:
            np.testing.assert_array_equal(result.read(1), np.full((4, 4), value))


@pytest.mark.parametrize("cal_type", ["beta", "sigma", "terrain"])
@pytest.mark.parametrize("burst_count", [1, 2])
def test_slc_channels(tmp_path, cal_type: str, burst_count: int) -> None:
    """Reuse burst geometry and terrain factors while retaining channel masks."""
    from types import SimpleNamespace

    from eo_tools.S1.process import preprocess_slc_iw

    rows, cols = np.indices((4, 4), dtype=float)
    gamma = np.full((4, 4), 4.0)
    gamma[0, 0] = np.nan
    image = np.full((4, 4), 2 + 3j, dtype=np.complex64)
    vh_image = image.copy()
    vh_image[2, 2] = np.nan
    geometry = gpd.GeoDataFrame(
        {"burst": range(1, burst_count + 1)},
        geometry=[box(0, 4 * i, 4, 4 * (i + 1)) for i in range(burst_count)],
        crs="EPSG:4326",
    )
    dem = tmp_path / "dem.tif"
    with rio.open(
        dem,
        "w",
        driver="GTiff",
        width=4,
        height=4 * burst_count,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=Affine(1, 0, 0, 0, -1, 4 * burst_count),
    ) as dst:
        dst.write(np.zeros((4 * burst_count, 4), dtype=np.float32), 1)

    def geocode(*args, **kwargs) -> tuple:
        coordinates = (rows.copy(), cols.copy())
        return (
            (*coordinates, rows.copy(), cols.copy(), rows.copy())
            if kwargs.get("return_vectors")
            else coordinates
        )

    primary = SimpleNamespace(
        is_partial=False,
        min_burst=1,
        max_burst=burst_count,
        burst_count=burst_count,
        lines_per_burst=4,
        samples_per_burst=4,
        gdf_burst_geom=geometry,
        fetch_dem=Mock(return_value=str(dem)),
        geocode_burst=Mock(side_effect=geocode),
        compute_burst_overlap=lambda *args, **kwargs: 0,
        compute_burst_offset=lambda burst_idx, min_burst: 4 * (burst_idx - min_burst),
        read_burst=lambda *args: image.copy(),
        calibration_factor=Mock(return_value=2),
    )
    secondary = SimpleNamespace(**vars(primary))
    secondary.geocode_burst = Mock(side_effect=geocode)
    secondary.read_burst = lambda *args: vh_image.copy()
    secondary.calibration_factor = Mock(return_value=4)
    with (
        patch("eo_tools.S1.process.S1IWSwath", side_effect=[primary, secondary]),
        patch(
            "eo_tools.S1.process.load_dem_coords",
            return_value=(rows, cols, gamma, {}, "EPSG:4326"),
        ) as load,
        patch(
            "eo_tools.S1.process.lla_to_ecef", return_value=(rows, cols, gamma)
        ) as ecef,
        patch("eo_tools.S1.process.simulate_terrain", return_value=gamma) as terrain,
    ):
        preprocess_slc_iw(
            "product",
            str(tmp_path),
            pol=["vv", "vh"],
            cal_type=cal_type,
            dem_buffer_arc_sec=0,
        )

    assert load.call_count == burst_count
    assert ecef.call_count == burst_count
    assert terrain.call_count == (burst_count if cal_type == "terrain" else 0)
    assert primary.fetch_dem.call_count == 1
    assert primary.geocode_burst.call_count == burst_count
    secondary.geocode_burst.assert_not_called()
    assert (tmp_path / "lut.tif").is_file()
    for p, source, factor, swath in [
        ("vv", image, 2, primary),
        ("vh", vh_image, 4, secondary),
    ]:
        expected = source / factor
        if cal_type == "terrain":
            expected /= np.sqrt(gamma)
        assert swath.calibration_factor.call_count == burst_count
        assert swath.calibration_factor.call_args.kwargs["cal_type"] == (
            "beta" if cal_type == "terrain" else cal_type
        )
        with rio.open(tmp_path / f"slc_{p}.tif") as result:
            np.testing.assert_allclose(
                result.read(1), np.tile(expected, (burst_count, 1))
            )


@pytest.mark.parametrize("field", [None, "frequency", "sampling", "origin", "samples"])
def test_phase_grid(field: str | None) -> None:
    """Even a one-ULP metadata difference selects the absolute-phase fallback."""
    from copy import deepcopy

    from eo_tools.S1.core import S1IWSwath
    from eo_tools.S1.process import _phase_scale

    primary = object.__new__(S1IWSwath)
    primary.samples_per_burst = 16
    primary.meta = {
        "product": {
            "imageAnnotation": {
                "imageInformation": {"slantRangeTime": "0.005334431164884956"}
            },
            "generalAnnotation": {
                "productInformation": {
                    "radarFrequency": "5405000454.33435",
                    "rangeSamplingRate": "64345238.12571428",
                }
            },
        }
    }
    secondary = deepcopy(primary)
    if field == "samples":
        secondary.samples_per_burst += 1
    elif field is not None:
        product = secondary.meta["product"]
        if field == "origin":
            metadata, key = (
                product["imageAnnotation"]["imageInformation"],
                "slantRangeTime",
            )
        else:
            metadata = product["generalAnnotation"]["productInformation"]
            key = {"frequency": "radarFrequency", "sampling": "rangeSamplingRate"}[
                field
            ]
        metadata[key] = np.nextafter(float(metadata[key]), np.inf)

    scale = _phase_scale(primary, secondary)

    if field is not None:
        assert scale is None
        return
    # The coefficient reproduces the phase slope without the large range origin.
    np.testing.assert_allclose(
        scale,
        primary.phi_topo(np.array([1.0]))[0] - primary.phi_topo(np.array([0.0]))[0],
        rtol=1e-10,
    )
