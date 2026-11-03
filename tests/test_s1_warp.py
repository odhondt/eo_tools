from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import from_origin
from shapely.geometry import box

from eo_tools.S1 import process
from eo_tools.S1.util import remap


@pytest.fixture
def burst_geometry() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"burst": [1], "subswath": ["IW1"]},
        geometry=[box(0, 0, 24, 24)],
        crs="EPSG:4326",
    )


@pytest.mark.parametrize(
    "settings, slc_kernel, geo_kernel",
    [
        ({}, "lanczos", "bicubic"),
        ({"warp_kernel": "nearest"}, "lanczos", "nearest"),
        ({"warp_kernel_slc": "sinc", "warp_kernel": "bilinear"}, "sinc", "bilinear"),
    ],
)
def test_insar_kernels(
    tmp_path: Path, settings: dict[str, str], slc_kernel: str, geo_kernel: str
) -> None:
    """SLC and geocoding options are independent at the processor entry point."""
    with (
        patch.object(
            process, "prepare_insar", return_value=str(tmp_path / "sar")
        ) as prepare,
        patch.object(process.os.path, "isfile", return_value=False),
        patch.object(process, "_child_process") as child,
    ):
        process.process_insar(
            "primary",
            "secondary",
            str(tmp_path),
            pol="vv",
            subswaths=["IW1"],
            **settings,
        )
    assert prepare.call_args.kwargs["warp_kernel"] == slc_kernel
    assert "warp_kernel_slc" not in prepare.call_args.kwargs
    geocoding = child.call_args.args[1]
    assert geocoding["warp_kernel"] == geo_kernel
    assert "warp_kernel_slc" not in geocoding


@pytest.mark.parametrize("kernel", [None, "bilinear"])
def test_prepare_kernel(
    tmp_path: Path, burst_geometry: gpd.GeoDataFrame, kernel: str | None
) -> None:
    """Preparation forwards the default or explicitly selected SLC kernel."""
    primary = SimpleNamespace(
        polarizations=["VV"],
        scanMetadata=lambda: {"orbitNumber_rel": 42, "start": "20230904T063730"},
    )
    secondary = SimpleNamespace(
        scanMetadata=lambda: {"orbitNumber_rel": 42, "start": "20230916T063730"}
    )
    primary_path, secondary_path = tmp_path / "primary", tmp_path / "secondary"
    primary_path.touch()
    secondary_path.touch()
    settings = {} if kernel is None else {"warp_kernel": kernel}
    with (
        patch.object(process, "read_partial_download_info", return_value={}),
        patch.object(process, "get_burst_geometry", return_value=burst_geometry),
        patch.object(
            process, "identify_safe_product", side_effect=[primary, secondary]
        ),
        patch.object(process, "preprocess_insar_iw") as preprocess,
        patch.object(process.os, "rename"),
    ):
        process.prepare_insar(
            str(primary_path),
            str(secondary_path),
            str(tmp_path),
            pol="vv",
            subswaths=["IW1"],
            **settings,
        )
    assert preprocess.call_args.kwargs["warp_kernel"] == (kernel or "lanczos")
    assert "warp_kernel_slc" not in preprocess.call_args.kwargs


@pytest.mark.parametrize("kernel", [None, "sinc", "bilinear"])
def test_slc_phase_kernel(
    tmp_path: Path, burst_geometry: gpd.GeoDataFrame, kernel: str | None
) -> None:
    """Written SLCs use the selected alignment and fixed bicubic phase warping."""
    rows, cols = np.indices((24, 24), dtype=float)
    phase = 0.03 * rows**2 + 0.02 * cols**2
    image = np.exp(0.2j * (rows + cols)).astype(np.complex64)
    dem = tmp_path / "dem.tif"
    with rio.open(
        dem,
        "w",
        driver="GTiff",
        width=24,
        height=24,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 24, 1, 1),
    ) as ds:
        ds.write(np.zeros((24, 24), dtype=np.float32), 1)
    primary = SimpleNamespace(
        is_partial=False,
        min_burst=1,
        max_burst=1,
        burst_count=1,
        lines_per_burst=24,
        samples_per_burst=24,
        gdf_burst_geom=burst_geometry,
        fetch_dem=lambda *args, **kwargs: str(dem),
        compute_burst_offset=lambda *args, **kwargs: 0,
        compute_burst_overlap=lambda *args, **kwargs: 0,
        geocode_burst=lambda *args, **kwargs: (rows.copy(), cols.copy()),
        read_burst=lambda burst_idx, remove_invalid: image.copy(),
        calibration_factor=lambda *args, **kwargs: 1,
        phi_topo=lambda rg: np.zeros(rg.shape),
    )
    secondary = SimpleNamespace(**vars(primary))
    secondary.deramp_burst = lambda _: phase.copy()
    rr, cc = rows + 0.37, cols + 0.23
    settings = {} if kernel is None else {"warp_kernel": kernel}
    with (
        patch.object(process, "S1IWSwath", side_effect=[primary, secondary]),
        patch.object(process, "get_burst_geometry", return_value=burst_geometry),
        patch.object(process, "coregister", return_value=(rr, cc)),
        patch.object(process, "USE_CP", False),
        patch.object(process, "remove"),
    ):
        process.preprocess_insar_iw(
            "primary", "secondary", str(tmp_path), dem_buffer_arc_sec=0, **settings
        )
    deramped = image.copy()
    deramped *= np.exp(1j * phase)
    expected = remap(deramped, rr, cc, kernel or "lanczos")
    expected *= np.exp(-1j * remap(phase, rr, cc, "bicubic"))
    with rio.open(tmp_path / "secondary.tif") as ds:
        actual = ds.read(1)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
