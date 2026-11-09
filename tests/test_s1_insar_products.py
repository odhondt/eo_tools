"""Numerical checks for shared, chunked InSAR product computation."""

import numpy as np
import pytest
import rasterio as rio
from affine import Affine
from skimage.morphology import erosion

from eo_tools.S1.process import amplitude, coherence, insar_products, interferogram
from eo_tools.S1.util import boxcar, presum


@pytest.mark.parametrize(
    "looks, box, filtered, magnitude",
    [
        ([1, 1], [3, 5], True, True),
        ([2, 3], [4, 2], False, False),
        ([2, 3], [3, 5], True, True),
    ],
)
def test_products_chunk_parity(tmp_path, looks, box, filtered, magnitude):
    """Match whole-array arithmetic across chunk edges, NaNs and trimmed looks."""
    rng = np.random.default_rng(17)
    shape = (2071, 35)
    prm = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype("complex64")
    sec = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype("complex64")
    # Place missing data at the 1024-row input chunk boundary.
    prm[1022:1026, 14:16] = np.nan
    sec[2046:2050, 18] = np.nan
    transform = Affine.translation(7, 11) * Affine.scale(2, 3)
    for name, data in (("prm", prm), ("sec", sec)):
        with rio.open(
            tmp_path / f"{name}.tif",
            "w",
            driver="GTiff",
            width=shape[1],
            height=shape[0],
            count=1,
            dtype=data.dtype,
            transform=transform,
        ) as ds:
            ds.write(data, 1)
    paths = {
        key: tmp_path / f"{key}.tif" for key in ("coh", "ifg", "amp_prm", "amp_sec")
    }
    insar_products(
        tmp_path / "prm.tif",
        tmp_path / "sec.tif",
        **{f"{key}_file": path for key, path in paths.items()},
        multilook=looks,
        box_size=box,
        filter_ifg=filtered,
        magnitude=magnitude,
    )
    corr = presum(prm * sec.conj(), *looks)
    corr_box = boxcar(corr, *box)
    pow1 = boxcar(presum(np.nan_to_num((prm * prm.conj()).real), *looks), *box)
    pow2 = boxcar(presum(np.nan_to_num((sec * sec.conj()).real), *looks), *box)
    coh = corr_box / np.sqrt(pow1) / np.sqrt(pow2)
    if magnitude:
        coh = abs(coh)
    valid = erosion(~np.isnan(corr), footprint=np.ones(box))
    expected = {
        "coh": np.where(valid, coh, np.nan),
        "ifg": corr_box if filtered else corr,
        "amp_prm": np.sqrt(presum(prm.real**2 + prm.imag**2, *looks)),
        "amp_sec": np.sqrt(presum(sec.real**2 + sec.imag**2, *looks)),
    }
    for key, path in paths.items():
        with rio.open(path) as ds:
            result = ds.read(1)
            np.testing.assert_array_equal(np.isnan(result), np.isnan(expected[key]))
            np.testing.assert_allclose(result, expected[key], atol=4e-7, rtol=2e-6)
            assert ds.transform == transform * Affine.scale(looks[1], looks[0])
            assert ds.dtypes[0] == expected[key].dtype.name


@pytest.mark.parametrize("role", ["prm", "sec"])
def test_amplitude_selection(tmp_path, role):
    """Amplitude-only requests need only the selected acquisition."""
    arr = np.arange(35, dtype="float32").reshape(5, 7).astype("complex64") * (1 + 2j)
    arr[0, 0] = np.nan
    source = tmp_path / f"{role}.tif"
    with rio.open(
        source,
        "w",
        driver="GTiff",
        width=7,
        height=5,
        count=1,
        dtype=arr.dtype,
        transform=Affine.scale(1, 1),
    ) as ds:
        ds.write(arr, 1)
    output = tmp_path / "selected.tif"
    insar_products(
        tmp_path / "prm.tif",
        tmp_path / "sec.tif",
        **{f"amp_{role}_file": output},
        multilook=[2, 3],
    )
    standalone = tmp_path / "standalone.tif"
    amplitude(source, standalone, multilook=[2, 3])
    with rio.open(output) as selected, rio.open(standalone) as separate:
        power = presum(arr.real**2 + arr.imag**2, 2, 3)
        np.testing.assert_allclose(selected.read(1), np.sqrt(power), rtol=3e-7)
        np.testing.assert_allclose(selected.read(1) ** 2, power, rtol=3e-7)
        np.testing.assert_array_equal(selected.read(1), separate.read(1))


def test_pair_wrappers(tmp_path):
    """Existing coherence and interferogram APIs retain their output choices."""
    data = np.ones((12, 14), dtype="complex64") * (1 + 2j)
    for name in ("prm", "sec"):
        with rio.open(
            tmp_path / f"{name}.tif",
            "w",
            driver="GTiff",
            width=14,
            height=12,
            count=1,
            dtype=data.dtype,
            transform=Affine.scale(1, 1),
        ) as ds:
            ds.write(data, 1)
    prm, sec = tmp_path / "prm.tif", tmp_path / "sec.tif"
    coherence(
        prm,
        sec,
        tmp_path / "coh.tif",
        multilook=[2, 3],
        complex_ifg_file=tmp_path / "filtered.tif",
        filter_ifg=False,
    )
    interferogram(prm, sec, tmp_path / "ifg.tif", multilook=[2, 3])
    with rio.open(tmp_path / "coh.tif") as ds:
        np.testing.assert_allclose(ds.read(1), 1, atol=2e-7)
    with (
        rio.open(tmp_path / "ifg.tif") as ds,
        rio.open(tmp_path / "filtered.tif") as other,
    ):
        np.testing.assert_array_equal(ds.read(1), other.read(1))
        np.testing.assert_allclose(ds.read(1), 5)
