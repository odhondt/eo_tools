"""Check ESD phase corrections independently of the full processor."""

import numpy as np
import pytest
import rasterio as rio
from affine import Affine

from eo_tools.S1.core import fast_esd
from eo_tools.S1.process import _apply_fast_esd


@pytest.mark.parametrize("path", ["core", "raster"])
@pytest.mark.parametrize("burst_count", [2, 3])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_esd_phase(tmp_path, path, burst_count, dtype):
    """Correct known phase jumps while preserving magnitudes and missing data."""
    naz, nrg, overlap = 17, 13, 4
    phases = np.array([0.2, -0.4, 0.8])[:burst_count]
    rng = np.random.default_rng(17)
    range_phase = np.linspace(-0.1, 0.1, nrg)
    bursts = [
        (rng.uniform(0.5, 2, (naz, nrg)) * np.exp(1j * (phase + range_phase))).astype(
            dtype
        )
        for phase in phases
    ]
    for burst in bursts:
        burst[1, 0] = np.nan
        burst[-1, -1] = np.nan
    original = np.stack(bursts)

    if path == "core":
        fast_esd(bursts, overlap)
        result = np.stack(bursts)
    else:
        primary, secondary = tmp_path / "primary.tif", tmp_path / "secondary.tif"
        profile = dict(
            driver="GTiff",
            width=nrg,
            height=naz * burst_count,
            count=1,
            dtype=dtype,
            nodata=np.nan,
            transform=Affine.scale(1, 1),
        )
        with rio.open(primary, "w", **profile) as ds:
            ds.write(np.ones((naz * burst_count, nrg), dtype=dtype), 1)
        with rio.open(secondary, "w", **profile) as ds:
            ds.write(original.conj().reshape(naz * burst_count, nrg), 1)
        # Exercise original burst indices that start above one.
        _apply_fast_esd(
            str(primary), str(secondary), 4, 3 + burst_count, naz, nrg, overlap, "vv"
        )
        with rio.open(secondary) as ds:
            result = ds.read(1).conj().reshape(original.shape)

    # Analytical ramps interpolate half-jump corrections at overlap midpoints.
    jumps = phases[:-1] - phases[1:]
    fraction = (np.arange(naz) - overlap / 2) / (naz - 1 - overlap)
    for i in range(burst_count):
        left = -jumps[max(0, i - 1)] / 2
        right = jumps[min(i, burst_count - 2)] / 2
        ramp = left * (1 - fraction) + right * fraction
        expected = original[i] * np.exp(-1j * ramp[:, None])
        np.testing.assert_allclose(result[i], expected, rtol=3e-7, atol=3e-7)
    np.testing.assert_array_equal(np.isnan(result), np.isnan(original))
    np.testing.assert_allclose(abs(result), abs(original), rtol=3e-7)
    assert result.dtype == original.dtype
