import pytest
import subprocess
import os
import time
import signal
import numpy as np
from eo_tools.S1.util import remap


def test_remap():
    shape_in = (512, 128)
    img = np.random.rand(*shape_in) + 1j * np.random.rand(*shape_in)

    shape_out = (1024, 2048)
    rr = np.random.rand(*shape_out) * (shape_in[0] - 1)
    cc = np.random.rand(*shape_out) * (shape_in[1] - 1)

    img_out = remap(img, rr, cc, kernel="bicubic")

    assert img_out.shape == rr.shape
    assert img_out.dtype == img.dtype
    assert np.all(~np.isnan(img_out))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize(
    "kernel", ["nearest", "bilinear", "bicubic", "bicubic6", "sinc", "lanczos"]
)
def test_remap_constant(dtype: type, kernel: str) -> None:
    """Preserve constants, identity and dtype, including clamped borders."""
    value = 2 + 3j if np.issubdtype(dtype, np.complexfloating) else 2.0
    image = np.full((20, 24), value, dtype=dtype)
    rr = np.array([[0, 0.2, 7.5, 19.99], [np.nan, np.inf, -0.1, 20]])
    cc = np.array([[0, 23.99, 8.2, 23.99], [1, 1, 1, 1]])
    output = remap(image, rr, cc, kernel)
    assert output.shape == rr.shape
    assert output.dtype == image.dtype
    assert np.iscomplexobj(output) == np.iscomplexobj(image)
    np.testing.assert_allclose(output[0], value, rtol=1e-12, atol=1e-12)
    assert np.isnan(output[1]).all()
    rows, cols = np.indices(image.shape, dtype=float)
    np.testing.assert_allclose(remap(image, rows, cols, kernel), image, atol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("kernel", ["sinc", "lanczos"])
def test_remap_wave(dtype: type, kernel: str) -> None:
    """Compare LUT output with independent normalized sinc sums on a strided image."""
    rows, cols = np.indices((80, 80), dtype=float)
    wave = np.exp(2j * np.pi * 0.13 * (rows + 0.7 * cols))
    image = (wave if np.issubdtype(dtype, np.complexfloating) else wave.real).astype(
        dtype
    )
    image = image[::2, ::2]
    rr, cc = np.array([15.37, 16.5]), np.array([17.11, 18.7])
    expected = []
    for r, c in zip(rr, cc):
        rows = np.arange(int(np.floor(r)) - 6, int(np.ceil(r)) + 7)
        cols = np.arange(int(np.floor(c)) - 6, int(np.ceil(c)) + 7)
        dr, dc = r - rows, c - cols
        rw, cw = np.sinc(dr), np.sinc(dc)
        if kernel == "lanczos":
            rw *= np.sinc(dr / 6)
            cw *= np.sinc(dc / 6)
        rw[np.abs(dr) > 6] = 0
        cw[np.abs(dc) > 6] = 0
        rw /= rw.sum()
        cw /= cw.sum()
        expected.append(rw @ image[np.ix_(rows, cols)] @ cw)
    actual = remap(image, rr, cc, kernel)
    assert actual.dtype == image.dtype
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("kernel", ["sinc", "lanczos"])
def test_remap_nan_border(dtype: type, kernel: str) -> None:
    """Border NaNs stay local; interior zeros are retained without a fallback."""
    image = np.full((40, 40), 2, dtype=dtype)
    image[:2] = complex(np.nan, np.nan) if np.iscomplexobj(image) else np.nan
    image[15, 15] = 0
    rr, cc = np.array([6.25, 7.25, 15]), np.full(3, 15.0)
    actual = remap(image, rr, cc, kernel)
    assert np.isnan(actual[0])
    np.testing.assert_allclose(actual[1:], [2, 0], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "kernel", ["nearest", "bilinear", "bicubic", "bicubic6", "sinc", "lanczos"]
)
def test_remap_shape_error(kernel: str) -> None:
    with pytest.raises(ValueError, match="same shape"):
        remap(np.ones((4, 4)), np.zeros(3), np.zeros(4), kernel)


def test_remap_kernel_error() -> None:
    with pytest.raises(ValueError, match="Unknown interpolation"):
        remap(np.ones((4, 4)), np.zeros(3), np.zeros(3), "unknown")
