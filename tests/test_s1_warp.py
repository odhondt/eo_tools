import numpy as np
import pytest

from eo_tools.S1.core import align
from eo_tools.S1.util import remap


@pytest.mark.parametrize("kernel", ["bilinear", "bicubic", "bicubic6"])
def test_remap_ramp(kernel: str) -> None:
    """Fractional coordinates reproduce an affine surface inside the image."""
    rows, cols = np.indices((24, 24), dtype=float)
    image = 2 * rows - 3 * cols + 5
    rr = np.array([[8.1, 9.37], [10.5, 11.9]])
    cc = np.array([[12.8, 11.23], [10.5, 9.1]])

    actual = remap(image, rr, cc, kernel)

    np.testing.assert_allclose(actual, 2 * rr - 3 * cc + 5, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize(
    "kernel", ["nearest", "bilinear", "bicubic", "bicubic6", "sinc", "lanczos"]
)
def test_remap_complex(dtype: type, kernel: str) -> None:
    """Complex interpolation agrees with interpolating each component separately."""
    rows, cols = np.indices((24, 24), dtype=float)
    image = (np.sin(0.2 * rows) + 1j * np.cos(0.3 * cols)).astype(dtype)
    rr = np.array([8.37, 10.5, 12.1])
    cc = np.array([11.23, 9.8, 8.6])

    actual = remap(image, rr, cc, kernel)
    expected = remap(image.real, rr, cc, kernel) + 1j * remap(
        image.imag, rr, cc, kernel
    )

    tolerance = 1e-6 if dtype == np.complex64 else 1e-12
    assert actual.dtype == image.dtype
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    "kernel", [None, "nearest", "bilinear", "bicubic", "bicubic6", "sinc", "lanczos"]
)
def test_align_kernel(kernel: str | None) -> None:
    """Alignment uses Lanczos by default and respects an explicit kernel."""
    rows, cols = np.indices((24, 24), dtype=float)
    image = np.exp(1j * (0.03 * rows**2 + 0.02 * cols**2)).astype(np.complex64)
    rr, cc = rows[8:16, 8:16] + 0.37, cols[8:16, 8:16] + 0.23

    actual = align(image, rr, cc) if kernel is None else align(image, rr, cc, kernel)
    expected = remap(image, rr, cc, kernel or "lanczos")

    np.testing.assert_array_equal(actual, expected)
