from functools import lru_cache

import numpy as np

from scipy.ndimage import convolve
from numba import njit, prange, cfunc


def boxcar(img, dimaz, dimrg):
    """
    Apply a boxcar filter to an image.

    Args:
        img (complex or real array): Input image with arbitrary number of dimensions, shape (naz, nrg, ...).
        dimaz (float): Size in azimuth of the filter.
        dimrg (float): Size in range of the filter.

    Returns:
        complex or real array: Filtered image, shape (naz, nrg, ...).

    Note:
        The filter is always applied along 2 dimensions (azimuth, range). Please ensure to provide a valid image.
    """

    # old implementation with uniform filter
    # uflt = uniform_filter
    # ndim = len(img.shape)
    # ws = np.ones(ndim)
    # ws[0] = dimaz
    # ws[1] = dimrg

    if (dimaz > 1) or (dimrg > 1):
        # avoid nan propagation
        msk = np.isnan(img)
        img_ = img.copy()
        img_[msk] = 0
        ker = np.ones((dimaz, dimrg)) / (dimaz * dimrg)
        if np.iscomplexobj(img_):
            # imgout = uflt(img_.real, size=ws) + 1j * uflt(img_.imag, size=ws)
            imgout = convolve(img_.real, ker) + 1j * convolve(img_.imag, ker)
            imgout[msk] = np.nan + 1j * np.nan
        else:
            # imgout = uflt(img_.real, size=ws)
            imgout = convolve(img_, ker)
            imgout[msk] = np.nan
        return imgout
    else:
        return img


def presum(img, m, n):
    """
    Computes the m by n presummed image.

    Args:
        img (array-like): Input image array with shape (naz, nrg,...).
        m (int): Number of lines to sum. Must be an integer >= 1.
        n (int): Number of columns to sum. Must be an integer >= 1.

    Raises:
        TypeError: If m or n are not integers.
        ValueError: If m or n are less than 1, or if m > img.shape[0] or n > img.shape[1].

    Returns:
        array: Presummed image array with shape (M, N,...), where M and N are the largest multiples of m and n that are less than or equal to img.shape[0] and img.shape[1], respectively.
    Note:
        Returns the input array if m==1 and n==1.
    """
    # Check if m and n are integers >= 1
    if not isinstance(m, int) or not isinstance(n, int):
        raise TypeError("Parameters m and n must be integers.")
    if m < 1 or n < 1:
        raise ValueError(
            "Parameters m and n must be integers greater than or equal to 1."
        )

    # Check if m and n are valid in relation to the image dimensions
    if m > img.shape[0] or n > img.shape[1]:
        raise ValueError(
            "Cannot presum with these parameters; m or n is too large for the image dimensions."
        )

    # skip if m = n = 1, avoids conditionals in calls
    if (m > 1) or (n > 1):
        M = (img.shape[0] // m) * m
        N = (img.shape[1] // n) * n

        img_trimmed = img[:M, :N]

        s = img_trimmed[::m].copy()  # Make a copy once for efficiency
        for i in range(1, m):
            s += img_trimmed[i::m]

        t = s[:, ::n].copy()
        for j in range(1, n):
            t += s[:, j::n]

        return t / float(m * n)
    else:
        return img


def remap(
    img: np.ndarray, rr: np.ndarray, cc: np.ndarray, kernel: str = "bicubic"
) -> np.ndarray:
    """Resample a real or complex image using row and column lookup tables.

    Args:
        img: Source image. Supports float32, float64, complex64 and complex128.
        rr: Row coordinates in the source image.
        cc: Column coordinates, with the same shape as rr.
        kernel: One of nearest, bilinear, bicubic (four-point), bicubic6
            (six-point), sinc, or lanczos6. Defaults to bicubic.

    Returns:
        Resampled image with the coordinates' shape and the source dtype.
        Nonfinite or out-of-image coordinates produce NaN. Stencil samples
        beyond the image boundary use the nearest edge pixel.

    Notes:
        Sinc is a normalized rectangular-window sinc; lanczos6 is normalized
        Lanczos-windowed sinc with radius six, not six taps. Both use 12-tap
        float64 coefficient tables with 4096 fractional-phase intervals and
        linear interpolation, so results approximate direct kernel evaluation.
        Tables are built lazily and cached per process. No fastmath is used.

        Real inputs use real arithmetic; complex inputs are processed in one
        pass. Source NaNs propagate through the interpolation stencil without
        a whole-image fallback. The tighter sinc/Lanczos stencil can leave
        more border pixels finite than the original wider implementation.
    """
    zero = np.complex128(0) if np.iscomplexobj(img) else np.float64(0)
    missing = complex(np.nan, np.nan) if np.iscomplexobj(img) else np.nan
    if kernel in ("sinc", "lanczos6"):
        return _remap_lut(img, rr, cc, _phase_table(kernel, 4096), zero, missing)
    return _remap(img, rr, cc, kernel, zero, missing)


# -----------------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------------


@cfunc("double(double)")
def _ker_near(x):
    ax = np.abs(x)
    if ax < 0.5:
        return 1.0
    elif ax == 0.5:
        return 0.5
    else:
        return 0.0


@cfunc("double(double)")
def _ker_lin(x):
    ax = np.abs(x)
    if ax < 1:
        return 1.0 - ax
    else:
        return 0.0


@cfunc("double(double)")
def _ker_cub(x):
    ax = np.abs(x)
    if ax < 1:
        return 1.5 * ax**3 - 2.5 * ax**2 + 1
    elif (ax >= 1) & (ax < 2):
        return -0.5 * ax**3 + 2.5 * ax**2 - 4 * ax + 2
    else:
        return 0.0


@cfunc("double(double)")
def _ker_cub6(x):
    """6-point bicubic kernel described in Keys81"""
    a = -0.5
    b = 0.5
    ax = np.abs(x)
    ax2 = ax**2
    ax3 = ax**3
    if ax < 1:
        return 4 * ax3 / 3 - 7 * ax2 / 3 + 1
    elif (ax >= 1) & (ax < 2):
        return -7 * ax3 / 12 + 3 * ax2 - 59 * ax / 12 + 15 / 6
    elif (ax >= 2) & (ax < 3):
        return ax3 / 12 - 2 * ax2 / 3 + 21 * ax / 12 - 3 / 2
    else:
        return 0.0


@cfunc("double(double)")
def _ker_sinc(x):
    """Truncated sinc interpolation kernel.

    Uses a symmetric window with a fixed number of lobes (samples)
    for truncation, defined internally.
    """
    ax = np.abs(x)
    n_lobes = 6  # number of lobes on each side (truncation half-width)

    if ax > n_lobes:
        return 0.0

    if ax < 1e-8:
        return 1.0  # avoid division by zero near the origin

    # Standard sinc(x) = sin(pi*x)/(pi*x)
    return np.sin(np.pi * x) / (np.pi * x)


@cfunc("double(double)")
def _ker_lanczos6(x):
    """Lanczos-windowed sinc kernel with six-pixel support."""
    ax = np.abs(x)
    if ax >= 6:
        return 0.0
    if ax < 1e-8:
        return 1.0

    pix = np.pi * x
    return (np.sin(pix) / pix) * (np.sin(pix / 6) / (pix / 6))


@njit(parallel=True, nogil=True)
def _remap(
    img: np.ndarray,
    rr: np.ndarray,
    cc: np.ndarray,
    kernel: str,
    zero: float | complex,
    missing: float | complex,
) -> np.ndarray:
    """Apply direct axis weights using separable, dtype-specialized sums."""
    if rr.shape != cc.shape:
        raise ValueError("Coordinate arrays must have the same shape.")
    normalize = False
    if kernel == "nearest":
        ker = _ker_near
        half_width = 0
    elif kernel == "bilinear":
        ker = _ker_lin
        half_width = 0
    elif kernel == "bicubic":
        ker = _ker_cub
        half_width = 1
    elif kernel == "bicubic6":
        ker = _ker_cub6
        half_width = 2
    elif kernel == "sinc":
        ker = _ker_sinc
        half_width = 6
        normalize = True
    elif kernel == "lanczos6":
        ker = _ker_lanczos6
        half_width = 6
        normalize = True
    else:
        raise ValueError("Unknown interpolation type.")
    output = np.full(rr.shape, missing, dtype=img.dtype)
    for index in prange(rr.size):
        r, c = rr.flat[index], cc.flat[index]
        if not (0 <= r < img.shape[0] and 0 <= c < img.shape[1]):
            continue
        if half_width == 6:
            rmin, rmax = int(np.ceil(r - 6)), int(np.floor(r + 6))
            cmin, cmax = int(np.ceil(c - 6)), int(np.floor(c + 6))
        else:
            rmin = int(np.floor(r)) - half_width
            rmax = int(np.ceil(r)) + half_width
            cmin = int(np.floor(c)) - half_width
            cmax = int(np.ceil(c)) + half_width
        rw = np.empty(14, dtype=np.float64)
        cw = np.empty(14, dtype=np.float64)
        rsum, csum = 0.0, 0.0
        for i in range(rmax - rmin + 1):
            rw[i] = ker(r - (rmin + i))
            rsum += rw[i]
        for j in range(cmax - cmin + 1):
            cw[j] = ker(c - (cmin + j))
            csum += cw[j]
        if normalize:
            for i in range(rmax - rmin + 1):
                rw[i] /= rsum
            for j in range(cmax - cmin + 1):
                cw[j] /= csum
        value = zero
        for i in range(rmax - rmin + 1):
            row = min(max(rmin + i, 0), img.shape[0] - 1)
            row_value = zero
            for j in range(cmax - cmin + 1):
                col = min(max(cmin + j, 0), img.shape[1] - 1)
                row_value += cw[j] * img[row, col]
            value += rw[i] * row_value
        output.flat[index] = value
    return output


@lru_cache(maxsize=8)
def _phase_table(kernel: str, phases: int) -> np.ndarray:
    """Build normalized 12-tap vectors over the closed fractional interval [0, 1]."""
    if kernel not in ("sinc", "lanczos6") or phases < 2:
        raise ValueError("LUT requires sinc/lanczos6 and at least two phase intervals")
    # Fixed indices floor(u)-5 ... floor(u)+6; the omitted integer endpoint
    # contributes only a floating-point sinc residue in the direct implementation.
    distances = np.linspace(0, 1, phases + 1)[:, None] - np.arange(-5, 7)
    weights = np.sinc(distances)
    if kernel == "lanczos6":
        weights *= np.sinc(distances / 6)
        weights[np.abs(distances) >= 6] = 0
    weights /= weights.sum(axis=1, keepdims=True)
    weights.setflags(write=False)
    return weights


@njit(parallel=True, nogil=True)
def _remap_lut(
    img: np.ndarray,
    rr: np.ndarray,
    cc: np.ndarray,
    table: np.ndarray,
    zero: float | complex,
    missing: float | complex,
) -> np.ndarray:
    """Apply linearly interpolated axis vectors using dtype-specialized sums."""
    if rr.shape != cc.shape:
        raise ValueError("Coordinate arrays must have the same shape.")
    output = np.full(rr.shape, missing, dtype=img.dtype)
    phases = table.shape[0] - 1
    for index in prange(rr.size):
        r, c = rr.flat[index], cc.flat[index]
        if not (0 <= r < img.shape[0] and 0 <= c < img.shape[1]):
            continue
        rbase, cbase = int(np.floor(r)), int(np.floor(c))
        rp, cp = (r - rbase) * phases, (c - cbase) * phases
        ri, ci = min(int(rp), phases - 1), min(int(cp), phases - 1)
        rf, cf = rp - ri, cp - ci
        rw = np.empty(12, dtype=np.float64)
        cw = np.empty(12, dtype=np.float64)
        for tap in range(12):
            rw[tap] = table[ri, tap] + rf * (table[ri + 1, tap] - table[ri, tap])
            cw[tap] = table[ci, tap] + cf * (table[ci + 1, tap] - table[ci, tap])
        value = zero
        for i in range(12):
            row = min(max(rbase - 5 + i, 0), img.shape[0] - 1)
            row_value = zero
            for j in range(12):
                col = min(max(cbase - 5 + j, 0), img.shape[1] - 1)
                row_value += cw[j] * img[row, col]
            value += rw[i] * row_value
        output.flat[index] = value
    return output
