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
    # uflt = flt.uniform_filter
    ndim = len(img.shape)
    ws = np.ones(ndim)
    ws[0] = dimaz
    ws[1] = dimrg
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


# @njit(parallel=True, nogil=True)
# def presum(img, m, n):
#     """
#     Computes the average of an image over m lines and n columns in parallel.

#     Args:
#         img (numpy.ndarray): Input image array of shape (height, width, ...).
#         m (int): Number of lines to average.
#         n (int): Number of columns to average.

#     Returns:
#         numpy.ndarray: Output image array of shape (height // m, width // n, ...).
#     """
#     height, width = img.shape[:2]
#     new_height = height // m
#     new_width = width // n

#     img_out = np.zeros((new_height, new_width) + img.shape[2:], dtype=img.dtype)

#     for i in prange(new_height):
#         for j in range(new_width):
#             # Compute the sum over the region
#             block_sum = 0
#             for x in range(i * m, (i + 1) * m):
#                 for y in range(j * n, (j + 1) * n):
#                     block_sum += img[x, y]

#             # Compute the average and assign to output image
#             img_out[i, j] = block_sum / (m * n)

#     return img_out


# TODO: add truncated sinc
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


def remap(img, rr, cc, kernel="bicubic"):
    """Resample an image using row, column lookup tables

    Args:
        img (array): image to resample (complex is allowed)
        rr (array): lookup table for row positions
        cc (array): lookup table for column positions
        kernel (str, optional): Kernel type ("nearest", "bilinear", "bicubic" -- 4 point, "bicubic6" -- 6 point). Defaults to "bicubic".

    Returns:
        array: Resampled image with same dimensions as rr and cc.
    """
    if np.iscomplexobj(img):
        return _remap(img.real, rr, cc, kernel) + 1j * _remap(img.imag, rr, cc, kernel)
    else:
        return _remap(img, rr, cc, kernel)


@njit(parallel=True, nogil=True)
def _remap(img, rr, cc, kernel="bicubic"):

    if rr.shape != cc.shape:
        raise ValueError("Coordinate arrays must have the same shape.")

    arr_out = np.full_like(rr, np.nan, dtype=img.dtype)
    if kernel == "nearest":
        ker = _ker_near
        H = 0
    elif kernel == "bilinear":
        ker = _ker_lin
        H = 0
    elif kernel == "bicubic":
        ker = _ker_cub
        H = 1
    elif kernel == "bicubic6":
        ker = _ker_cub6
        H = 2
    else:
        raise ValueError("Unknown interpolation type.")

    for idx in prange(len(rr.flat)):
        r = rr.flat[idx]
        c = cc.flat[idx]

        # change boundaries if using other kernels
        rmin = np.floor(r) - H
        rmax = np.ceil(r) + H
        cmin = np.floor(c) - H
        cmax = np.ceil(c) + H

        if np.isnan(r) | np.isnan(c):
            continue
        is_in_image = (r >= 0) & (r < img.shape[0]) & (c >= 0) & (c < img.shape[1])
        if not is_in_image:
            continue
        val = 0.0
        for i in range(int(rmin), int(rmax) + 1):
            for j in range(int(cmin), int(cmax) + 1):
                # using nearest neighbor on image border
                i2 = min(max(0, i), img.shape[0] - 1)
                j2 = min(max(0, j), img.shape[1] - 1)
                val += ker(r - i) * ker(c - j) * img[i2, j2]
        arr_out.flat[idx] = val
    return arr_out
