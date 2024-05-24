import numpy as np

from scipy.ndimage import convolve
from numba import njit, prange, cfunc


def boxcar(img, dimaz, dimrg):
    """Apply a boxcar filter to an image

    Parameters
    ----------
    img: complex or real array, shape (naz, nrg,...)
        input image, with arbitrary number of dimension
    dimaz, dimrg: floats
        size in azimuth and range of the filter

    Returns
    -------
    imgout: complex or real array, shape (naz, nrg,...)
        filtered image

    Note
    ----
    The filter is always along 2 dimensions (azimuth, range), please
    ensure to provide a valid image.
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
    """m by n presumming of an image

    Parameters
    ----------
    img: array, shape (naz, nrg,...)

    m,n: integer
        number of lines and columns to sum

    Returns
    -------
    out: array, shape(M, N,...)
        M and N are closest multiples of m and n
        to naz and nrg
    """
    if m > img.shape[0] or n > img.shape[1]:
        raise ValueError("Cannot presum with these parameters.")

    # TODO: write exception controlling size
    # and validity of parameters m, n
    M = int(np.floor(img.shape[0] / int(m)) * m)
    N = int(np.floor(img.shape[1] / int(n)) * n)
    img0 = img[:M, :N].copy()  # keep for readability
    s = img0[::m].copy()
    for i in range(1, m):
        s += img0[i::m]
    t = s[:, ::n]
    for j in range(1, n):
        t += s[:, j::n]
    return t / float(m * n)


# fast parallel resampling
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
