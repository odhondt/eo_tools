import numpy as np
import scipy.ndimage.filters as flt


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

    uflt = flt.uniform_filter
    ndim = len(img.shape)
    ws = np.ones(ndim)
    ws[0] = dimaz
    ws[1] = dimrg
    if np.iscomplexobj(img):
        imgout = uflt(img.real, size=ws) + 1j * uflt(img.imag, size=ws)
    else:
        imgout = uflt(img.real, size=ws)
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