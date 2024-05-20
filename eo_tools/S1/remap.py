import numpy as np
from numba import njit, prange


# fast parallel bicubic resampling
@njit(parallel=True, nogil=True, fastmath=True)
def remap(img, rr, cc):

    # bicubic kernel
    def ker(x):
        ax = np.abs(x)
        if ax < 1:
            return 1.5 * ax**3 - 2.5 * ax**2 + 1
        elif (ax >= 1) & (ax < 2):
            return -0.5 * ax**3 + 2.5 * ax**2 - 4 * ax + 2
        else:
            return 0.0

    if rr.shape != cc.shape:
        raise ValueError("Coordinate arrays must have the same shape.")

    arr_out = np.full_like(rr, np.nan, dtype=img.dtype)

    for idx in prange(len(rr.flat)):
        r = rr.flat[idx]
        c = cc.flat[idx]

        # change boundaries if using other kernels
        rmin = np.floor(r) - 1
        rmax = np.ceil(r) + 1
        cmin = np.floor(c) - 1
        cmax = np.ceil(c) + 1

        if np.isnan(r) | np.isnan(c):
            continue
        is_in_image = (r >= 0) & (r < img.shape[0]) & (c >= 0) & (c < img.shape[1])
        if not is_in_image:
            continue
        val = 0.0
        for i in range(int(rmin), int(rmax) + 1):
            for j in range(int(cmin), int(cmax) + 1):
                # using nearest neighbor on image border
                i2 = np.minimum(np.maximum(0, i), img.shape[0] - 1)
                j2 = np.minimum(np.maximum(0, j), img.shape[1] - 1)
                val += ker(r - i) * ker(c - j) * img[i2, j2]
        arr_out.flat[idx] = val
    return arr_out
