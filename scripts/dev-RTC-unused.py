import numpy as np
from numba import njit, prange

# project area in the SAR geometry with interpolation
@njit(nogil=True, parallel=True, cache=True)
def project_area_to_sar(naz, nrg, azp, rgp, gamma):

    # barycentric coordinates in a triangle
    def bary(p, a, b, c):
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        l1 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / det
        l2 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / det
        l3 = 1 - l1 - l2
        return l1, l2, l3

    # test if point is in triangle
    def is_in_tri(l1, l2):
        return (l1 >= 0) and (l2 >= 0) and (l1 + l2 < 1)

    # linear barycentric interpolation
    def interp(v1, v2, v3, l1, l2, l3):
        return l1 * v1 + l2 * v2 + l3 * v3

    gamma_proj = np.zeros((naz, nrg))
    p = np.zeros(2)
    nl, nc = azp.shape
    # - loop on DEM
    for i in prange(0, nl - 1):
        for j in range(0, nc - 1):
            # - for each 4 neighborhood
            aa = azp[i : i + 2, j : j + 2].flatten()  # .ravel()
            rr = rgp[i : i + 2, j : j + 2].flatten()  # .ravel()
            gg = gamma[i : i + 2, j : j + 2].flatten()  # .ravel()
            # - collect triangle vertices
            xx = np.vstack((aa, rr)).T
            # yy = np.vstack((aas, rrs)).T
            if np.isnan(xx).any() or np.isnan(gg).any():
                continue
            # - compute bounding box in the primary grid
            amin, amax = np.floor(aa.min()), np.ceil(aa.max())
            rmin, rmax = np.floor(rr.min()), np.ceil(rr.max())
            amin = np.maximum(amin, 0)
            rmin = np.maximum(rmin, 0)
            amax = np.minimum(amax, naz - 1)
            rmax = np.minimum(rmax, nrg - 1)
            # - loop on integer positions based on box
            for a in range(int(amin), int(amax) + 1):
                for r in range(int(rmin), int(rmax) + 1):
                    # - separate into 2 triangles
                    # - test if each point falls into triangle 1 or 2
                    # - interpolate the secondary range and azimuth using triangle vertices
                    # p = np.array([a, r])
                    p[0] = a
                    p[1] = r
                    l1, l2, l3 = bary(p, xx[0], xx[1], xx[2])
                    if is_in_tri(l1, l2):
                        gamma_proj[a, r] += interp(gg[0], gg[1], gg[2], l1, l2, l3)
                        # npts[a, r] += 1
                    l1, l2, l3 = bary(p, xx[3], xx[1], xx[2])
                    if is_in_tri(l1, l2):
                        gamma_proj[a, r] += interp(gg[3], gg[1], gg[2], l1, l2, l3)
                        # npts[a, r] += 1

    return gamma_proj


# project area in the SAR geometry with interpolation
@njit(nogil=True, parallel=True, cache=True)
def project_area_to_sar_vec(naz, nrg, azp, rgp, nv, lv):

    # barycentric coordinates in a triangle
    def bary(p, a, b, c):
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        l1 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / det
        l2 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / det
        l3 = 1 - l1 - l2
        return l1, l2, l3

    # test if point is in triangle
    def is_in_tri(l1, l2):
        return (l1 >= 0) and (l2 >= 0) and (l1 + l2 < 1)

    # linear barycentric interpolation
    def interp(v1, v2, v3, l1, l2, l3):
        return l1 * v1 + l2 * v2 + l3 * v3

    gamma_proj = np.zeros((naz, nrg))
    p = np.zeros(2)
    nl, nc = azp.shape
    # - loop on DEM
    for i in prange(0, nl - 1):
        for j in range(0, nc - 1):
            # - for each 4 neighborhood
            aa = azp[i : i + 2, j : j + 2].flatten()
            rr = rgp[i : i + 2, j : j + 2].flatten()
            nn = nv[i : i + 2, j : j + 2].copy().reshape((4, 3))
            ll = lv[i : i + 2, j : j + 2].copy().reshape((4, 3))
            # - collect triangle vertices
            xx = np.vstack((aa, rr)).T
            # yy = np.vstack((aas, rrs)).T
            if np.isnan(xx).any():
                continue
            # - compute bounding box in the primary grid
            amin, amax = np.floor(aa.min()), np.ceil(aa.max())
            rmin, rmax = np.floor(rr.min()), np.ceil(rr.max())
            amin = np.maximum(amin, 0)
            rmin = np.maximum(rmin, 0)
            amax = np.minimum(amax, naz - 1)
            rmax = np.minimum(rmax, nrg - 1)
            # - loop on integer positions based on box
            for a in range(int(amin), int(amax) + 1):
                for r in range(int(rmin), int(rmax) + 1):
                    # - separate into 2 triangles
                    # - test if each point falls into triangle 1 or 2
                    # - interpolate normal and look vectors triangle vertices
                    # - Compute the area (inverse of the tangent)
                    # p = np.array([a, r])
                    p[0] = a
                    p[1] = r
                    l1, l2, l3 = bary(p, xx[0], xx[1], xx[2])
                    if is_in_tri(l1, l2):
                        li = interp(ll[0], ll[1], ll[2], l1, l2, l3)
                        ni = interp(nn[0], nn[1], nn[2], l1, l2, l3)
                        # area is the inverse of the tangent
                        area = (ni * li).sum() / np.sqrt((np.cross(ni, li) ** 2).sum())
                        # do not apply if in shadow
                        area = area if area >= 1e-10 else 1
                        gamma_proj[a, r] += area
                    l1, l2, l3 = bary(p, xx[3], xx[1], xx[2])
                    if is_in_tri(l1, l2):
                        li = interp(ll[3], ll[1], ll[2], l1, l2, l3)
                        ni = interp(nn[3], nn[1], nn[2], l1, l2, l3)
                        # area is the inverse of the tangent
                        area = (ni * li).sum() / np.sqrt((np.cross(ni, li) ** 2).sum())
                        # do not apply if in shadow
                        area = area if area >= 1e-10 else 1
                        gamma_proj[a, r] += area

    return gamma_proj


## The code below implements basic incidence angle tangent estimation
# compute x, y gradients
# v1 = np.zeros_like(dem_x)[..., None] + np.zeros((1, 1, 3))
# v1[:, :-1, 0] = np.diff(dem_x)
# v1[:, :-1, 1] = np.diff(dem_y)
# v1[:, :-1, 2] = np.diff(dem_z)

# v2 = np.zeros_like(dem_x)[..., None] + np.zeros((1, 1, 3))
# v2[:-1, :, 0] = np.diff(dem_x, axis=0)
# v2[:-1, :, 1] = np.diff(dem_y, axis=0)
# v2[:-1, :, 2] = np.diff(dem_z, axis=0)

# normalized look vector
# lv = np.dstack((dx, dy, dz))
# free memory ?
# del dx, dy, dz
# lv /= np.sqrt(np.sum((lv**2), 2))[..., None]

# normalized normal vector
# nv = np.cross(v1, v2)
# free memory ?
# del v1, v2
# nv /= np.sqrt(np.sum(nv**2, 2))[..., None]
# nv = np.nan_to_num(nv)

# Area is the inverse of the tangent
# gamma_t =  (nv * lv).sum(2) / np.sqrt((np.cross(nv, lv) ** 2).sum(2))
# gamma_t = gamma_t.clip(1e-10)