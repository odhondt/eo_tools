import os
from pathlib import Path
import xmltodict
import numpy as np
import rasterio
from scipy.interpolate import CubicHermiteSpline
from dateutil.parser import isoparse
from eo_tools.dem import retrieve_dem
from shapely.geometry import box
from rasterio.enums import Resampling
from rasterio.warp import transform
from numba import jit, prange
from scipy.ndimage import map_coordinates
from pymap3d.ecef import geodetic2ecef
from rasterio.windows import Window
from scipy.interpolate import griddata, interpn

import logging

log = logging.getLogger(__name__)


# TODO: make Product class that contains metadata
# TODO: better function names
class S1IWSwath:
    def __init__(self, safe_dir, iw=1, pol="vv"):
        if not os.path.isdir(safe_dir):
            raise ValueError("Directory not found.")
        dir_tiff = Path(f"{safe_dir}/measurement/")
        dir_xml = Path(f"{safe_dir}/annotation/")
        pth_xml = dir_xml.glob(f"*iw{iw}*{pol}*.xml")
        pth_tiff = dir_tiff.glob(f"*iw{iw}*{pol}*.tiff")
        pth_xml = list(pth_xml)[0]
        self.pth_tiff = list(pth_tiff)[0]
        self.meta = read_metadata(pth_xml)
        log.info(f"S1IWSwath Initialization:")
        log.info(f"- Reading metadata file {pth_xml}")
        log.info(f"- Setting up raster path {self.pth_tiff}")

    def fetch_dem_burst(self, burst_idx=1, dir_dem="/tmp"):

        burst_info = self.meta["product"]["swathTiming"]
        lines_per_burst = int(burst_info["linesPerBurst"])
        first_line = (burst_idx - 1) * lines_per_burst

        name_dem = f"dem-b{burst_idx}-{self.pth_tiff.stem}.tiff"
        file_dem = f"{dir_dem}/{name_dem}"
        gcps, _ = read_gcps(
            self.pth_tiff, first_line=first_line, number_of_lines=lines_per_burst
        )
        auto_dem(file_dem, gcps)
        return file_dem

    # TODO check parameter validity
    def geocode_burst(self, file_dem, burst_idx=1):

        meta = self.meta

        # general info
        image_info = meta["product"]["imageAnnotation"]["imageInformation"]
        azimuth_time_interval = image_info["azimuthTimeInterval"]
        slant_range_time = image_info["slantRangeTime"]
        product_info = meta["product"]["generalAnnotation"]["productInformation"]
        range_sampling_rate = product_info["rangeSamplingRate"]

        # look for burst info
        burst_info = meta["product"]["swathTiming"]
        lines_per_burst = int(burst_info["linesPerBurst"])
        samples_per_burst = int(burst_info["samplesPerBurst"])
        burst_count = int(burst_info["burstList"]["@count"])
        if burst_idx > burst_count or burst_idx < 1:
            raise ValueError(f"Burst index must be between 1 and {burst_count}")
        burst = burst_info["burstList"]["burst"][burst_idx - 1]
        az_time = burst["azimuthTime"]

        # state vectors
        orbit_list = meta["product"]["generalAnnotation"]["orbitList"]
        state_vectors = orbit_list["orbit"]

        interp_orb, interp_orb_v = orbit_interpolator(state_vectors)

        log.info("DEM upsampling and extract coordinates")
        lat, lon, alt, dem_prof = load_dem_coords(file_dem, 1)

        log.info("Convert latitude, longitude & altitude to ECEF x, y & z")
        dem_x, dem_y, dem_z = lla_to_ecef(lat, lon, alt, dem_prof["crs"])

        log.info("Terrain correction (index computation)")
        tt0 = isoparse(state_vectors[0]["time"])
        t0_az = (isoparse(az_time) - tt0).total_seconds()
        dt_az = float(azimuth_time_interval)
        naz = int(lines_per_burst)
        nrg = int(samples_per_burst)
        t_interp = np.linspace(t0_az, t0_az + dt_az * naz, naz)
        orb = interp_orb(t_interp)
        orb_v = interp_orb_v(t_interp)

        az_geo, d2_geo = geocode_doppler_bisect(dem_x, dem_y, dem_z, orb, orb_v)

        # convert range - azimuth to pixel indices
        c0 = 299792458.0
        r0 = float(slant_range_time) * c0 / 2
        dr = c0 / (2 * float(range_sampling_rate))

        rg_geo = (np.sqrt(d2_geo) - r0) / dr

        cnd1 = (rg_geo >= 0) & (rg_geo < nrg)
        cnd2 = (az_geo >= 0) & (az_geo < naz)
        valid = cnd1 & cnd2

        rg = rg_geo.copy()
        az = az_geo.copy()

        rg[~valid] = np.nan
        az[~valid] = np.nan
        return az, rg

    def resample_burst(
        self,
        file_dem,
        file_out,
        az,
        rg,
        burst_idx=1,
        order=1,
        output_complex=True,
    ):
        meta = self.meta
        burst_info = meta["product"]["swathTiming"]
        lines_per_burst = int(burst_info["linesPerBurst"])

        first_line = (burst_idx - 1) * lines_per_burst
        log.info("Read burst")
        arr = self.read_burst(burst_idx)

        # replace nan to work with map_coordinates
        if output_complex:
            raster = np.nan_to_num(arr, nan=-9999)
        else:
            raster = np.abs(arr)
            np.nan_to_num(raster, copy=False, nan=-9999)

        # retrieve dem profile
        with rasterio.open(file_dem) as ds_dem:
            out_prof = ds_dem.profile.copy()

        log.info("Warp to match DEM geometry")
        width = out_prof["width"]
        height = out_prof["height"]
        wped = np.zeros_like(rg, dtype=raster.dtype)
        valid = (az != np.nan) & (rg != np.nan)
        wped[valid] = map_coordinates(
            raster, [az[valid], rg[valid]], order=order, prefilter=True
        )
        wped[~valid] = 0.0
        # wped[wped==np.nan] = 0.0
        wped[wped < 0.0] = 0.0
        wped = wped.reshape(height, width)

        log.info("Write output GeoTIFF")
        out_prof.update({"dtype": raster.dtype, "count": 1, "nodata": 0.0})
        with rasterio.open(file_out, "w", **out_prof) as dst:
            dst.write(wped, 1)

    def deramp_burst(self, burst_idx=1):

        meta = self.meta
        meta_image = meta["product"]["imageAnnotation"]
        meta_general = meta["product"]["generalAnnotation"]
        meta_burst = meta["product"]["swathTiming"]["burstList"]["burst"][burst_idx - 1]

        log.info("Computing TOPS deramping phase")

        c0 = 299792458.0
        lines_per_burst = int(meta["product"]["swathTiming"]["linesPerBurst"])
        az_time = meta_burst["azimuthTime"]
        az_dt = float(meta_image["imageInformation"]["azimuthTimeInterval"])
        range_sampling_rate = float(
            meta_general["productInformation"]["rangeSamplingRate"]
        )
        slant_range_time = float(meta_image["imageInformation"]["slantRangeTime"])
        nrg = int(meta_image["imageInformation"]["numberOfSamples"])
        kp = float(meta_general["productInformation"]["azimuthSteeringRate"])
        fc = float(meta_general["productInformation"]["radarFrequency"])
        rg_dt = 1 / range_sampling_rate

        orbit_list = meta_general["orbitList"]
        state_vectors = orbit_list["orbit"]
        _, orb_v = orbit_interpolator(state_vectors)
        tt0 = isoparse(state_vectors[0]["time"])
        t0_az = (isoparse(az_time) - tt0).total_seconds()
        t_mid = t0_az + az_dt * lines_per_burst / 2.0
        v_mid = orb_v(t_mid)
        ks = (2 * np.sqrt((v_mid**2).sum()) / c0) * fc * np.radians(kp)

        fm_rate_list = meta_general["azimuthFmRateList"]["azimuthFmRate"]
        fm_rate_times = [
            (isoparse(it["azimuthTime"]) - tt0).total_seconds() for it in fm_rate_list
        ]
        poly_fm_idx = np.argmin(np.abs(np.array(fm_rate_times) - t_mid))
        poly_fm_str = fm_rate_list[poly_fm_idx]["azimuthFmRatePolynomial"]["#text"]
        poly_fm_coeffs = np.array((poly_fm_str).split(" "), dtype="float64")

        rg_tau = slant_range_time + np.arange(nrg) * rg_dt

        def ka_fun(tau):
            return (
                poly_fm_coeffs[0]
                + poly_fm_coeffs[1] * (tau - slant_range_time)
                + poly_fm_coeffs[2] * (tau - slant_range_time) ** 2
            )

        ka = ka_fun(rg_tau)

        dc_list = meta["product"]["dopplerCentroid"]["dcEstimateList"]["dcEstimate"]
        dc_times = [
            (isoparse(it["azimuthTime"]) - tt0).total_seconds() for it in dc_list
        ]
        poly_dc_idx = np.argmin(np.abs(np.array(dc_times) - t_mid))
        poly_dc_str = dc_list[poly_dc_idx]["dataDcPolynomial"]["#text"]
        poly_dc_coeffs = np.array((poly_dc_str).split(" "), dtype="float64")

        def fdc_fun(tau):
            return (
                poly_dc_coeffs[0]
                + poly_dc_coeffs[1] * (tau - slant_range_time)
                + poly_dc_coeffs[2] * (tau - slant_range_time) ** 2
            )

        fdc = fdc_fun(rg_tau)
        kt = ka * ks / (ka - ks)
        eta = np.linspace(
            -az_dt * lines_per_burst / 2.0,
            az_dt * lines_per_burst / 2.0,
            lines_per_burst,
        )
        eta_c = -fdc / ka
        rg_mid = slant_range_time + 0.5 * nrg * rg_dt
        eta_mid = fdc_fun(rg_mid) / ka_fun(rg_mid)
        eta_ref = eta_c - eta_mid

        phi_deramp = -np.pi * kt[None] * (eta[:, None] - eta_ref[None]) ** 2
        return phi_deramp

    def read_burst(self, burst_idx=1, remove_border_noise=True):
        meta = self.meta
        burst_info = meta["product"]["swathTiming"]
        burst_data = burst_info["burstList"]["burst"][burst_idx - 1]
        lines_per_burst = int(burst_info["linesPerBurst"])

        first_line = (burst_idx - 1) * lines_per_burst
        arr = read_chunk(self.pth_tiff, first_line, lines_per_burst)

        if remove_border_noise:
            nodataval = np.nan
            fs_str = burst_data["firstValidSample"]["#text"]
            ls_str = burst_data["lastValidSample"]["#text"]
            first_sample_arr = np.array((fs_str).split(" "), dtype="int")
            last_sample_arr = np.array((ls_str).split(" "), dtype="int")
            for i in range(lines_per_burst):
                if first_sample_arr[i] > -1:
                    arr[i, : first_sample_arr[i]] = nodataval
                    arr[i, last_sample_arr[i] + 1 :] = nodataval
                else:
                    arr[i] = nodataval
        return arr

    # def nodata_mask_burst()

    # use metadata and range coordinate to compute topographic phase
    def phi_topo(self, rg):
        meta = self.meta
        image_info = meta["product"]["imageAnnotation"]["imageInformation"]
        slant_range_time = image_info["slantRangeTime"]
        product_info = meta["product"]["generalAnnotation"]["productInformation"]
        range_sampling_rate = product_info["rangeSamplingRate"]

        freq = float(product_info["radarFrequency"])
        c0 = 299792458.0
        lam = c0 / freq
        r0 = float(slant_range_time) * c0 / 2
        dr = c0 / (2 * float(range_sampling_rate))
        dist = rg * dr + r0

        return (4 * np.pi / lam) * dist

    # def mask_invalid_data(self, az, rg, burst_idx=1):
    # meta = self.meta
    # meta_burst = meta["product"]["swathTiming"]["burstList"]["burst"][burst_idx - 1]
    # first_valid = meta["firstValidSample"]
    # last_valid = meta["lastValidSample"]
    # for


# TODO: write a separate warping function
def coregister(
    arr_mst,
    arr_slv,
    az_mst,
    rg_mst,
    az_slv,
    rg_slv,
):

    naz, nrg = arr_mst.shape

    # TODO: add subsampling as parameters?
    rg_fx = np.linspace(0, nrg - 1, int(nrg / 16))
    az_fx = np.linspace(0, naz - 1, int(naz / 8))

    # coarse grid to perform a first interpolation
    cnd1 = (rg_mst >= 0) & (rg_mst < nrg)
    cnd2 = (az_mst >= 0) & (az_mst < naz)
    valid = (
        (az_mst != np.nan)
        & (rg_mst != np.nan)
        & (az_slv != np.nan)
        & (rg_slv != np.nan)
        & cnd1
        & cnd2
    )

    log.info("Interpolate irregular coordinates to coarse regular grid.")
    grg, gaz = np.meshgrid(rg_fx, az_fx)
    coords = np.stack((az_mst[valid], rg_mst[valid])).T
    # TODO try pre-triangulating to avoid repeating this step
    log.info("Processing azimuth.")
    az_re = griddata(coords, az_slv[valid], (gaz.flat, grg.flat), method="linear")
    log.info("Processing range.")
    rg_re = griddata(coords, rg_slv[valid], (gaz.flat, grg.flat), method="linear")

    log.info("Match gridded data to master dimensions.")
    grg2, gaz2 = np.meshgrid(np.arange(0, nrg), np.arange(0, naz))
    log.info("Processing azimuth.")
    az_w = interpn(
        (az_fx, rg_fx),
        az_re.reshape((len(az_fx), len(rg_fx))),
        (gaz2.flat, grg2.flat),
        method="linear",
    )
    log.info("Processing range.")
    rg_w = interpn(
        (az_fx, rg_fx),
        rg_re[az_re != np.nan].reshape((len(az_fx), len(rg_fx))),
        (gaz2.flat, grg2.flat),
        method="linear",
    )
    return az_w, rg_w, gaz2, grg2
    # log.info("Warp slave to master geometry.")
    # arr_slv_re = map_coordinates(arr_slv, (az_w, rg_w)).reshape((naz, nrg))
    # return arr_slv_re


def align(arr_mst, arr_slv, az, rg, order=3):
    log.info("Warp slave to master geometry.")
    arr_slv_re = map_coordinates(arr_slv, (az, rg), order=order).reshape(*arr_mst.shape)
    return arr_slv_re


# utility functions


def read_metadata(pth_xml):
    with open(pth_xml) as f:
        meta = xmltodict.parse(f.read())
    return meta


def read_gcps(pth_tiff, first_line=0, number_of_lines=1500):
    with rasterio.open(pth_tiff) as src:
        gcps, gcps_crs = src.gcps
        gcps_burst = [
            it
            for it in gcps
            if it.row >= first_line and it.row <= first_line + number_of_lines
        ]
    return gcps_burst, gcps_crs


def read_chunk(pth_tiff, first_line=0, number_of_lines=1500):

    with rasterio.open(pth_tiff) as src:
        arr = src.read(1, window=Window(0, first_line, src.width, number_of_lines))
    return arr


def orbit_interpolator(state_vectors):
    tt0 = isoparse(state_vectors[0]["time"])
    tt = [(isoparse(it["time"]) - tt0).total_seconds() for it in state_vectors]
    xx = [float(it["position"]["x"]) for it in state_vectors]
    yy = [float(it["position"]["y"]) for it in state_vectors]
    zz = [float(it["position"]["z"]) for it in state_vectors]

    vxx = [float(it["velocity"]["x"]) for it in state_vectors]
    vyy = [float(it["velocity"]["y"]) for it in state_vectors]
    vzz = [float(it["velocity"]["z"]) for it in state_vectors]

    interp_orb = CubicHermiteSpline(
        tt, np.array([xx, yy, zz]).T, np.array([vxx, vyy, vzz]).T
    )
    interp_orb_v = interp_orb.derivative(1)
    return interp_orb, interp_orb_v


def auto_dem(file_dem, gcps):
    minmax = lambda x: (x.min(), x.max())
    xmin, xmax = minmax(np.array([p.x for p in gcps]))
    ymin, ymax = minmax(np.array([p.y for p in gcps]))
    shp = box(xmin, ymin, xmax, ymax)

    if not os.path.exists(file_dem):
        retrieve_dem(shp, file_dem)
    else:
        log.info("--DEM already on disk")


# TODO add resampling options
def load_dem_coords(file_dem, upscale_factor=2):
    with rasterio.open(file_dem) as ds:
        # alt = ds.read(1)
        # on-the-fly resampling
        alt = ds.read(
            out_shape=(
                ds.count,
                int(ds.height * upscale_factor),
                int(ds.width * upscale_factor),
            ),
            resampling=Resampling.bilinear,
        )[0]

        # scale image transform
        dem_prof = ds.profile.copy()
        dem_trans = ds.transform * ds.transform.scale(
            (ds.width / alt.shape[-1]), (ds.height / alt.shape[-2])
        )

    # output lat-lon coordinates
    width, height = alt.shape[1], alt.shape[0]
    if dem_trans[1] > 1.0e-8 or dem_trans[3] > 1.0e-8:
        grid = np.meshgrid(np.arange(width), np.arange(height))
        lat, lon = rasterio.transform.xy(dem_trans, grid[1].ravel(), grid[0].ravel())
        lat = np.array(lat)
        lon = np.array(lon)
    else:
        # much faster
        ix, iy = np.arange(width), np.arange(height)
        lat_ = dem_trans[0] * ix + dem_trans[2]
        lon_ = dem_trans[4] * iy + dem_trans[5]
        lon = lon_[:, None] + np.zeros_like(alt)
        lat = lat_[None, :] + np.zeros_like(alt)

    dem_prof.update({"width": width, "height": height, "transform": dem_trans})
    return lat.ravel(), lon.ravel(), alt.ravel(), dem_prof


# TODO enforce DEM crs to WGS84
def lla_to_ecef(lat, lon, alt, dem_crs):
    # WGS84_points = (lat, lon, alt)

    # for some reason lat and lon are inverted respect to pymap3d docs
    WGS84_points = (lon, lat, alt)
    # TODO use DEM CRS
    # WGS84_crs = "EPSG:4326+3855"
    # ECEF_crs = "EPSG:4978"  # cartesian
    # dem_pts = transform(dem_crs, ECEF_crs, *WGS84_points)
    # dem_pts = (np.array(dem_pts[0]), np.array(dem_pts[1]), np.array(dem_pts[2]))
    dem_pts = geodetic2ecef(*WGS84_points)
    return dem_pts


@jit(parallel=True)
def geocode_doppler_bisect(dem_x, dem_y, dem_z, orb, orb_v, tol=1e-4, maxiter=10000):

    # convenience function
    def linterp(x, x0, x1, y0, y1):
        if x1 > x0:
            return y0 + (x - x0) * (y1 - y0) / (x1 - x0)
        else:
            return y0

    def doppler(t, x, y, z, orb, orb_v):
        t0 = int(np.floor(t))
        t1 = int(np.ceil(t))
        px = linterp(t, t0, t1, orb[t0, 0], orb[t1, 0])
        py = linterp(t, t0, t1, orb[t0, 1], orb[t1, 1])
        pz = linterp(t, t0, t1, orb[t0, 2], orb[t1, 2])
        vx = linterp(t, t0, t1, orb_v[t0, 0], orb_v[t1, 0])
        vy = linterp(t, t0, t1, orb_v[t0, 1], orb_v[t1, 1])
        vz = linterp(t, t0, t1, orb_v[t0, 2], orb_v[t1, 2])
        dx = x - px
        dy = y - py
        dz = z - pz
        d2 = dx**2 + dy**2 + dz**2
        return -(vx * dx + vy * dy + vz * dz) / np.sqrt(d2), dx, dy, dz

    def bisect(x, y, z, orb, orb_v, tol, maxiter):
        a = 0
        b = len(orb) - 1
        fa, dxa, dya, dza = doppler(a, x, y, z, orb, orb_v)
        fb, dxb, dyb, dzb = doppler(b, x, y, z, orb, orb_v)

        # exit if no solution
        if np.sign(fa * fb) > 0:
            return np.nan, np.nan, np.nan, np.nan

        if abs(fa) < tol:
            return a, dxa, dya, dza
        elif abs(fb) < tol:
            return b, dxb, dyb, dzb

        c = (a + b) / 2.0
        fc, dx, dy, dz = doppler(c, x, y, z, orb, orb_v)

        its = 0
        while abs(fc) > tol and its < maxiter:
            its = its + 1
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            c = (a + b) / 2.0
            fc, dx, dy, dz = doppler(c, x, y, z, orb, orb_v)
        return c, dx, dy, dz

    az_min = np.zeros_like(dem_x)
    d2_min = np.zeros_like(dem_x)

    for r in prange(dem_x.shape[0]):
        x = dem_x[r]
        y = dem_y[r]
        z = dem_z[r]
        t, dx, dy, dz = bisect(x, y, z, orb, orb_v, tol, maxiter)
        az_min[r] = t
        d2_min[r] = dx**2 + dy**2 + dz**2
    return az_min, d2_min


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

    # todo: write exception controlling size
    # and validity of parameters m, n
    M = int(np.floor(img.shape[0] / int(m)) * m)
    N = int(np.floor(img.shape[1] / int(n)) * n)
    img0 = img[:M, :N].copy()  # keep for readability
    s = img0[::m].copy()
    for i in range(1, m):
        s += img0[i::m]
    t = s[:, ::n]
    for j in range(
        1,
        n,
    ):
        t += s[:, j::n]
    return t / float(m * n)
