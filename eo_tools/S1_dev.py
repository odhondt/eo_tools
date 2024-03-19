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


def geocode_burst(safe_dir, iw=1, pol="vv", burst_idx=1, file_dem="autodem.tif"):
    if not os.isdir(safe_dir):
        raise ValueError("Directory not found.")

    pth_tiff = Path(f"{safe_dir}/measurement/").glob(f"*iw{iw}*{pol}*.tiff")
    pth_xml = Path(f"{safe_dir}/annotation/").glob(f"*iw{iw}*{pol}*.xml")
    pth_tiff = list(pth_tiff)[0]
    pth_xml = list(pth_xml)[0]

    meta = read_metadata(pth_xml)

    # general info
    image_info = meta["product"]["imageAnnotation"]["imageInformation"]
    azimuth_time_interval = image_info["azimuthTimeInterval"]
    slant_range_time = image_info["slantRangeTime"]
    range_pixel_spacing = image_info["rangePixelSpacing"]
    product_info = meta["product"]["generalAnnotation"]["productInformation"]
    range_sampling_rate = product_info["rangeSamplingRate"]
    radar_frequency = product_info["radarFrequency"]

    # look for burst info
    burst_info = meta["product"]["swathTiming"]
    lines_per_burst = int(burst_info["linesPerBurst"])
    samples_per_burst = int(burst_info["samplesPerBurst"])
    burst_count = int(burst_info["burstList"]["@count"])
    first_burst = burst_info["burstList"]["burst"][0]
    az_time = first_burst["azimuthTime"]

    first_line = (burst_idx - 1) * lines_per_burst
    im, gcps, gcps_crs = read_chunk(
        pth_tiff, first_line=first_line, number_of_lines=lines_per_burst
    )

    # state vectors
    orbit_list = meta["product"]["generalAnnotation"]["orbitList"]
    orbit_count = orbit_list["@count"]
    state_vectors = orbit_list["orbit"]

    interp_orb, interp_orb_v = orbit_interpolator(state_vectors)

    auto_dem(file_dem, gcps)
    lat, lon, alt, dem_prof = load_dem(file_dem)
    dem_x, dem_y, dem_z = lla_to_ecef(lat, lon, alt)

    # preparing geocoding
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
    # dr = float(range_pixel_spacing)
    dr = c0 / (2*float(range_sampling_rate))

    rg_geo = ((np.sqrt(d2_geo)-r0)/dr)

    cnd1 = (rg_geo >= 0) & (rg_geo < nrg)
    cnd2 = (az_geo >= 0) & (az_geo < naz)
    valid = cnd1 & cnd2

    rg = rg_geo.copy()
    az = az_geo.copy()

    rg[~valid] = np.nan
    az[~valid] = np.nan
    return rg, az


def read_metadata(pth_xml):
    # read product XML
    with open(pth_xml) as f:
        meta = xmltodict.parse(f.read())
    return meta


def read_chunk(pth_tiff, first_line=0, number_of_lines=1500):
    with rasterio.open(pth_tiff) as src:
        # prof = src.profile.copy()
        gcps, gcps_crs = src.gcps
        arr = np.log(
            np.abs(src.read(1, window=(0, first_line, src.width, number_of_lines))) + 1
        )
        # TODO: use filter?
        gcps_burst = [
            it
            for it in gcps
            if it.row >= first_line and it.row <= first_line + number_of_lines
        ]
    return arr, gcps_burst, gcps_crs


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

    # TODO handle better
    if not os.path.exists(file_dem):
        retrieve_dem(shp, file_dem)


# TODO add resampling options
def load_dem(file_dem, upscale_factor=2):
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
        )[0].ravel()

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
        lat_ = dem_trans[0] * ix + dem_trans[4]
        lon_ = dem_trans[2] * iy + dem_trans[5]
        lat = lat_[:, None] + np.zeros_like(alt)
        lon = lon_[None, :] + np.zeros_like(alt)

    dem_prof.update({"width": width, "height": height, "transform": dem_trans})
    return lat, lon, alt, dem_prof


def lla_to_ecef(lat, lon, alt):
    WGS84_points = (lat, lon, alt)
    # TODO: use DEM CRS
    WGS84_crs = "EPSG:4326+3855"
    ECEF_crs = "EPSG:4978"  # cartesian
    # dem_pts = transform(dem_crs, ECEF_crs, *WGS84_points)
    dem_pts = transform(WGS84_crs, ECEF_crs, *WGS84_points)
    dem_pts = (np.array(dem_pts[0]), np.array(dem_pts[1]), np.array(dem_pts[2]))
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
