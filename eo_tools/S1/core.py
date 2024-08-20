import os
from pathlib import Path
import xmltodict
import numpy as np
import rasterio
from math import floor, ceil
from scipy.interpolate import CubicHermiteSpline
from numpy.polynomial import Polynomial
from dateutil.parser import isoparse
from eo_tools.dem import retrieve_dem
from eo_tools.S1.util import remap
from eo_tools.auxils import get_burst_geometry
from shapely.geometry import box
from rasterio.enums import Resampling
from numba import njit, prange, float64, int64
from scipy.ndimage import map_coordinates
from rasterio.windows import Window
from pyproj import Transformer
from joblib import Parallel, delayed

from pyroSAR import identify
from xmltodict import parse
from zipfile import ZipFile


from eo_tools.bench import timeit

# from cv2 import remap, INTER_LANCZOS4, INTER_LINEAR, INTER_CUBIC
# WARP_RELATIVE_MAP only in dev version for now

# benchmark & debug


import logging

log = logging.getLogger(__name__)


# TODO: Goldstein, LOS displacement
class S1IWSwath:
    """Class that contains metadata & orbit related to a Sentinel-1 subswath for a IW product. Member functions allow to pre-process individual bursts for further TOPS-InSAR processing. It includes:

    - DEM retrieval (only SRTM 1sec for now)
    - Back-geocoding to the DEM grid (by computing lookup tables)
    - Computing the azimuth deramping correction term
    - Read the raster burst from the SLC tiff file
    - Computing the topographic phase from slant range values
    """

    def __init__(self, safe_dir, iw=1, pol="vv", dir_orb="/tmp"):
        """Object intialization

        Args:
            safe_dir (str): Directory containing the (unzipped) product.
            iw (int, optional): Subswath index (1 to 3). Defaults to 1.
            pol (str, optional): Polarization ("vv" or "vh"). Defaults to "vv".
            dir_orb (str, optional): Directory containing orbits. Defaults to "/tmp".
        """
        if not os.path.isdir(safe_dir):
            raise ValueError("Directory not found.")

        if not isinstance(iw, int) or iw < 1 or iw > 3:
            raise ValueError("Parameter 'iw' must an int be between 1 and 3")

        if pol not in ["vv", "vh"]:
            raise ValueError("Parameter 'pol' must be either 'vv' or 'vh'.")

        # check product type using dir name
        parts = Path(safe_dir).stem.split("_")
        if not all(["S1" in parts[0], parts[1] == "IW", parts[2] == "SLC"]):
            raise RuntimeError(
                "Unexpected product name. Should start with S1{A,B}_IW_SLC."
            )

        # read raster file
        dir_tiff = Path(safe_dir) / "measurement"
        pth_tiff = dir_tiff.glob(f"*iw{iw}*{pol}*.tiff")
        try:
            self.pth_tiff = list(pth_tiff)[0]
        except IndexError:
            raise FileNotFoundError("Tiff file is missing.")

        # read metadata file
        dir_xml = Path(safe_dir) / "annotation"
        pth_xml = dir_xml.glob(f"*iw{iw}*{pol}*.xml")
        try:
            pth_xml = list(pth_xml)[0]
        except IndexError:
            raise FileNotFoundError("Metadata file is missing.")

        # read calibration file
        dir_cal = Path(safe_dir) / "annotation" / "calibration"
        pth_cal = dir_cal.glob(f"calibration*iw{iw}*{pol}*.xml")
        try:
            pth_cal = list(pth_cal)[0]
        except IndexError:
            raise FileNotFoundError("Calibration file is missing.")

        self.meta = read_metadata(pth_xml)
        self.start_time = self.meta["product"]["adsHeader"]["startTime"]
        burst_info = self.meta["product"]["swathTiming"]
        self.lines_per_burst = int(burst_info["linesPerBurst"])
        self.samples_per_burst = int(burst_info["samplesPerBurst"])
        self.burst_count = int(burst_info["burstList"]["@count"])

        # extract beta_nought to rescale data
        # TODO: more advanced calibration (sigma nought)
        calinfo = read_metadata(pth_cal)
        calvec = calinfo["calibration"]["calibrationVectorList"]["calibrationVector"]
        BN_str = calvec[0]["betaNought"]["#text"]
        self.beta_nought = float(BN_str.split(" ")[0])

        # read burst geometries
        self.gdf_burst_geom = get_burst_geometry(
            path=safe_dir, target_subswaths=f"IW{iw}", polarization=pol.upper()
        )
        if self.gdf_burst_geom.empty:
            raise RuntimeError("Invalid product: no burst geometry was found.")

        log.info(f"S1IWSwath Initialization:")
        log.info(f"- Reading metadata file {pth_xml}")
        log.info(f"- Reading calibration file {pth_cal}")
        log.info(f"- Setting up raster path {self.pth_tiff}")
        log.info(f"- Looking for available OSV (Orbit State Vectors)")

        # read state vectors
        product = identify(safe_dir)
        zip_orb = product.getOSV(dir_orb, returnMatch=True)
        if not zip_orb:
            raise RuntimeError("No orbit file available for this product")

        if "POEORB" in zip_orb:
            log.info("-- Precise orbit found")
        elif "RESORB" in zip_orb:
            log.info("-- Restituted orbit found")
        else:
            raise RuntimeError("Unknown orbit file")

        with ZipFile(zip_orb) as zf:
            file_orb = zf.namelist()[0]
            with zf.open(file_orb) as f:
                orbdict = parse(f.read())
        orbdata = orbdict["Earth_Explorer_File"]["Data_Block"]["List_of_OSVs"]["OSV"]
        self.state_vectors = {}
        t0 = isoparse(orbdata[0]["UTC"][4:])
        self.state_vectors["t0"] = t0
        self.state_vectors["t"] = np.array(
            [(isoparse(it["UTC"][4:]) - t0).total_seconds() for it in orbdata]
        )
        self.state_vectors["x"] = np.array([float(it["X"]["#text"]) for it in orbdata])
        self.state_vectors["y"] = np.array([float(it["Y"]["#text"]) for it in orbdata])
        self.state_vectors["z"] = np.array([float(it["Z"]["#text"]) for it in orbdata])
        self.state_vectors["vx"] = np.array(
            [float(it["VX"]["#text"]) for it in orbdata]
        )
        self.state_vectors["vy"] = np.array(
            [float(it["VY"]["#text"]) for it in orbdata]
        )
        self.state_vectors["vz"] = np.array(
            [float(it["VZ"]["#text"]) for it in orbdata]
        )

    def fetch_dem(
        self,
        min_burst=1,
        max_burst=None,
        dir_dem="/tmp",
        buffer_arc_sec=40,
        force_download=False,
        upscale_factor=1,
    ):
        """Downloads the DEM for a given burst range

        Args:
            min_burst (int, optional): Minimum burst index. Defaults to 1.
            max_burst (int, optional): Maximum burst index. If None, set to last burst. Defaults to None.
            dir_dem (str, optional): Directory to store DEM files. Defaults to "/tmp".
            buffer_arc_sec (int, optional): Enlarges the bounding box computed using burst geometries by a number of arc seconds. Defaults to 40.
            force_download (bool, optional): Force downloading the file to even if a DEM is already present on disk. Defaults to False.

        Returns:
            str: path to the downloaded file
        """

        if not max_burst:
            max_burst_ = self.burst_count
        else:
            max_burst_ = max_burst

        if min_burst < 1 or min_burst > self.burst_count:
            raise ValueError(
                f"Invalid min burst index (must be between 1 and {self.burst_count})"
            )
        if max_burst_ < 1 or max_burst_ > self.burst_count:
            raise ValueError(
                f"Invalid max burst index (must be between 1 and {self.burst_count})"
            )
        if max_burst_ < min_burst:
            raise ValueError("max_burst must be >= min_burst")

        if min_burst < max_burst_:
            name_dem = f"dem-b{min_burst}-b{max_burst_}-{self.pth_tiff.stem}.tiff"
        else:
            name_dem = f"dem-b{min_burst}-{self.pth_tiff.stem}.tiff"
        file_dem = f"{dir_dem}/{name_dem}"

        # use buffer bounds around union of burst geometries
        geom_all = self.gdf_burst_geom
        geom_sub = (
            geom_all[
                (geom_all["burst"] >= min_burst) & (geom_all["burst"] <= max_burst)
            ]
            .union_all()
            .buffer(buffer_arc_sec / 3600)
        )
        shp = box(*geom_sub.bounds)

        if not os.path.exists(file_dem) or force_download:
            retrieve_dem(
                shp, file_dem, dem_name="nasadem", upscale_factor=upscale_factor
            )
        else:
            log.info("--DEM already on disk")

        return file_dem

    # kept for backwards compatibility
    @timeit
    def fetch_dem_burst(
        self, burst_idx=1, dir_dem="/tmp", buffer_arc_sec=40, force_download=False
    ):
        """Downloads the DEM for a given burst

        Args:
            burst_idx (int, optional): Burst index. Defaults to 1.
            dir_dem (str, optional): Directory to store DEM files. Defaults to "/tmp".
            buffer_arc_sec (int, optional): Enlarges the bounding box computed using GPCS by a number of arc seconds. Defaults to 40.
            force_download (bool, optional): Force downloading the file to even if a DEM is already present on disk. Defaults to False.

        Returns:
            str: path to the downloaded file
        """

        return self.fetch_dem(burst_idx, burst_idx, dir_dem, buffer_arc_sec, force_download)



    @timeit
    def geocode_burst(self, file_dem, burst_idx=1, dem_upsampling=1):
        """Computes azimuth-range lookup tables for each pixel of the DEM by solving the Range Doppler equations.

        Args:
            file_dem (str): path to the DEM
            burst_idx (int, optional): Burst index. Defaults to 1.
            dem_upsampling (int, optional): DEM upsampling to increase the resolution of the geocoded image. Defaults to 2.

        Returns:
            (array, array): azimuth slant range indices. Arrays have the shape of the DEM.
        """

        if burst_idx < 1 or burst_idx > self.burst_count:
            raise ValueError(
                f"Invalid burst index (must be between 1 and {self.burst_count})"
            )

        if dem_upsampling < 0:
            raise ValueError("dem_upsampling must be > 0")

        meta = self.meta

        # general info
        image_info = meta["product"]["imageAnnotation"]["imageInformation"]
        azimuth_time_interval = image_info["azimuthTimeInterval"]
        slant_range_time = image_info["slantRangeTime"]
        product_info = meta["product"]["generalAnnotation"]["productInformation"]
        range_sampling_rate = product_info["rangeSamplingRate"]

        # look for burst info
        burst_info = meta["product"]["swathTiming"]
        if burst_idx > self.burst_count or burst_idx < 1:
            raise ValueError(f"Burst index must be between 1 and {self.burst_count}")
        burst = burst_info["burstList"]["burst"][burst_idx - 1]
        az_time = burst["azimuthTime"]

        # state vectors
        # orbit_list = meta["product"]["generalAnnotation"]["orbitList"]
        # state_vectors = orbit_list["orbit"]

        if dem_upsampling != 1:
            log.info("DEM resampling and extract coordinates")
        else:
            log.info("Extract DEM coordinates")
        lat, lon, alt, dem_prof = load_dem_coords(file_dem, dem_upsampling)

        log.info("Convert latitude, longitude & altitude to ECEF x, y & z")
        dem_x, dem_y, dem_z = lla_to_ecef(lat, lon, alt, dem_prof["crs"])

        tt0 = self.state_vectors["t0"]
        t0_az = (isoparse(az_time) - tt0).total_seconds()
        dt_az = float(azimuth_time_interval)
        naz = self.lines_per_burst
        nrg = self.samples_per_burst

        t_end_burst = t0_az + dt_az * naz
        t_sv_burst = self.state_vectors["t"]

        # crop a few minutes before and after burst
        t_spacing = 10
        t_pad = t_spacing * 36
        cnd = (t_sv_burst > t0_az - t_pad) & (t_sv_burst < t_end_burst + t_pad)

        state_vectors = {k: v[cnd] for k, v in self.state_vectors.items() if k != "t0"}

        # TODO (optional) integrate other models to orbit interpolation
        interp_pos, interp_vel = sv_interpolator(state_vectors)
        # interp_pos, interp_vel = sv_interpolator_poly(state_vectors)

        log.info("Orbit interpolation")
        t_arr = np.linspace(t0_az, t0_az + dt_az * (naz - 1), naz)
        pos = interp_pos(t_arr)
        vel = interp_vel(t_arr)

        log.info("Range-Doppler terrain correction (LUT computation)")
        az_geo, dist_geo = range_doppler(
            # Removing first pos to get more precision. Is this useful?
            dem_x.ravel() - pos[0, 0],
            dem_y.ravel() - pos[0, 1],
            dem_z.ravel() - pos[0, 2],
            pos - pos[0],
            vel,
            tol=1e-8,
            maxiter=10000
            # dem_x.ravel(), dem_y.ravel(), dem_z.ravel(), pos, vel, tol=1e-8
        )

        # convert range - azimuth to pixel indices
        c0 = 299792458.0
        r0 = float(slant_range_time) * c0 / 2
        dr = c0 / (2 * float(range_sampling_rate))

        rg_geo = (dist_geo - r0) / dr

        cnd1 = (rg_geo >= 0) & (rg_geo < nrg)
        cnd2 = (az_geo >= 0) & (az_geo < naz)
        valid = cnd1 & cnd2
        rg_geo[~valid] = np.nan
        az_geo[~valid] = np.nan

        return az_geo.reshape(alt.shape), rg_geo.reshape(alt.shape), dem_prof

    def deramp_burst(self, burst_idx=1):
        """Computes the azimuth deramping phase using product metadata.

        Args:
            burst_idx (int, optional): Burst index. Defaults to 1.

        Returns:
            array: phase correction to apply to the SLC burst.
        """

        if burst_idx < 1 or burst_idx > self.burst_count:
            raise ValueError(
                f"Invalid burst index (must be between 1 and {self.burst_count})"
            )

        meta = self.meta
        meta_image = meta["product"]["imageAnnotation"]
        meta_general = meta["product"]["generalAnnotation"]
        meta_burst = meta["product"]["swathTiming"]["burstList"]["burst"][burst_idx - 1]

        log.info("Computing TOPS deramping phase")

        c0 = 299792458.0
        # lines_per_burst = int(meta["product"]["swathTiming"]["linesPerBurst"])
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

        # keeping a few points before and after burst
        image_info = meta["product"]["imageAnnotation"]["imageInformation"]
        azimuth_time_interval = float(image_info["azimuthTimeInterval"])
        dt_az = float(azimuth_time_interval)
        # naz = int(lines_per_burst)
        naz = self.lines_per_burst
        tt0 = self.state_vectors["t0"]
        t0_az = (isoparse(az_time) - tt0).total_seconds()
        t_end_burst = t0_az + dt_az * naz
        t_sv_burst = self.state_vectors["t"]
        cnd = (t_sv_burst > t0_az - 360) & (t_sv_burst < t_end_burst + 360)

        state_vectors = {k: v[cnd] for k, v in self.state_vectors.items() if k != "t0"}

        # _, orb_v = sv_interpolator_poly(state_vectors)
        _, orb_v = sv_interpolator(state_vectors)
        t_mid = t0_az + az_dt * self.lines_per_burst / 2.0
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
            -az_dt * self.lines_per_burst / 2.0,
            az_dt * self.lines_per_burst / 2.0,
            self.lines_per_burst,
        )
        eta_c = -fdc / ka
        rg_mid = slant_range_time + 0.5 * nrg * rg_dt
        eta_mid = fdc_fun(rg_mid) / ka_fun(rg_mid)
        eta_ref = eta_c - eta_mid

        phi_deramp = -np.pi * kt[None] * (eta[:, None] - eta_ref[None]) ** 2
        return phi_deramp  # .astype("float32")

    def read_burst(self, burst_idx=1, remove_invalid=True):
        """Reads raster SLC burst.

        Args:
            burst_idx (int, optional): burst index. Defaults to 1.
            remove_invalid (bool, optional): Sets non-valid pixels to NaN. Defaults to True.

        Returns:
            array: Complex raster
        """

        if burst_idx < 1 or burst_idx > self.burst_count:
            raise ValueError(
                f"Invalid burst index (must be between 1 and {self.burst_count})"
            )

        meta = self.meta
        burst_info = meta["product"]["swathTiming"]
        burst_data = burst_info["burstList"]["burst"][burst_idx - 1]
        # lines_per_burst = int(burst_info["linesPerBurst"])

        first_line = (burst_idx - 1) * self.lines_per_burst

        nodataval = np.nan + 1j * np.nan
        arr = (
            read_chunk(self.pth_tiff, first_line, self.lines_per_burst).astype(
                np.complex64
            )
            / self.beta_nought
        )

        # not sure about that, should we consider these holes as NaN?
        # leaving as is for now, to avoid propagating NaN in filtering / resampling
        # arr[arr == 0 + 1j * 0] = nodataval

        if remove_invalid:
            fs_str = burst_data["firstValidSample"]["#text"]
            ls_str = burst_data["lastValidSample"]["#text"]
            first_sample_arr = np.array((fs_str).split(" "), dtype="int")
            last_sample_arr = np.array((ls_str).split(" "), dtype="int")
            for i in range(self.lines_per_burst):
                if first_sample_arr[i] > -1:
                    arr[i, : first_sample_arr[i]] = nodataval
                    arr[i, last_sample_arr[i] + 1 :] = nodataval
                else:
                    arr[i] = nodataval
        return arr

    def phi_topo(self, rg):
        """Computes the topographic phase using slant range indices.

        Args:
            rg (array): slant range pixel indices that will be converted to distances using annotation data.

        Returns:
            array: topographic phase for the given burst.

        Note:
            For the primary burst, range is simply the pixel slant range index. For a secondary burst, it is the range index of the burst reprojected in the primary grid thanks to the coregistration function.

        """
        meta = self.meta
        image_info = meta["product"]["imageAnnotation"]["imageInformation"]
        slant_range_time = image_info["slantRangeTime"]
        product_info = meta["product"]["generalAnnotation"]["productInformation"]
        range_sampling_rate = product_info["rangeSamplingRate"]

        log.info("Computing topographic phase")

        freq = float(product_info["radarFrequency"])
        c0 = 299792458.0
        lam = c0 / freq
        r0 = float(slant_range_time) * c0 / 2
        dr = c0 / (2 * float(range_sampling_rate))
        dist = rg * dr + r0

        return (4 * np.pi / lam) * dist

    def compute_burst_overlap(self, burst_idx=2):
        """Computes the overlap between a burst and the previous one.
        Used for ESD.

        Args:
            burst_idx (int, optional): Burst index, must be >=2. Defaults to 2.

        Raises:
            ValueError: Burst index is out of bounds.

        Returns:
            int: number of overlapping lines.
        """
        if burst_idx < 2 or burst_idx > self.burst_count:
            raise ValueError(
                f"Invalid burst index (must be between 2 and {self.burst_count})"
            )
        meta = self.meta
        image_info = meta["product"]["imageAnnotation"]["imageInformation"]
        azimuth_time_interval = float(image_info["azimuthTimeInterval"])
        burst_info = meta["product"]["swathTiming"]
        burst_1 = burst_info["burstList"]["burst"][burst_idx - 1]
        az_time_1 = isoparse(burst_1["azimuthTime"])
        burst_2 = burst_info["burstList"]["burst"][burst_idx]
        az_time_2 = isoparse(burst_2["azimuthTime"])

        diff_az_time = (
            az_time_1 - az_time_2
        ).total_seconds() + self.lines_per_burst * azimuth_time_interval
        return diff_az_time / azimuth_time_interval


def coregister(arr_p, az_p2g, rg_p2g, az_s2g, rg_s2g):
    """Fast parallel coregistration based on lookup-tables in a DEM geometry.

    Args:
        arr_p (array): array containing the data of the primary burst to coregister
        az_p2g (array): primary azimuth coordinates
        rg_p2g (array): primary range coordinates
        az_s2g (array): secondary azimuth coordinates
        rg_s2g (array): secondary range coordinates

    Returns:
        (array, array): az_co and rg_co are azimuth range of the secondary expressed in the primary geometry
    """
    log.info("Projecting secondary coordinates to primary grid.")
    return coreg_fast(arr_p, az_p2g, rg_p2g, az_s2g, rg_s2g)

@timeit
@njit(nogil=True, parallel=True, cache=True)
def coreg_fast(arr_p, azp, rgp, azs, rgs):

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

    naz, nrg = arr_p.shape

    az_s2p = np.full((naz, nrg), np.nan)
    rg_s2p = np.full((naz, nrg), np.nan)
    p = np.zeros(2)
    nl, nc = azp.shape
    # - loop on DEM
    for i in prange(0, nl - 1):
        for j in range(0, nc - 1):
            # - for each 4 neighborhood
            aa = azp[i : i + 2, j : j + 2].flatten()  # .ravel()
            rr = rgp[i : i + 2, j : j + 2].flatten()  # .ravel()
            aas = azs[i : i + 2, j : j + 2].flatten()  # .ravel()
            rrs = rgs[i : i + 2, j : j + 2].flatten()  # .ravel()
            # - collect triangle vertices
            xx = np.vstack((aa, rr)).T
            yy = np.vstack((aas, rrs)).T
            if np.isnan(xx).any() or np.isnan(yy).any():
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
                        az_s2p[a, r] = interp(aas[0], aas[1], aas[2], l1, l2, l3)
                        rg_s2p[a, r] = interp(rrs[0], rrs[1], rrs[2], l1, l2, l3)
                    l1, l2, l3 = bary(p, xx[3], xx[1], xx[2])
                    if is_in_tri(l1, l2):
                        az_s2p[a, r] = interp(aas[3], aas[1], aas[2], l1, l2, l3)
                        rg_s2p[a, r] = interp(rrs[3], rrs[1], rrs[2], l1, l2, l3)

    return az_s2p, rg_s2p

@timeit
def align(arr_s, az_s2p, rg_s2p, kernel="bicubic"):
    """Aligns the secondary image to the geometry of the primary

    Args:
        arr_s (array): image in the secondary geometry
        az_s2p (array): azimuth lookup table to project secondary to primary
        rg_s2p (array): range lookup table project secondary to primary
        kernel (str, optional): Type of kernel (values are "nearest", "bilinear", "bicubic" -- 4 point bicubic, "bicubic6" -- six point bicubic). Defaults to "bicubic".

    Returns:
        array: projected image
    """
    log.info("Warp secondary to primary geometry.")
    return remap(arr_s, az_s2p, rg_s2p, kernel)


def resample(
    arr, file_dem, file_out, az_p2g, rg_p2g, kernel="bicubic", write_phase=False
):
    """Reproject array to using a lookup table.

    Args:
        arr (array): image in the SAR geometry
        file_dem (str): file of the original DEM used to compute the lookup table
        file_out (str): output file
        az_p2g (array): azimuth coordinates of the lookup table
        rg_p2g (array): range coordinates of the lookup table
        kernel (str): kernel used to align secondary SLC. Possible values are "nearest", "bilinear", "bicubic" and "bicubic6".Defaults to "bilinear".
        write_phase (bool): writes the array's phase . Defaults to False.
    """
    # retrieve dem profile

    dst_height, dst_width = az_p2g.shape

    with rasterio.open(file_dem) as ds_dem:
        out_prof = ds_dem.profile.copy()

        # account for DEM resampling
        dst_trans = ds_dem.transform * ds_dem.transform.scale(
            (ds_dem.width / dst_width), (ds_dem.height / dst_height)
        )

    out_prof.update({"width": dst_width, "height": dst_height, "transform": dst_trans})

    log.info("Warp to match DEM geometry")
    wped = remap(arr, az_p2g, rg_p2g, kernel=kernel)

    # TODO: enforce COG
    log.info("Write output GeoTIFF")
    if write_phase:
        phi = np.angle(wped)
        nodata = -9999
        phi[np.isnan(wped)] = nodata
        out_prof.update({"dtype": phi.dtype, "count": 1, "nodata": nodata})
        with rasterio.open(file_out, "w", **out_prof) as dst:
            dst.write(phi, 1)
    else:
        if np.iscomplexobj(arr):
            out_prof.update({"dtype": arr.real.dtype, "count": 2, "nodata": np.nan})
            with rasterio.open(file_out, "w", **out_prof) as dst:
                # real outputs to avoid complex cast warnings in rasterio
                dst.write(wped.real, 1)
                dst.write(wped.imag, 2)
        else:
            out_prof.update({"dtype": arr.dtype, "count": 1, "nodata": 0})
            with rasterio.open(file_out, "w", **out_prof) as dst:
                dst.write(wped, 1)


def fast_esd(ifgs, overlap):
    """Applies an in-place phase correction to burst (complex) interferograms to mitigate phase jumps between the bursts.
    Args:
        ifgs (list): List of complex SLC interferograms
        overlap (int): Number of overlapping azimuth pixels between two bursts (can be computed with `compute_burst_overlap`)

    Note:
        Based on ideas introduced in:
        Qin, Y.; Perissin, D.; Bai, J. A Common “Stripmap-Like” Interferometric Processing Chain for TOPS and ScanSAR Wide Swath Mode. Remote Sens. 2018, 10, 1504.
    """

    import matplotlib.pyplot as plt

    if len(ifgs) < 2:
        log.warning(
            "Skipping ESD: there must be at least 2 consecutive bursts from the same subsawths."
        )
    else:

        # nodataval = np.nan + 1j * np.nan
        phase_diffs = []
        for i in range(len(ifgs) - 1):
            log.info(f"Compute cross interferogram {i+1} / {len(ifgs) - 1}")
            cross = ifgs[i][-overlap:] * ifgs[i + 1][:overlap].conj()
            phi_clx = cross[~np.isnan(cross)]
            phase_diffs.append(np.angle(phi_clx.mean()))

        naz, nrg = ifgs[0].shape
        x = np.arange(naz)
        xdown, xup = overlap / 2, naz - 1 - overlap / 2

        def make_ramp(idx):
            if idx == 0:
                ydown, yup = -phase_diffs[idx] / 2, phase_diffs[idx] / 2
            elif idx == len(ifgs) - 1:
                ydown, yup = -phase_diffs[idx - 1] / 2, phase_diffs[idx - 1] / 2
            else:
                ydown, yup = -phase_diffs[idx - 1] / 2, phase_diffs[idx] / 2
            slope = (yup - ydown) / (xup - xdown)
            off = ydown - slope * xdown
            ramp = slope * x + off
            return np.exp(-1j * (ramp[:, None] + np.zeros((nrg))))

        # TODO: improve by downweighting points far from mid overlap ?
        naz, nrg = ifgs[0].shape
        for i, ifg in enumerate(ifgs):
            log.info(f"Apply ESD to interferogram {i+1} / {len(ifgs)}")
            esd_ramp = make_ramp(i).astype(np.complex64)
            ifg *= esd_ramp


def stitch_bursts(bursts, overlap):
    """Stitch bursts in the single look radar geometry.

    Args:
        bursts (list): list of bursts
        overlap (int): number of overlapping pixels

    Raises:
        ValueError: list is empty
    """
    # if not isinstance(overlap, int):
    # log.warning("overlap must be an integer, rounding to the lowest integer.")
    H = int(overlap / 2)
    naz, nrg = bursts[0].shape
    nburst = len(bursts)
    if nburst >= 2:
        siz = (naz - H) * 2 + (nburst - 2) * (naz - 2 * H)
    elif nburst == 1:
        siz = naz - H
    else:
        raise ValueError("Empty burst list")

    log.info("Stitching bursts to make a continuous image")
    arr = np.zeros((siz, nrg), dtype=bursts[0].dtype)
    off = 0
    for i in range(nburst):
        if i == 0:
            arr[: naz - H] = bursts[i][: naz - H]
            off += naz - H
        elif i == nburst - 1:
            arr[-naz + H :] = bursts[i][-naz + H :]
        else:
            arr[off : off + (naz - 2 * H)] = bursts[i][H:-H]
            off += naz - 2 * H
    return arr


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
        arr = src.read(
            1, window=Window(0, first_line, src.width, number_of_lines)
        ).astype("complex64")
    return arr


def sv_interpolator(state_vectors):

    t = state_vectors["t"]
    x = state_vectors["x"]
    y = state_vectors["y"]
    z = state_vectors["z"]
    vx = state_vectors["vx"]
    vy = state_vectors["vy"]
    vz = state_vectors["vz"]

    interp_pos = CubicHermiteSpline(t, np.array([x, y, z]).T, np.array([vx, vy, vz]).T)
    interp_vel = interp_pos.derivative(1)

    return interp_pos, interp_vel


# TODO: order as a parmeter
def sv_interpolator_poly(state_vectors):
    t = state_vectors["t"]
    x = state_vectors["x"]
    y = state_vectors["y"]
    z = state_vectors["z"]
    vx = state_vectors["vx"]
    vy = state_vectors["vy"]
    vz = state_vectors["vz"]

    def interp_pos(t_arr):
        px = Polynomial.fit(t, x, 5)
        py = Polynomial.fit(t, y, 5)
        pz = Polynomial.fit(t, z, 5)
        return np.vstack((px(t_arr), py(t_arr), pz(t_arr))).T

    def interp_vel(t_arr):
        pvx = Polynomial.fit(t, vx, 5)
        pvy = Polynomial.fit(t, vy, 5)
        pvz = Polynomial.fit(t, vz, 5)
        return np.vstack((pvx(t_arr), pvy(t_arr), pvz(t_arr))).T

    return interp_pos, interp_vel


# TODO: allow cop-dem-glo30 and return composite (horiz.+vertical CRS)
# TODO: add upsampling option
def auto_dem(file_dem, gcps, buffer_arc_sec=40, force_download=False, upscale_factor=1):
    minmax = lambda x: (x.min(), x.max())
    xmin, xmax = minmax(np.array([p.x for p in gcps]))
    ymin, ymax = minmax(np.array([p.y for p in gcps]))

    # DEM dependent
    step = 1 / 3600

    xmin = floor(xmin / step) * step
    xmax = ceil(xmax / step) * step
    ymin = floor(ymin / step) * step
    ymax = ceil(ymax / step) * step

    off = buffer_arc_sec / 3600
    shp = box(xmin - off, ymin - off, xmax + off, ymax + off)

    if not os.path.exists(file_dem) or force_download:
        retrieve_dem(shp, file_dem, dem_name="nasadem", upscale_factor=upscale_factor)
    else:
        log.info("--DEM already on disk")


# TODO add resampling type option
@timeit
def load_dem_coords(file_dem, upscale_factor=1):

    with rasterio.open(file_dem) as ds:
        if upscale_factor != 1:
            # on-read resampling
            alt = ds.read(
                out_shape=(
                    ds.count,
                    int(ds.height * upscale_factor),
                    int(ds.width * upscale_factor),
                ),
                resampling=Resampling.bilinear,
                # resampling=Resampling.cubic,
            )[0]
            # scale image transform
            dem_prof = ds.profile.copy()
            dem_trans = ds.transform * ds.transform.scale(
                (ds.width / alt.shape[-1]), (ds.height / alt.shape[-2])
            )
            nodata = ds.nodata
        else:
            alt = ds.read(1)
            dem_prof = ds.profile.copy()
            dem_trans = ds.transform
            nodata = ds.nodata

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

    # make sure nodata is nan in output
    if not np.isnan(nodata):
        msk = alt == nodata
    alt = alt.astype("float64")
    if not np.isnan(nodata):
        alt[msk] = np.nan

    dem_prof.update({"width": width, "height": height, "transform": dem_trans})
    return lat, lon, alt, dem_prof


# TODO produce right composite crs for each DEM
@timeit
def lla_to_ecef(lat, lon, alt, dem_crs):

    # TODO: use parameter instead
    WGS84_crs = "EPSG:4326+5773"
    ECEF_crs = "EPSG:4978"

    # much faster than rasterio transform
    tf = Transformer.from_crs(WGS84_crs, ECEF_crs)

    # single-threaded
    # !! pyproj uses lon, lat whereas rasterio uses lat, lon
    # WGS84_points = (lon, lat, alt)
    # dem_pts = tf.transform(*WGS84_points)

    # multi-threaded
    chunk = 128
    wgs_pts = [
        (lon[b : b + chunk], lat[b : b + chunk], alt[b : b + chunk])
        for b in range(0, len(lon), chunk)
    ]
    chunked = Parallel(n_jobs=-1, prefer="threads")(
        delayed(tf.transform)(*b) for b in wgs_pts
    )
    dem_x = np.zeros_like(lon)
    dem_y = np.zeros_like(lon)
    dem_z = np.zeros_like(lon)
    for i in range(len(chunked)):
        b1 = i * chunk
        b2 = np.minimum((i + 1) * chunk, lon.shape[0])
        dem_x[b1:b2] = chunked[i][0]
        dem_y[b1:b2] = chunked[i][1]
        dem_z[b1:b2] = chunked[i][2]

    log.info(f"demx shape: {dem_x.shape}")
    return dem_x, dem_y, dem_z


@njit(nogil=True, cache=True, parallel=True)
def range_doppler(xx, yy, zz, positions, velocities, tol=1e-8, maxiter=10000):
    def doppler_freq(t, x, y, z, positions, velocities, t0, t1):
        factors = t - np.floor(t)

        px = positions[t0, 0] + factors * (positions[t1, 0] - positions[t0, 0])
        py = positions[t0, 1] + factors * (positions[t1, 1] - positions[t0, 1])
        pz = positions[t0, 2] + factors * (positions[t1, 2] - positions[t0, 2])
        vx = velocities[t0, 0] + factors * (velocities[t1, 0] - velocities[t0, 0])
        vy = velocities[t0, 1] + factors * (velocities[t1, 1] - velocities[t0, 1])
        vz = velocities[t0, 2] + factors * (velocities[t1, 2] - velocities[t0, 2])

        dx = x - px
        dy = y - py
        dz = z - pz
        d2 = dx**2 + dy**2 + dz**2
        fc = -(vx * dx + vy * dy + vz * dz) / np.sqrt(d2)

        return fc, dx, dy, dz

    i_zd = np.zeros_like(xx)
    r_zd = np.zeros_like(xx)
    num_orbits = len(positions)

    for i in prange(xx.shape[0]):
        x_val = xx[i]
        y_val = yy[i]
        z_val = zz[i]
        if np.isnan(x_val):
            continue
        a = 0
        b = num_orbits - 1
        fa, _, _, _ = doppler_freq(
            a, x_val, y_val, z_val, positions, velocities, int(a), int(np.ceil(a))
        )
        fb, _, _, _ = doppler_freq(
            b, x_val, y_val, z_val, positions, velocities, int(b), int(np.ceil(b))
        )

        # exit if no solution
        if np.sign(fa * fb) > 0:
            i_zd[i] = np.nan
            r_zd[i] = np.nan
            continue

        if np.abs(fa) < tol:
            i_zd[i] = a
            r_zd[i] = 0
            continue
        elif np.abs(fb) < tol:
            i_zd[i] = b
            r_zd[i] = 0
            continue

        c = (a + b) / 2.0
        fc, _, _, _ = doppler_freq(
            c, x_val, y_val, z_val, positions, velocities, int(c), int(np.ceil(c))
        )

        its = 0
        while np.abs(fc) > tol and its < maxiter:
            its += 1
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            c = (a + b) / 2.0
            fc, _, _, _ = doppler_freq(
                c, x_val, y_val, z_val, positions, velocities, int(c), int(np.ceil(c))
            )

        i_zd[i] = c
        dx, dy, dz = doppler_freq(
            c, x_val, y_val, z_val, positions, velocities, int(c), int(np.ceil(c))
        )[1:]
        r_zd[i] = np.sqrt(dx**2 + dy**2 + dz**2)

    return i_zd, r_zd
