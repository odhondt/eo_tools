from eo_tools.S1.core import S1IWSwath, coregister, align
from eo_tools.S1.util import presum, boxcar, remap
from eo_tools.auxils import get_burst_geometry
from eo_tools.auxils import remove
import numpy as np
import xarray as xr
import rasterio as rio
from rasterio.windows import Window
import rioxarray as riox
from rioxarray.merge import merge_arrays
import warnings
import os
import concurrent
import dask.array as da
from rasterio.errors import NotGeoreferencedWarning
import logging
from pyroSAR import identify
from typing import Union, List, Callable
from datetime import datetime
from pathlib import Path
from shapely.geometry import shape
from osgeo import gdal
from rasterio.features import geometry_window
from affine import Affine
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import uniform_filter as uflt
from eo_tools.auxils import block_process
from eo_tools.util import _has_overlap

# use child processes
# USE_CP = False
USE_CP = True

log = logging.getLogger(__name__)


def process_insar(
    dir_prm: str,
    dir_sec: str,
    outputs_prefix: str,
    aoi_name: str = None,
    shp: shape = None,
    pol: Union[str, List[str]] = "full",
    subswaths: List[str] = ["IW1", "IW2", "IW3"],
    write_coherence: bool = True,
    write_interferogram: bool = True,
    write_primary_amplitude: bool = True,
    write_secondary_amplitude: bool = False,
    apply_fast_esd: bool = False,
    dir_dem: str = "/tmp",
    dem_upsampling: float = 1.8,
    dem_force_download: bool = False,
    dem_buffer_arc_sec: float = 40,
    boxcar_coherence: Union[int, List[int]] = [3, 10],
    filter_ifg: bool = True,
    multilook: List[int] = [1, 4],
    warp_kernel: str = "bicubic",
    cal_type: str = "beta",
    clip_to_shape: bool = True,
    skip_preprocessing: bool = False,
) -> str:
    """Performs InSAR processing of a pair of SLC Sentinel-1 products, geocode the outputs and writes them as COG (Cloud Optimized GeoTiFF) files.
    AOI crop is optional.

    Args:
        dir_prm (str): primary image (SLC Sentinel-1 product directory).
        dir_sec (str): secondary image (SLC Sentinel-1 productdirectory).
        outputs_prefix (str): location in which the product subdirectory will be created
        aoi_name (str, optional): optional suffix to describe AOI / experiment. Defaults to None.
        shp (shapely.geometry.shape, optional): Shapely geometry describing an area of interest as a polygon. Defaults to None.
        pol (Union[str, List[str]], optional): Polarimetric channels to process (Either 'VH','VV, 'full' or a list like ['HV', 'VV']).  Defaults to "full".
        subswaths (List[str], optional): limit the processing to a list of subswaths like `["IW1", "IW2"]`. Defaults to ["IW1", "IW2", "IW3"].
        write_coherence (bool, optional): Write the magnitude of the complex coherence. Defaults to True.
        write_interferogram (bool, optional): Write the interferogram phase. Defaults to True.
        write_primary_amplitude (bool, optional): Write the amplitude of the primary image. Defaults to True.
        write_secondary_amplitude (bool, optional): Write the amplitude of the secondary image. Defaults to False.
        apply_fast_esd (bool, optional): correct the phase to avoid jumps between bursts. This has no effect if only one burst is processed. Defaults to False.
        dem_upsampling (float, optional): upsampling factor for the DEM, it is recommended to keep the default value. Defaults to 1.8.
        dem_force_download (bool, optional):  To reduce execution time, DEM files are stored on disk. Set to True to redownload these files if necessary. Defaults to False.
        dem_buffer_arc_sec (float, optional): Increase if the image area is not completely inside the DEM. Defaults to 40.
        boxcar_coherence (Union[int, List[int]], optional): Size of the boxcar filter to apply for coherence estimation. Defaults to [3, 10].
        filter_ifg (bool): Also applies boxcar to interferogram. Has no effect if file_complex_ifg is set to None or write_coherence is set to False. Defaults to True.x
        multilook (List[int], optional): Multilooking to apply prior to geocoding. Defaults to [1, 4].
        warp_kernel (str, optional): Resampling kernel used in coregistration and geocoding. Possible values are "nearest", "bilinear", "bicubic" and "bicubic6". Defaults to "bicubic".
        cal_type (str, optional): Type of radiometric calibration. "beta" or "sigma" nought. Defaults to "beta"
        clip_to_shape (bool, optional): If set to False the geocoded images are not clipped according to the `shp` parameter. They are made of all the bursts intersecting the `shp` geometry. Defaults to True.
        skip_preprocessing (bool, optional): Skip the processing part in case the files are already written. Defaults to False.

    Returns:
        str: output directory
    """

    if not np.any([coherence, interferogram]):
        raise ValueError(
            "At least one of `write_coherence` and `write_interferogram` must be True."
        )

    # prepare pair for interferogram computation
    out_dir = prepare_insar(
        dir_prm=dir_prm,
        dir_sec=dir_sec,
        outputs_prefix=outputs_prefix,
        aoi_name=aoi_name,
        shp=shp,
        pol=pol,
        apply_fast_esd=apply_fast_esd,
        subswaths=subswaths,
        warp_kernel=warp_kernel,
        cal_type=cal_type,
        dir_dem=dir_dem,
        dem_upsampling=dem_upsampling,
        dem_force_download=dem_force_download,
        dem_buffer_arc_sec=dem_buffer_arc_sec,
        skip_preprocessing=skip_preprocessing,
    )

    var_names = []
    if write_coherence:
        var_names.append("coh")
    if write_interferogram:
        var_names.append("ifg")
    if write_primary_amplitude:
        var_names.append("amp_prm")
    if write_secondary_amplitude:
        var_names.append("amp_sec")

    if isinstance(pol, str):
        if pol == "full":
            pol_ = ["vv", "vh"]
        else:
            pol_ = [pol.lower()]
    elif isinstance(pol, list):
        pol_ = [p.lower() for p in pol]

    iw_idx = [iw[2] for iw in subswaths]
    patterns = [f"{p}_iw{iw}" for p in pol_ for iw in iw_idx]
    for pattern in patterns:
        file_prm = f"{out_dir}/slc_prm_{pattern}.tif"
        file_sec = f"{out_dir}/slc_sec_{pattern}.tif"

        if os.path.isfile(file_prm) and os.path.isfile(file_sec):
            log.info(
                f"---- Interferometric outputs for {" ".join(pattern.split('/')[-1].split('_')).upper()}"
            )
            if write_coherence and write_interferogram:
                file_coh = f"{out_dir}/coh_{pattern}.tif"
                file_ifg = f"{out_dir}/ifg_{pattern}.tif"
                coherence(
                    file_prm=file_prm,
                    file_sec=file_sec,
                    file_out=file_coh,
                    box_size=boxcar_coherence,
                    multilook=multilook,
                    magnitude=True,
                    file_complex_ifg=file_ifg,
                    filter_ifg=filter_ifg,
                )
            elif write_coherence and not write_interferogram:
                file_coh = f"{out_dir}/coh_{pattern}.tif"
                coherence(
                    file_prm=file_prm,
                    file_sec=file_sec,
                    file_out=file_coh,
                    box_size=boxcar_coherence,
                    multilook=multilook,
                    magnitude=True,
                )
            elif not write_coherence and write_interferogram:
                file_ifg = f"{out_dir}/ifg_{pattern}.tif"
                interferogram(
                    file_prm=file_prm,
                    file_sec=file_sec,
                    file_out=file_ifg,
                    multilook=multilook,
                )

            if write_primary_amplitude:
                file_ampl = f"{out_dir}/amp_prm_{pattern}.tif"
                amplitude(file_in=file_prm, file_out=file_ampl, multilook=multilook)

            if write_secondary_amplitude:
                file_ampl = f"{out_dir}/amp_sec_{pattern}.tif"
                amplitude(file_in=file_sec, file_out=file_ampl, multilook=multilook)

    # by default, we use iw and pol which exist
    _child_process(
        geocode_and_merge_iw,
        dict(
            input_dir=Path(out_dir).parent,
            var_names=var_names,
            shp=shp,
            pol=["vv", "vh"],
            subswaths=["IW1", "IW2", "IW3"],
            warp_kernel=warp_kernel,
            clip_to_shape=clip_to_shape,
        ),
    )
    return Path(out_dir).parent


def prepare_insar(
    dir_prm: str,
    dir_sec: str,
    outputs_prefix: str,
    aoi_name: str = None,
    shp: shape = None,
    pol: Union[str, List[str]] = "full",
    subswaths: List[str] = ["IW1", "IW2", "IW3"],
    apply_fast_esd: bool = False,
    warp_kernel: str = "bicubic",
    cal_type: str = "beta",
    dir_dem: str = "/tmp",
    dem_upsampling: float = 1.8,
    dem_force_download: bool = False,
    dem_buffer_arc_sec: float = 40,
    skip_preprocessing: bool = False,
) -> str:
    """Produce a coregistered pair of Single Look Complex images and associated lookup tables.

    Args:
        dir_prm (str): Primary image (SLC Sentinel-1 product directory).
        dir_sec (str): Secondary image (SLC Sentinel-1 productdirectory).
        outputs_prefix (str): location in which the product subdirectory will be created.
        aoi_name (str, optional): optional suffix to describe AOI / experiment. Defaults to None.
        shp (shapely.geometry.shape, optional): Shapely geometry describing an area of interest as a polygon. Defaults to None.
        pol (Union[str, List[str]], optional):  Polarimetric channels to process (Either 'VH','VV, 'full' or a list like ['HV', 'VV']).  Defaults to "full".
        subswaths (List[str], optional):  limit the processing to a list of subswaths like `["IW1", "IW2"]`. Defaults to ["IW1", "IW2", "IW3"].
        apply_fast_esd (bool, optional): correct the phase to avoid jumps between bursts. This has no effect if only one burst is processed.  Defaults to False.
        warp_kernel (str, optional): kernel used to align secondary SLC. Possible values are "nearest", "bilinear", "bicubic" and "bicubic6".Defaults to "bilinear".
        cal_type (str, optional): Type of radiometric calibration. "beta" or "sigma" nought. Defaults to "beta"
        dem_upsampling (float, optional): upsampling factor for the DEM, it is recommended to keep the default value. Defaults to 1.8.
        dem_force_download (bool, optional):   To reduce execution time, DEM files are stored on disk. Set to True to redownload these files if necessary. Defaults to True.
        dem_buffer_arc_sec (float, optional): Increase if the image area is not completely inside the DEM. Defaults to 40.
        skip_preprocessing (bool, optional): Skip the processing part in case the files are already written. It is recommended to leave this parameter to default value. Defaults to False.

    Returns:
        str: output directory
    """

    if aoi_name is None:
        aoi_substr = ""
    else:
        aoi_substr = f"_{aoi_name}"

    if not isinstance(subswaths, list):
        raise ValueError("Subswaths must be a list like ['IW1', 'IW2'].")

    # retrieve burst geometries
    gdf_burst_prm = get_burst_geometry(
        dir_prm, target_subswaths=["IW1", "IW2", "IW3"], polarization="VV"
    )
    gdf_burst_sec = get_burst_geometry(
        dir_sec, target_subswaths=["IW1", "IW2", "IW3"], polarization="VV"
    )

    # find what subswaths and bursts intersect AOI
    if shp is not None:
        gdf_burst_prm = gdf_burst_prm[gdf_burst_prm.intersects(shp)]
        gdf_burst_sec = gdf_burst_sec[gdf_burst_sec.intersects(shp)]

    if gdf_burst_prm.empty:
        raise RuntimeError(
            "The list of bursts to process is empty. Make sure shp intersects the product."
        )

    # identify corresponding subswaths
    sel_subsw_prm = gdf_burst_prm["subswath"]
    sel_subsw_sec = gdf_burst_sec["subswath"]
    unique_subswaths = np.unique(np.concatenate((sel_subsw_prm, sel_subsw_sec)))
    unique_subswaths = [it for it in unique_subswaths if it in subswaths]

    # check that polarization is correct
    info_prm = identify(dir_prm)
    if isinstance(pol, str):
        if pol == "full":
            pol_ = info_prm.polarizations
        else:
            if pol.upper() in info_prm.polarizations:
                pol_ = [pol]
            else:
                raise RuntimeError(
                    f"polarization {pol} does not exists in the source product"
                )
    elif isinstance(pol, list):
        pol_ = [x for x in pol if x in info_prm.polarizations]
    else:
        raise RuntimeError("polarizations must be of type str or list")

    # do a check on orbits
    info_sec = identify(dir_sec)
    meta_prm = info_prm.scanMetadata()
    meta_sec = info_sec.scanMetadata()
    orbnum = meta_prm["orbitNumber_rel"]
    if meta_sec["orbitNumber_rel"] != orbnum:
        raise RuntimeError("Images must be from the same relative orbit.")

    # parse dates
    datestr_prm = meta_prm["start"]
    datestr_sec = meta_sec["start"]
    date_prm = datetime.strptime(datestr_prm, "%Y%m%dT%H%M%S")
    date_sec = datetime.strptime(datestr_sec, "%Y%m%dT%H%M%S")

    id_prm = date_prm.strftime("%Y-%m-%d-%H%M%S")
    id_sec = date_sec.strftime("%Y-%m-%d-%H%M%S")

    out_dir = f"{outputs_prefix}/S1_InSAR_{id_prm}__{id_sec}{aoi_substr}/sar"
    if not os.path.isdir(out_dir):
        log.info(f"Creating directory {out_dir}")
        os.makedirs(out_dir)
    for p in pol_:
        for subswath in unique_subswaths:
            log.info(f"---- Processing subswath {subswath} in {p.upper()} polarization")

            # identify bursts to process
            bursts_prm = gdf_burst_prm[gdf_burst_prm["subswath"] == subswath][
                "burst"
            ].values
            burst_prm_min = bursts_prm.min()
            burst_prm_max = bursts_prm.max()

            iw = int(subswath[2])
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            if not skip_preprocessing:
                preprocess_insar_iw(
                    dir_prm,
                    dir_sec,
                    out_dir,
                    iw=iw,
                    pol=p.lower(),
                    min_burst=burst_prm_min,
                    max_burst=burst_prm_max,
                    apply_fast_esd=apply_fast_esd,
                    warp_kernel=warp_kernel,
                    cal_type=cal_type,
                    dir_dem=dir_dem,
                    dem_upsampling=dem_upsampling,
                    dem_buffer_arc_sec=dem_buffer_arc_sec,
                    dem_force_download=dem_force_download,
                )
                os.rename(
                    f"{out_dir}/primary.tif",
                    f"{out_dir}/slc_prm_{p.lower()}_iw{iw}.tif",
                )
                os.rename(
                    f"{out_dir}/secondary.tif",
                    f"{out_dir}/slc_sec_{p.lower()}_iw{iw}.tif",
                )
                os.rename(f"{out_dir}/lut.tif", f"{out_dir}/lut_{p.lower()}_iw{iw}.tif")
            else:
                log.info("Skipping preprocessing.")
    return out_dir


def preprocess_insar_iw(
    dir_primary: str,
    dir_secondary: str,
    dir_out: str,
    iw: int = 1,
    pol: Union[str, List[str]] = "vv",
    min_burst: int = 1,
    max_burst: int = None,
    apply_fast_esd: bool = True,
    warp_kernel: str = "bicubic",
    cal_type: str = "beta",
    dir_dem: str = "/tmp",
    dem_upsampling: float = 1.8,
    dem_buffer_arc_sec: float = 40,
    dem_force_download: bool = False,
) -> None:
    """Pre-process S1 InSAR subswaths pairs. Write coregistered primary and secondary SLC files as well as a lookup table that can be used to geocode rasters in the single-look radar geometry.

    Args:
        dir_primary (str): directory containing the primary SLC product of the pair.
        dir_secondary (str): directory containing the secondary SLC product of the pair.
        dir_out (str): output directory (creating it if does not exist).
        dir_dem (str, optional): directory where DEMs used for geocoding are stored. Defaults to "/tmp".
        iw (int, optional): subswath index. Defaults to 1.
        pol (str, optional): polarization ('vv','vh'). Defaults to "vv".
        min_burst (int, optional): first burst to process. Defaults to 1.
        max_burst (int, optional): fast burst to process. If not set, last burst of the subswath. Defaults to None.
        dir_dem (str, optional): directory where the DEM is downloaded. Must be created beforehand. Defaults to "/tmp".
        apply_fast_esd: (bool, optional): correct the phase to avoid jumps between bursts. This has no effect if only one burst is processed. Defaults to True.
        warp_kernel (str, optional): kernel used to align secondary SLC. Possible values are "nearest", "bilinear", "bicubic" and "bicubic6".Defaults to "bilinear".
        cal_type (str, optional): Type of radiometric calibration. "beta" or "sigma" nought. Defaults to "beta"
        dem_upsampling (float, optional): Upsample the DEM, it is recommended to keep the default value. Defaults to 2.
        dem_buffer_arc_sec (float, optional): Increase if the image area is not completely inside the DEM. Defaults to 40.
        dem_force_download (bool, optional): To reduce execution time, DEM files are stored on disk. Set to True to redownload these files if necessary. Defaults to false.

    Note:
        DEM-assisted coregistration is performed to align the secondary with the Primary. A lookup table file is written to allow the geocoding images from the radar (single-look) grid to the geographic coordinates of the DEM. Bursts are stitched together to form continuous images. All output files are in the GeoTiff format that can be handled by most GIS softwares and geospatial raster tools such as GDAL and rasterio. Because they are in the SAR geometry, SLC rasters are not georeferenced.
    """

    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)

    if iw not in [1, 2, 3]:
        ValueError("iw must be 1, 2 or 3")

    if pol not in ["vv", "vh"]:
        ValueError("pol must be 'vv' or 'vh'")

    prm = S1IWSwath(dir_primary, iw=iw, pol=pol)
    sec = S1IWSwath(dir_secondary, iw=iw, pol=pol)

    # retrieve burst geometries
    sub_str = f"IW{iw}"
    gdf_burst_prm = get_burst_geometry(
        dir_primary, target_subswaths=[sub_str], polarization="VV"
    )
    gdf_burst_sec = get_burst_geometry(
        dir_secondary, target_subswaths=[sub_str], polarization="VV"
    )

    # here we deal with partial overlap
    offsets = []
    for _, it in gdf_burst_prm.iterrows():
        for _, it2 in gdf_burst_sec.iterrows():
            pair = (it["burst"], it2["burst"])
            if _has_overlap(it["geometry"], it2["geometry"]):
                offsets.append(pair[1] - pair[0])

    if not offsets:
        raise RuntimeError(
            "No overlapping bursts. Cannot further process this product pair."
        )
    if not np.all(np.array(offsets) == offsets[0]):
        raise RuntimeError("Overlapping bursts must be consecutive.")

    # == 0 if full overlap
    burst_offset = offsets[0]

    # intra-product overlap
    overlap = np.round(prm.compute_burst_overlap(2)).astype(int)

    if not max_burst:
        max_burst_ = prm.burst_count
    else:
        max_burst_ = max_burst

    if max_burst_ > min_burst:
        tmp_prm = f"{dir_out}/tmp_primary.tif"
        tmp_sec = f"{dir_out}/tmp_secondary.tif"
    elif max_burst_ < min_burst:
        raise ValueError("max_burst must be >= min_burst")
    else:
        tmp_prm = f"{dir_out}/primary.tif"
        tmp_sec = f"{dir_out}/secondary.tif"

    if (
        max_burst_ > prm.burst_count
        or max_burst_ < 1
        or min_burst > prm.burst_count
        or min_burst < 1
    ):
        raise ValueError(
            f"min_burst and max_burst must be values between 1 and {prm.burst_count}"
        )

    naz = prm.lines_per_burst * (max_burst_ - min_burst + 1)
    nrg = prm.samples_per_burst

    warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
    _child_process(
        _process_bursts_insar,
        (
            prm,
            sec,
            tmp_prm,
            tmp_sec,
            dir_out,
            dir_dem,
            naz,
            nrg,
            min_burst,
            max_burst_,
            burst_offset,
            dem_upsampling,
            dem_buffer_arc_sec,
            dem_force_download,
            warp_kernel,
            overlap,
            cal_type,
        ),
    )

    if (max_burst_ > min_burst) & apply_fast_esd:
        _child_process(
            _apply_fast_esd,
            (
                tmp_prm,
                tmp_sec,
                min_burst,
                max_burst_,
                prm.lines_per_burst,
                nrg,
                overlap,
            ),
        )

    if max_burst_ > min_burst:
        _child_process(
            _stitch_bursts,
            (
                tmp_sec,
                f"{dir_out}/secondary.tif",
                prm.lines_per_burst,
                max_burst_ - min_burst + 1,
                overlap,
            ),
        )

        _child_process(
            _stitch_bursts,
            (
                tmp_prm,
                f"{dir_out}/primary.tif",
                prm.lines_per_burst,
                max_burst_ - min_burst + 1,
                overlap,
            ),
        )

    log.info("Cleaning temporary files")
    if max_burst_ > min_burst:
        if os.path.isfile(tmp_prm):
            os.remove(tmp_prm)
        if os.path.isfile(tmp_sec):
            os.remove(tmp_sec)


def process_slc(
    dir_slc: str,
    outputs_prefix: str,
    aoi_name: str = None,
    shp: shape = None,
    pol: Union[str, List[str]] = "full",
    subswaths: List[str] = ["IW1", "IW2", "IW3"],
    dir_dem: str = "/tmp",
    dem_upsampling: float = 1.8,
    dem_force_download: bool = False,
    dem_buffer_arc_sec: float = 40,
    multilook: List[int] = [1, 4],
    warp_kernel: str = "bicubic",
    cal_type: str = "beta",
    clip_to_shape: bool = True,
    skip_preprocessing: bool = False,
) -> str:
    """Geocode the amplitude of a Sentinel-1 SLC product in the DEM geometry and writes the results as a COG (Cloud Optimized GeoTiFF) file.
    AOI crop is optional.

    Args:
        dir_slc (str): input image (SLC Sentinel-1 product directory).
        outputs_prefix (str): location in which the product subdirectory will be created
        aoi_name (str, optional): optional suffix to describe AOI / experiment. Defaults to None.
        shp (shapely.geometry.shape, optional): Shapely geometry describing an area of interest as a polygon. Defaults to None.
        pol (Union[str, List[str]], optional): Polarimetric channels to process (Either 'VH','VV, 'full' or a list like ['HV', 'VV']).  Defaults to "full".
        subswaths (List[str], optional): limit the processing to a list of subswaths like `["IW1", "IW2"]`. Defaults to ["IW1", "IW2", "IW3"].
        dem_upsampling (float, optional): upsampling factor for the DEM, it is recommended to keep the default value. Defaults to 1.8.
        dem_force_download (bool, optional):  To reduce execution time, DEM files are stored on disk. Set to True to redownload these files if necessary. Defaults to False.
        dem_buffer_arc_sec (float, optional): Increase if the image area is not completely inside the DEM. Defaults to 40.
        multilook (List[int], optional): Multilooking to apply prior to geocoding. Defaults to [1, 4].
        warp_kernel (str, optional): Resampling kernel used in coregistration and geocoding. Possible values are "nearest", "bilinear", "bicubic" and "bicubic6". Defaults to "bicubic".
        cal_type (str, optional): Type of radiometric calibration. "beta" or "sigma" nought. Defaults to "beta"
        clip_to_shape (bool, optional): If set to False the geocoded images are not clipped according to the `shp` parameter. They are made of all the bursts intersecting the `shp` geometry. Defaults to True.
        skip_preprocessing (bool, optional): Skip the processing part in case the files are already written. Defaults to False.

    Returns:
        str: output directory
    """

    if not np.any([coherence, interferogram]):
        raise ValueError(
            "At least one of `write_coherence` and `write_interferogram` must be True."
        )

    # prepare pair for interferogram computation
    out_dir = prepare_slc(
        dir_slc=dir_slc,
        outputs_prefix=outputs_prefix,
        aoi_name=aoi_name,
        shp=shp,
        pol=pol,
        subswaths=subswaths,
        cal_type=cal_type,
        dir_dem=dir_dem,
        dem_upsampling=dem_upsampling,
        dem_force_download=dem_force_download,
        dem_buffer_arc_sec=dem_buffer_arc_sec,
        skip_preprocessing=skip_preprocessing,
    )

    var_names = ["amp"]

    if isinstance(pol, str):
        if pol == "full":
            pol_ = ["vv", "vh"]
        else:
            pol_ = [pol.lower()]
    elif isinstance(pol, list):
        pol_ = [p.lower() for p in pol]

    iw_idx = [iw[2] for iw in subswaths]
    patterns = [f"{p}_iw{iw}" for p in pol_ for iw in iw_idx]
    for pattern in patterns:
        file_slc = f"{out_dir}/slc_{pattern}.tif"

        if os.path.isfile(file_slc):
            log.info(
                f"---- Amplitude for {" ".join(pattern.split('/')[-1].split('_')).upper()}"
            )

            file_ampl = f"{out_dir}/amp_{pattern}.tif"
            amplitude(file_in=file_slc, file_out=file_ampl, multilook=multilook)

    # by default, we use iw and pol which exist
    _child_process(
        geocode_and_merge_iw,
        dict(
            input_dir=Path(out_dir).parent,
            var_names=var_names,
            shp=shp,
            pol=["vv", "vh"],
            subswaths=["IW1", "IW2", "IW3"],
            warp_kernel=warp_kernel,
            clip_to_shape=clip_to_shape,
        ),
    )
    return Path(out_dir).parent


def prepare_slc(
    dir_slc: str,
    outputs_prefix: str,
    aoi_name: str = None,
    shp: shape = None,
    pol: Union[str, List[str]] = "full",
    subswaths: List[str] = ["IW1", "IW2", "IW3"],
    cal_type: str = "beta",
    dir_dem: str = "/tmp",
    dem_upsampling: float = 1.8,
    dem_force_download: bool = False,
    dem_buffer_arc_sec: float = 40,
    skip_preprocessing: bool = False,
) -> str:
    """Pre-process a Sentinel-1 SLC product with the ability to select subswaths polarizations and an area of interest.  Apply radiometric calibration, stitch the selected bursts and compute lookup tables for each subswath of interest, which can be used to project the data in the DEM geometry.

    Args:
        dir_slc (str): Input image (SLC Sentinel-1 product directory).
        outputs_prefix (str): location in which the product subdirectory will be created.
        aoi_name (str, optional): optional suffix to describe AOI / experiment. Defaults to None.
        shp (shapely.geometry.shape, optional): Shapely geometry describing an area of interest as a polygon. Defaults to None.
        pol (Union[str, List[str]], optional):  Polarimetric channels to process (Either 'VH','VV, 'full' or a list like ['HV', 'VV']).  Defaults to "full".
        subswaths (List[str], optional):  limit the processing to a list of subswaths like `["IW1", "IW2"]`. Defaults to ["IW1", "IW2", "IW3"].
        dem_upsampling (float, optional): upsampling factor for the DEM, it is recommended to keep the default value. Defaults to 1.8.
        dem_force_download (bool, optional):   To reduce execution time, DEM files are stored on disk. Set to True to redownload these files if necessary. Defaults to True.
        dem_buffer_arc_sec (float, optional): Increase if the image area is not completely inside the DEM. Defaults to 40.
        skip_preprocessing (bool, optional): Skip the processing part in case the files are already written. It is recommended to leave this parameter to default value. Defaults to False.

    Returns:
        str: output directory
    """

    if aoi_name is None:
        aoi_substr = ""
    else:
        aoi_substr = f"_{aoi_name}"

    if not isinstance(subswaths, list):
        raise ValueError("Subswaths must be a list like ['IW1', 'IW2'].")

    # retrieve burst geometries
    gdf_burst_prm = get_burst_geometry(
        dir_slc, target_subswaths=["IW1", "IW2", "IW3"], polarization="VV"
    )

    # find what subswaths and bursts intersect AOI
    if shp is not None:
        gdf_burst_prm = gdf_burst_prm[gdf_burst_prm.intersects(shp)]

    if gdf_burst_prm.empty:
        raise RuntimeError(
            "The list of bursts to process is empty. Make sure shp intersects the product."
        )

    # identify corresponding subswaths
    sel_subsw_prm = gdf_burst_prm["subswath"]
    unique_subswaths = np.unique(sel_subsw_prm)
    unique_subswaths = [it for it in unique_subswaths if it in subswaths]

    # check that polarization is correct
    info_prm = identify(dir_slc)
    if isinstance(pol, str):
        if pol == "full":
            pol_ = info_prm.polarizations
        else:
            if pol.upper() in info_prm.polarizations:
                pol_ = [pol]
            else:
                raise RuntimeError(
                    f"polarization {pol} does not exists in the source product"
                )
    elif isinstance(pol, list):
        pol_ = [x for x in pol if x in info_prm.polarizations]
    else:
        raise RuntimeError("polarizations must be of type str or list")

    meta_prm = info_prm.scanMetadata()
    # parse dates
    datestr_prm = meta_prm["start"]
    date_prm = datetime.strptime(datestr_prm, "%Y%m%dT%H%M%S")

    id_prm = date_prm.strftime("%Y-%m-%d-%H%M%S")

    out_dir = f"{outputs_prefix}/S1_{id_prm}{aoi_substr}/sar"
    if not os.path.isdir(out_dir):
        log.info(f"Creating directory {out_dir}")
        os.makedirs(out_dir)
    for p in pol_:
        for subswath in unique_subswaths:
            log.info(f"---- Processing subswath {subswath} in {p.upper()} polarization")

            # identify bursts to process
            bursts_prm = gdf_burst_prm[gdf_burst_prm["subswath"] == subswath][
                "burst"
            ].values
            burst_prm_min = bursts_prm.min()
            burst_prm_max = bursts_prm.max()

            iw = int(subswath[2])
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            if not skip_preprocessing:
                preprocess_slc_iw(
                    dir_slc,
                    out_dir,
                    iw=iw,
                    pol=p.lower(),
                    min_burst=burst_prm_min,
                    max_burst=burst_prm_max,
                    cal_type=cal_type,
                    # warp_kernel=warp_kernel,
                    dir_dem=dir_dem,
                    dem_upsampling=dem_upsampling,
                    dem_buffer_arc_sec=dem_buffer_arc_sec,
                    dem_force_download=dem_force_download,
                )
                os.rename(
                    f"{out_dir}/slc.tif",
                    f"{out_dir}/slc_{p.lower()}_iw{iw}.tif",
                )
                os.rename(f"{out_dir}/lut.tif", f"{out_dir}/lut_{p.lower()}_iw{iw}.tif")
            else:
                log.info("Skipping preprocessing.")
    return out_dir


def preprocess_slc_iw(
    dir_slc: str,
    dir_out: str,
    iw: int = 1,
    pol: Union[str, List[str]] = "vv",
    min_burst: int = 1,
    max_burst: int = None,
    cal_type: str = "beta",
    dir_dem: str = "/tmp",
    dem_upsampling: float = 1.8,
    dem_buffer_arc_sec: float = 40,
    dem_force_download: bool = False,
) -> None:
    """Pre-process a Sentinel-1 SLC subswath, with the ability to select a subset of bursts. Apply radiometric calibration, stitch the selected bursts and compute a lookup table, wich can be used to project the data in the DEM geometry.

    Args:
        dir_slc (str): directory containing the SLC input product.
        dir_out (str): output directory (creating it if does not exist).
        dir_dem (str, optional): directory where DEMs used for geocoding are stored. Defaults to "/tmp".
        iw (int, optional): subswath index. Defaults to 1.
        pol (str, optional): polarization ('vv','vh'). Defaults to "vv".
        min_burst (int, optional): first burst to process. Defaults to 1.
        max_burst (int, optional): fast burst to process. If not set, last burst of the subswath. Defaults to None.
        dir_dem (str, optional): directory where the DEM is downloaded. Must be created beforehand. Defaults to "/tmp".
        warp_kernel (str, optional): kernel used to align secondary SLC. Possible values are "nearest", "bilinear", "bicubic" and "bicubic6".Defaults to "bilinear".
        dem_upsampling (float, optional): Upsample the DEM, it is recommended to keep the default value. Defaults to 2.
        dem_buffer_arc_sec (float, optional): Increase if the image area is not completely inside the DEM. Defaults to 40.
        dem_force_download (bool, optional): To reduce execution time, DEM files are stored on disk. Set to True to redownload these files if necessary. Defaults to false.

    Note:
        DEM-assisted coregistration is performed to align the secondary with the Primary. A lookup table file is written to allow the geocoding images from the radar (single-look) grid to the geographic coordinates of the DEM. Bursts are stitched together to form continuous images. All output files are in the GeoTiff format that can be handled by most GIS softwares and geospatial raster tools such as GDAL and rasterio. Because they are in the SAR geometry, SLC rasters are not georeferenced.
    """

    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)

    if iw not in [1, 2, 3]:
        ValueError("iw must be 1, 2 or 3")

    if pol not in ["vv", "vh"]:
        ValueError("pol must be 'vv' or 'vh'")

    slc = S1IWSwath(dir_slc, iw=iw, pol=pol)

    prm_burst_info = slc.meta["product"]["swathTiming"]["burstList"]["burst"]

    overlap = np.round(slc.compute_burst_overlap(2)).astype(int)

    if not max_burst:
        max_burst_ = slc.burst_count
    else:
        max_burst_ = max_burst

    if max_burst_ > min_burst:
        tmp_slc = f"{dir_out}/tmp_slc.tif"
    elif max_burst_ < min_burst:
        raise ValueError("max_burst must be >= min_burst")
    else:
        tmp_slc = f"{dir_out}/slc.tif"

    if (
        max_burst_ > slc.burst_count
        or max_burst_ < 1
        or min_burst > slc.burst_count
        or min_burst < 1
    ):
        raise ValueError(
            f"min_burst and max_burst must be values between 1 and {slc.burst_count}"
        )

    naz = slc.lines_per_burst * (max_burst_ - min_burst + 1)
    nrg = slc.samples_per_burst

    warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
    _child_process(
        _process_bursts_slc,
        (
            slc,
            tmp_slc,
            dir_out,
            dir_dem,
            naz,
            nrg,
            min_burst,
            max_burst_,
            dem_upsampling,
            dem_buffer_arc_sec,
            dem_force_download,
            overlap,
            cal_type,
        ),
    )

    if max_burst_ > min_burst:
        _child_process(
            _stitch_bursts,
            (
                tmp_slc,
                f"{dir_out}/slc.tif",
                slc.lines_per_burst,
                max_burst_ - min_burst + 1,
                overlap,
            ),
        )

    log.info("Cleaning temporary files")
    if max_burst_ > min_burst:
        if os.path.isfile(tmp_slc):
            os.remove(tmp_slc)


def geocode_and_merge_iw(
    input_dir: str,
    var_names: List[str],
    shp: shape = None,
    pol: Union[str, List[str]] = "full",
    subswaths: List[str] = ["IW1", "IW2", "IW3"],
    warp_kernel: str = "bicubic",
    clip_to_shape: bool = True,
) -> None:
    """Geocode and merge subswaths from the SAR geometry to the geographic coordinate system.

    Args:
        input_dir (str): Interferometric product directory.
        var_names (List[str]): List of the variable names to process. For instance ['coh', 'ifg', 'amp_prm']
        shp (shapely.geometry.shape, optional): Area of interest. Defaults to None.
        pol (Union[str, List[str]], optional):  Polarimetric channels to process (Either 'VH','VV, 'full' or a list like ['HV', 'VV']).  Defaults to "full".
        subswaths (List[str], optional): Limit the processing to a list of subswaths like `["IW1", "IW2"]`. Defaults to ["IW1", "IW2", "IW3"].
        multilook (List[int], optional): Multilooking in azimuth and range. Defaults to [1, 4].
        warp_kernel (str, optional): Warping kernel. Defaults to "bicubic".
        clip_to_shape (bool, optional): If set to True, whole bursts intersecting shp will be included. Defaults to True.
    Note:
        variables starting with the substring 'ifg' are interpreted as
        interferograms. Their phase will extracted after geocoding. The
        output file will start with 'phi'.
    """
    if isinstance(pol, str):
        if pol == "full":
            pol_ = ["vv", "vh"]
        else:
            pol_ = [pol.lower()]
    elif isinstance(pol, list):
        pol_ = [p.lower() for p in pol]
    else:
        raise RuntimeError("polarizations must be of type str or list")
    iw_idx = [iw[2] for iw in subswaths]

    for var in var_names:
        no_file_found = True
        for p in pol_:
            patterns = [f"{input_dir}/sar/{var}_{p}_iw{iw}.tif" for iw in iw_idx]

            matched_files = [pattern for pattern in patterns if os.path.isfile(pattern)]

            tmp_files = []
            if matched_files:
                no_file_found = False
            for file_var in matched_files:
                log.info(f"Geocode file {Path(file_var).name}.")
                base_name = Path(file_var).stem
                parts = base_name.split("_")
                postfix = "_".join(parts[-2:])
                file_lut = f"{input_dir}/sar/lut_{postfix}.tif"
                file_out = f"{input_dir}/sar/{var}_{postfix}_geo.tif"

                if not os.path.exists(file_lut):
                    raise FileNotFoundError(
                        f"Corresponding LUT file {file_lut} not found for {file_var}"
                    )

                # handling phase as a special case
                if var == "phi":
                    darr = riox.open_rasterio(file_var)
                    if not np.iscomplexobj(darr[0]):
                        warnings.warn(
                            "Geocode real-valued phase? If so, the result might not be optimal if the phase is wrapped."
                        )
                # if var == "ifg":
                if var.startswith("ifg"):
                    file_out = (
                        f"{input_dir}/sar/{var.replace("ifg", "phi")}_{postfix}_geo.tif"
                    )
                    sar2geo(
                        file_var,
                        file_lut,
                        file_out,
                        warp_kernel,
                        write_phase=True,
                        magnitude_only=False,
                    )
                else:
                    sar2geo(
                        file_var,
                        file_lut,
                        file_out,
                        warp_kernel,
                        write_phase=False,
                        magnitude_only=False,
                    )
                tmp_files.append(file_out)
            if tmp_files:
                # if var != "ifg":
                if not var.startswith("ifg"):
                    file_out = f"{input_dir}/{var}_{p}.tif"
                else:
                    file_out = f"{input_dir}/{var.replace("ifg", "phi")}_{p}.tif"
                log.info(f"Merge file {Path(file_out).name}")
                da_to_merge = [riox.open_rasterio(file) for file in tmp_files]

                if any(np.iscomplexobj(it) for it in da_to_merge):
                    raise NotImplementedError(
                        f"Trying to merge complex arrays ({var}). This is forbidden to prevent potential type casting errors."
                    )

                if shp and clip_to_shape:
                    merged = merge_arrays(
                        da_to_merge, parse_coordinates=False
                    ).rio.clip([shp], all_touched=True)
                else:
                    merged = merge_arrays(da_to_merge, parse_coordinates=False)
                merged.rio.to_raster(
                    file_out,
                    driver="COG",
                    compress="zstd",
                    num_threads="all_cpus",
                    resampling="nearest",
                    overview_resampling="nearest",
                )
                # clean tmp files
                for file in tmp_files:
                    remove(file)
        if no_file_found:
            raise FileNotFoundError(f"No file was found for variable {var}")


def sar2geo(
    sar_file: str,
    lut_file: str,
    out_file: str,
    kernel: str = "bicubic",
    write_phase: bool = False,
    magnitude_only: bool = False,
) -> None:
    """Reproject slc file to a geographic grid using a lookup table with optional multilooking.

    Args:
        sar_file (str): file in the SAR geometry
        lut_file (str): file containing a lookup table (output of the `preprocess_insar_iw` function)
        out_file (str): output file
        kernel (str): kernel used to align secondary SLC. Possible values are "nearest", "bilinear", "bicubic" and "bicubic6".Defaults to "bilinear".
        write_phase (bool): writes the array's phase . Defaults to False.
        magnitude_only (bool): writes the array's magnitude instead of its complex values. Has no effect it `write_phase` is True. Defaults to False.
    Note:
        Multilooking is recommended as it reduces the spatial resolution and mitigates speckle effects.
    """
    log.info("Project image with the lookup table.")

    with rio.open(sar_file) as ds_sar:
        arr = ds_sar.read(1)
        prof_src = ds_sar.profile.copy()
        trans_src = ds_sar.transform

    with rio.open(lut_file) as ds_lut:
        lut = ds_lut.read()
        prof_dst = ds_lut.profile.copy()

    if not trans_src.is_rectilinear:
        raise ValueError("The input dataset is not in the SAR geometry")
    if prof_src["count"] != 1:
        raise ValueError("Only single band rasters are supported.")

    if write_phase and not np.iscomplexobj(arr):
        warnings.warn(
            "write_phase: Trying to write phase of a real-valued array. This option will have no effect."
        )
    if magnitude_only and not np.iscomplexobj(arr):
        warnings.warn(
            "magnitude_only: Writing magnitude (absolute value) of a real-valued array."
        )

    # check if input was rescaled (multilook, etc.)
    sx = trans_src.a
    sy = trans_src.e
    arr_out = remap(arr, lut[0] / sy, lut[1] / sx, kernel)

    prof_dst.update({k: prof_src[k] for k in ["count", "dtype", "nodata"]})

    cog_dict = dict(
        driver="COG",
        compress="zstd",
        num_threads="all_cpus",
        resampling="nearest",
        overview_resampling="nearest",
    )
    # incompatible with COG, not needed (?) elsewhere
    prof_dst.pop("blockxsize", None)
    prof_dst.pop("blockysize", None)
    prof_dst.pop("tiled", None)
    prof_dst.pop("interleave", None)
    if write_phase and np.iscomplexobj(arr_out):
        phi = np.angle(arr_out)
        nodata = -9999
        phi[np.isnan(phi)] = nodata
        prof_dst.update({"dtype": phi.dtype.name, "nodata": nodata, **cog_dict})
        with rio.open(out_file, "w", **prof_dst) as dst:
            dst.write(phi, 1)
    else:
        if magnitude_only:
            mag = np.abs(arr_out)
            nodata = 0
            mag[np.isnan(mag)] = nodata
            prof_dst.update({"dtype": mag.dtype.name, "nodata": nodata, **cog_dict})
            with rio.open(out_file, "w", **prof_dst) as dst:
                dst.write(mag, 1)
        else:
            # Using COG only if real-valued
            if not np.iscomplexobj(arr_out):
                nodata = 0
                prof_dst.update({"driver": "COG", "nodata": nodata, **cog_dict})
                arr_out[np.isnan(arr_out)] = nodata
            else:
                prof_dst.update({"compress": "zstd", "num_threads": "all_cpus"})
            with rio.open(out_file, "w", **prof_dst) as dst:
                dst.write(arr_out, 1)


def apply_multilook(file_in: str, file_out: str, multilook: List = [1, 1]) -> None:
    """Apply multilooking to raster.

    Args:
        file_in (str): GeoTiff file of the primary SLC image
        file_out (str): output file
        multilook (list): number of looks in azimuth and range. Defaults to [1, 1]
        mlt_az (int): multilook in azimuth. Defaults to 1.
        mlt_rg (int): multilook in range. Defaults to 1.
    """

    if not isinstance(multilook, list):
        raise TypeError("Multilook must be a list like [mlt_az, mlt_rg]")
    else:
        mlt_az, mlt_rg = multilook

    log.info(f"Apply {mlt_az} by {mlt_rg} multilooking.")

    with rio.open(file_in) as ds_src:
        prof = ds_src.profile.copy()
        trans = ds_src.transform
        w_out = ds_src.width // mlt_rg
        h_out = ds_src.height // mlt_az
        prof.update(
            {
                "width": w_out,
                "height": h_out,
                "transform": trans * Affine.scale(mlt_rg, mlt_az),
            }
        )
        warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
        with rio.open(file_out, "w", **prof) as dst:
            if mlt_az > 1 or mlt_rg > 1:
                for i in range(prof["count"]):
                    arr = ds_src.read(i + 1)
                    arr_out = presum(arr, mlt_az, mlt_rg)
                    dst.write(arr_out, i + 1)


def amplitude(file_in: str, file_out: str, multilook: List = [1, 1]) -> None:
    """Compute the amplitude of a complex-valued image.

    Args:
        file_in (str): GeoTiff file of the primary SLC image
        file_out (str): output file
        multilook (list): number of looks in azimuth and range. Defaults to [1, 1]
    """

    if not isinstance(multilook, list):
        raise ValueError("Multilook must be a list like [mlt_az, mlt_rg]")
    else:
        mlt_az, mlt_rg = multilook

    log.info("Compute amplitude")
    with rio.open(file_in) as ds_slc:
        slc = ds_slc.read(1)
        prof = ds_slc.profile.copy()
        trans = ds_slc.transform

    amp = np.abs(slc)

    amp = presum(amp, mlt_az, mlt_rg)
    prof.update(
        {
            "width": amp.shape[1],
            "height": amp.shape[0],
            "transform": trans * Affine.scale(mlt_rg, mlt_az),
        }
    )

    warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
    # prof.update(
    # {"dtype": amp.dtype.name, "compress": "zstd", "num_threads": "all_cpus"}
    # )
    prof.update({"dtype": amp.dtype.name})
    with rio.open(file_out, "w", **prof) as dst:
        dst.write(amp, 1)


def interferogram(
    file_prm: str, file_sec: str, file_out: str, multilook: List = [1, 1]
) -> None:
    """Compute a complex interferogram from two SLC image files.

    Args:
        file_prm (str): GeoTiff file of the primary SLC image
        file_sec (str): GeoTiff file of the secondary SLC image
        file_out (str): output file
    """

    if not isinstance(multilook, list):
        raise ValueError("Multilook must be a list like [mlt_az, mlt_rg]")
    else:
        mlt_az, mlt_rg = multilook

    log.info("Compute interferogram")
    with rio.open(file_prm) as ds_prm:
        prm = ds_prm.read(1)
        prof = ds_prm.profile.copy()
        trans = ds_prm.transform
    with rio.open(file_sec) as ds_sec:
        sec = ds_sec.read(1)
    ifg = prm * sec.conj()
    ifg = presum(ifg, mlt_az, mlt_rg)
    prof.update(
        {
            "width": ifg.shape[1],
            "height": ifg.shape[0],
            "transform": trans * Affine.scale(mlt_rg, mlt_az),
        }
    )

    warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
    # prof.update({"compress": "zstd", "num_threads": "all_cpus"})
    with rio.open(file_out, "w", **prof) as dst:
        dst.write(ifg, 1)


def coherence(
    file_prm: str,
    file_sec: str,
    file_out: str,
    box_size: Union[int, List[int]] = 5,
    multilook: List = [1, 1],
    magnitude: bool = True,
    file_complex_ifg: str = None,
    filter_ifg: bool = True,
) -> None:
    """Compute the complex coherence from two SLC image files.

    Args:
        file_prm (str): GeoTiff file of the primary SLC image
        file_sec (str): GeoTiff file of the secondary SLC image
        file_out (str): output file
        box_size (int, optional): Window size in pixels for boxcar filtering. Defaults to 5.
        magnitude (bool, optional): Writes magnitude only. Otherwise a complex valued raster is written. Defaults to True.
        file_complex_ifg (str, optional): Writes complex interferogram as well. Defaults to None.
        filter_ifg (bool): Also applies boxcar to interferogram. Has no effect if file_complex_ifg is set to None. Defaults to True.
    """

    if not file_complex_ifg:
        log.info("Compute coherence")
    else:
        log.info("Compute coherence & interferogram")

    if isinstance(box_size, list):
        box_az = box_size[0]
        box_rg = box_size[1]
    else:
        box_az = box_size
        box_rg = box_size

    if not isinstance(multilook, list):
        raise ValueError("Multilook must be a list like [mlt_az, mlt_rg]")
    else:
        mlt_az, mlt_rg = multilook

    # open_args = dict(lock=False, chunks="auto", cache=True, masked=True)
    open_args = dict(lock=False, chunks=(1, 1024, 1024), cache=True, masked=True)

    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    ds_prm = riox.open_rasterio(file_prm, **open_args)
    ds_sec = riox.open_rasterio(file_sec, **open_args)

    # accessing dask arrays
    prm = ds_prm[0].data
    sec = ds_sec[0].data

    process_args = dict(
        dimaz=box_az,
        dimrg=box_rg,
        depth=(box_az, box_rg),
    )

    # we need these for interferogram
    ifg = prm * sec.conj()
    ifg_box = da.map_overlap(boxcar, ifg, **process_args, dtype="complex64")

    coh = ifg_box / np.sqrt(
        da.map_overlap(
            boxcar,
            np.nan_to_num((prm * prm.conj()).real),
            **process_args,
            dtype="float32",
        )
    )
    coh /= np.sqrt(
        da.map_overlap(
            boxcar,
            np.nan_to_num((sec * sec.conj()).real),
            **process_args,
            dtype="float32",
        )
    )

    if magnitude:
        coh = np.abs(coh)

    coh = presum(coh, mlt_az, mlt_rg)

    nodataval = np.nan

    da_coh = xr.DataArray(
        name="coh",
        data=coh[None],
        dims=("band", "y", "x"),
    )
    da_coh.rio.write_transform(
        ds_prm.rio.transform() * Affine.scale(mlt_rg, mlt_az), inplace=True
    )
    da_coh.rio.write_nodata(nodataval, inplace=True)
    da_coh.rio.to_raster(file_out)
    # da_coh.rio.to_raster(file_out, compress="zstd", num_threads="all_cpus")

    # useful as users may want non-filtered interferograms
    if file_complex_ifg:
        if filter_ifg:
            ifg_box = presum(ifg_box, mlt_az, mlt_rg)
            da_ifg = xr.DataArray(
                data=ifg_box[None],
                dims=("band", "y", "x"),
            )
        else:
            ifg = presum(ifg, mlt_az, mlt_rg)
            da_ifg = xr.DataArray(
                data=ifg[None],
                dims=("band", "y", "x"),
            )
        da_ifg.rio.write_transform(
            ds_prm.rio.transform() * Affine.scale(mlt_rg, mlt_az), inplace=True
        )
        da_ifg.rio.write_nodata(np.nan, inplace=True)
        da_ifg.rio.to_raster(
            file_complex_ifg, driver="GTiff", tiled=True, blockxsize=512, blockysize=512
        )


def goldstein(
    file_ifg: str, file_out: str, alpha: float = 0.5, overlap: int = 14
) -> None:
    """Apply the Goldstein filter to a complex interferogam to reduce phase noise.

    Args:
        file_ifg (str): Input file.
        file_out (str): Output file.
        alpha (float, optional): Filter parameter. Should be between 0 (no filtering) and 1 (strongest). Defaults to 0.5.
        overlap (int, optional): Overlap between 64x64 patches. Defaults to 14.
    Note:
        The method is described in:
        R.M. Goldstein and C.L. Werner, "Radar Interferogram Phase Filtering for Geophysical Applications," Geophysical Research Letters, 25, 4035-4038, 1998
    """

    # base filter to be applied on a patch
    def filter_base(arr, alpha=1):
        smooth = lambda x: uflt(x, 3)
        Z = fftshift(fft2(arr))
        H = smooth(abs(Z))
        # H /= H.max()
        H = H ** (alpha)
        # H = smooth(abs(Z)) ** (alpha)
        arrout = ifft2(ifftshift(H * Z))
        return arrout

    # base filter to be sequentially applied on a chunk
    def filter_chunk(chunk, alpha=0.5, overlap=14):
        # complex phase
        chunk_ = np.exp(1j * np.angle(chunk))
        # overlap value found in modified Goldstein paper
        return block_process(
            chunk_, (32-overlap//2, 32-overlap//2), (overlap//2, overlap//2), filter_base, alpha=alpha
            # chunk_, (64, 64), (overlap, overlap), filter_base, alpha=alpha
        )

    # TODO: find a way to automatically tune chunk size
    open_args = dict(lock=False, chunks=(1, 2048, 2048), masked=True)
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    ds_ifg = riox.open_rasterio(file_ifg, **open_args)
    ifg = ds_ifg[0].data

    # process multiple chunks in parallel
    process_args = dict(alpha=alpha, depth=(32, 32), dtype="complex64")
    ifg_out = da.map_overlap(filter_chunk, ifg, **process_args)
    da_out = xr.DataArray(
        name="ifg",
        data=ifg_out[None],
        dims=("band", "y", "x"),
    )
    da_out.rio.write_transform(ds_ifg.rio.transform(), inplace=True)

    nodataval = np.nan
    da_out.rio.write_nodata(nodataval, inplace=True)
    # block size manually set until better solution
    da_out.rio.to_raster(file_out, tiled=True, blockxsize=512, blockysize=512)


def apply_to_patterns_for_pair(
    func: Callable,
    out_dir: str,
    file_prm_prefix: str,
    file_sec_prefix: str,
    file_out_prefix: str,
    *args,
    **kwargs,
) -> None:
    """Apply the given function to all file patterns for pairs of input files, skipping patterns with missing files.

    This function generates file paths based on predefined polarization (`vh`, `vv`)
    and interferometric wide-swath (IW) indices (`iw1`, `iw2`, `iw3`), then calls the
    provided function for each pattern with two input files and one output file, only
    if the input files exist.

    Args:
        func (Callable): The function to apply to each set of file paths. It should
            accept at least three string arguments for input and output file names.
        out_dir (str): The directory where the input and output files are located.
        file_prm_prefix (str): The prefix for the primary input file.
        file_sec_prefix (str): The prefix for the secondary input file.
        file_out_prefix (str): The prefix for the output file.
        *args: Additional positional arguments to pass to the `func`.
        **kwargs: Additional keyword arguments to pass to the `func`.

    Returns:
        None: The function is executed for each pattern where the input files exist, but no return value is expected.
    """
    pol = ["vh", "vv"]
    iw_idx = [1, 2, 3]
    patterns = [f"{p}_iw{iw}" for p in pol for iw in iw_idx]

    for pattern in patterns:
        file_prm = f"{out_dir}/{file_prm_prefix}_{pattern}.tif"
        file_sec = f"{out_dir}/{file_sec_prefix}_{pattern}.tif"
        file_out = f"{out_dir}/{file_out_prefix}_{pattern}.tif"

        # Check if both input files exist
        if os.path.exists(file_prm) and os.path.exists(file_sec):
            # Call the original function with updated file names
            log.info(
                f"Apply '{func.__name__}' to {Path(file_prm).name} and {Path(file_sec).name}"
            )
            func(file_prm, file_sec, file_out, *args, **kwargs)
            log.info(f"File {Path(file_out).name} written")


def apply_to_patterns_for_single(
    func: Callable,
    out_dir: str,
    file_in_prefix: str,
    file_out_prefix: str,
    *args,
    **kwargs,
) -> None:
    """Apply the given function to all file patterns for a single input file, skipping patterns with missing files.

    This function generates file paths based on predefined polarization (`vh`, `vv`)
    and interferometric wide-swath (IW) indices (`iw1`, `iw2`, `iw3`), then calls the
    provided function for each pattern with one input file and one output file, only
    if the input file exists.

    Args:
        func (Callable): The function to apply to each set of file paths. It should
            accept at least two string arguments for input and output file names.
        out_dir (str): The directory where the input and output files are located.
        file_in_prefix (str): The prefix for the input file.
        file_out_prefix (str): The prefix for the output file.
        *args: Additional positional arguments to pass to the `func`.
        **kwargs: Additional keyword arguments to pass to the `func`.

    Returns:
        None: The function is executed for each pattern where the input file exists, but no return value is expected.
    """
    pol = ["vh", "vv"]
    iw_idx = [1, 2, 3]
    patterns = [f"{p}_iw{iw}" for p in pol for iw in iw_idx]

    for pattern in patterns:
        file_in = f"{out_dir}/{file_in_prefix}_{pattern}.tif"
        file_out = f"{out_dir}/{file_out_prefix}_{pattern}.tif"

        # Check if the input file exists
        if os.path.exists(file_in):
            # Call the original function with updated file names
            log.info(f"Apply '{func.__name__}' to {Path(file_in).name}")
            func(file_in, file_out, *args, **kwargs)
            log.info(f"File {Path(file_out).name} written")


# Auxiliary functions which are not supposed to be used outside of the processor

# TODO: adjust with a burst offset parameter that will be applied to bursts of the secondary
# and will skip invalid (out of range)


def _process_bursts_insar(
    prm,
    sec,
    tmp_prm,
    tmp_sec,
    dir_out,
    dir_dem,
    naz,
    nrg,
    min_burst,
    max_burst,
    burst_offset,
    dem_upsampling,
    dem_buffer_arc_sec,
    dem_force_download,
    warp_kernel,
    overlap,
    cal_type,
):

    H = int(overlap / 2)
    prof_tmp = dict(
        width=nrg,
        height=naz,
        count=1,
        dtype="complex64",
        driver="GTiff",
        nodata=np.nan,
        # compress="zstd",
        # num_threads="all_cpus",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
    # process individual bursts
    file_dem = prm.fetch_dem(
        min_burst,
        max_burst,
        dir_dem,
        buffer_arc_sec=dem_buffer_arc_sec,
        force_download=dem_force_download,
        upscale_factor=dem_upsampling,
    )
    file_lut = f"{dir_out}/lut.tif"
    with rio.open(file_dem) as ds_dem:
        width_lut = ds_dem.width
        height_lut = ds_dem.height
        crs_lut = ds_dem.crs
        transform_lut = ds_dem.transform

    prof_lut = dict(
        width=width_lut,
        height=height_lut,
        count=2,
        dtype=np.float64,
        crs=crs_lut,
        transform=transform_lut,
        nodata=np.nan,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )

    arr_lut = np.full((2, height_lut, width_lut), fill_value=np.nan)

    with rio.open(tmp_prm, "w", **prof_tmp) as ds_prm, rio.open(
        tmp_sec, "w", **prof_tmp
    ) as ds_sec:
        off_az = 0
        for burst_idx in range(min_burst, max_burst + 1):
            log.info(f"---- Processing burst {burst_idx} ----")

            # compute geocoding LUTs (lookup tables) for primary and secondary bursts
            file_dem_burst = f"{dir_out}/dem_burst.tif"
            burst_geoms = prm.gdf_burst_geom
            burst_geom = burst_geoms[burst_geoms["burst"] == burst_idx].iloc[0]
            shp = burst_geom.geometry.buffer(dem_buffer_arc_sec / 3600)

            with rio.open(file_dem) as ds_dem:
                w = geometry_window(ds_dem, shapes=[shp])
                # window to read in the DEM
                burst_window = [w.col_off, w.row_off, w.width, w.height]
                # pixel position to write burst in the LUT
                slices = w.toslices()

            # use virtual raster to keep using the same geocoding function
            file_dem_burst = f"{dir_out}/dem_burst.vrt"
            gdal.Translate(
                destName=file_dem_burst,
                srcDS=file_dem,
                format="VRT",
                srcWin=burst_window,
                creationOptions=["BLOCKXSIZE=512", "BLOCKYSIZE=512"],
            )

            # this implementation upsamples DEM at download, not during geocoding
            az_p2g, rg_p2g, _ = prm.geocode_burst(
                file_dem_burst,
                burst_idx=burst_idx,
                dem_upsampling=1,
            )
            az_s2g, rg_s2g, _ = sec.geocode_burst(
                file_dem_burst,
                burst_idx=burst_idx + burst_offset,
                dem_upsampling=1,
            )

            # read primary and secondary burst rasters
            arr_p = prm.read_burst(burst_idx, True)
            arr_s = sec.read_burst(burst_idx + burst_offset, True)

            # radiometric calibration (beta or sigma nought)
            cal_p = prm.calibration_factor(burst_idx, cal_type=cal_type)
            cal_s = sec.calibration_factor(burst_idx + burst_offset, cal_type=cal_type)
            log.info("Apply calibration factor")
            arr_p /= cal_p
            arr_s /= cal_s

            # deramp secondary
            pdb_s = sec.deramp_burst(burst_idx + burst_offset)
            log.info("Apply phase deramping")
            arr_s *= np.exp(1j * pdb_s)

            # project Secondary LUT into Primary grid
            az_s2p, rg_s2p = coregister(arr_p, az_p2g, rg_p2g, az_s2g, rg_s2g)

            # warp raster secondary and deramping phase
            arr_s = align(arr_s, az_s2p, rg_s2p, warp_kernel)
            pdb_s = align(pdb_s, az_s2p, rg_s2p, warp_kernel)

            # reramp secondary
            log.info("Apply phase reramping")
            arr_s *= np.exp(-1j * pdb_s)

            # compute topographic phases
            rg_p = np.zeros(arr_p.shape[0])[:, None] + np.arange(0, arr_p.shape[1])
            pht_p = prm.phi_topo(rg_p).reshape(*arr_p.shape)
            pht_s = sec.phi_topo(rg_s2p.ravel()).reshape(*arr_p.shape)
            log.info("Apply topographic phase removal")
            pha_topo = np.exp(-1j * (pht_p - pht_s)).astype(np.complex64)

            arr_s *= pha_topo

            first_line = (burst_idx - min_burst) * prm.lines_per_burst

            # write the coregistered SLCs
            ds_prm.write(
                arr_p, 1, window=Window(0, first_line, nrg, prm.lines_per_burst)
            )
            ds_sec.write(
                arr_s,
                1,
                window=Window(0, first_line, nrg, prm.lines_per_burst),
            )

            # place overlapping burst LUT with azimuth offset
            if burst_idx > min_burst:
                msk_overlap = az_p2g < H
                az_p2g[msk_overlap] = np.nan
                rg_p2g[msk_overlap] = np.nan
            msk = ~np.isnan(az_p2g)
            arr_lut[0, slices[0], slices[1]][msk] = az_p2g[msk] + off_az
            arr_lut[1, slices[0], slices[1]][msk] = rg_p2g[msk]
            off_az += prm.lines_per_burst - 2 * H

    remove(file_dem_burst)

    with rio.open(file_lut, "w", **prof_lut) as ds_lut:
        ds_lut.write(arr_lut)


def _process_bursts_slc(
    slc,
    tmp_slc,
    dir_out,
    dir_dem,
    naz,
    nrg,
    min_burst,
    max_burst,
    dem_upsampling,
    dem_buffer_arc_sec,
    dem_force_download,
    overlap,
    cal_type,
):

    H = int(overlap / 2)
    prof_tmp = dict(
        width=nrg,
        height=naz,
        count=1,
        dtype="complex64",
        driver="GTiff",
        nodata=np.nan,
        # compress="zstd",
        # num_threads="all_cpus",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
    # process individual bursts
    file_dem = slc.fetch_dem(
        min_burst,
        max_burst,
        dir_dem,
        buffer_arc_sec=dem_buffer_arc_sec,
        force_download=dem_force_download,
        upscale_factor=dem_upsampling,
    )
    file_lut = f"{dir_out}/lut.tif"
    with rio.open(file_dem) as ds_dem:
        width_lut = ds_dem.width
        height_lut = ds_dem.height
        crs_lut = ds_dem.crs
        transform_lut = ds_dem.transform

    prof_lut = dict(
        width=width_lut,
        height=height_lut,
        count=2,
        dtype=np.float64,
        crs=crs_lut,
        transform=transform_lut,
        nodata=np.nan,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )

    arr_lut = np.full((2, height_lut, width_lut), fill_value=np.nan)

    with rio.open(tmp_slc, "w", **prof_tmp) as ds_prm:
        off_az = 0
        for burst_idx in range(min_burst, max_burst + 1):
            log.info(f"---- Processing burst {burst_idx} ----")

            # compute geocoding LUTs (lookup tables) for primary and secondary bursts
            file_dem_burst = f"{dir_out}/dem_burst.tif"
            burst_geoms = slc.gdf_burst_geom
            burst_geom = burst_geoms[burst_geoms["burst"] == burst_idx].iloc[0]
            shp = burst_geom.geometry.buffer(dem_buffer_arc_sec / 3600)

            with rio.open(file_dem) as ds_dem:
                w = geometry_window(ds_dem, shapes=[shp])
                # window to read in the DEM
                burst_window = [w.col_off, w.row_off, w.width, w.height]
                # pixel position to write burst in the LUT
                slices = w.toslices()

            # use virtual raster to keep using the same geocoding function
            file_dem_burst = f"{dir_out}/dem_burst.vrt"
            gdal.Translate(
                destName=file_dem_burst,
                srcDS=file_dem,
                format="VRT",
                srcWin=burst_window,
                creationOptions=["BLOCKXSIZE=512", "BLOCKYSIZE=512"],
            )

            # this implementation upsamples DEM at download, not during geocoding
            az_p2g, rg_p2g, _ = slc.geocode_burst(
                file_dem_burst,
                burst_idx=burst_idx,
                dem_upsampling=1,
            )

            # read primary and secondary burst rasters
            arr_p = slc.read_burst(burst_idx, True)

            # radiometric calibration (beta or sigma nought)
            cal_p = slc.calibration_factor(burst_idx, cal_type=cal_type)
            log.info("Apply calibration factor")
            arr_p /= cal_p

            first_line = (burst_idx - min_burst) * slc.lines_per_burst

            # write the coregistered SLCs
            ds_prm.write(
                arr_p, 1, window=Window(0, first_line, nrg, slc.lines_per_burst)
            )

            # place overlapping burst LUT with azimuth offset
            if burst_idx > min_burst:
                msk_overlap = az_p2g < H
                az_p2g[msk_overlap] = np.nan
                rg_p2g[msk_overlap] = np.nan
            msk = ~np.isnan(az_p2g)
            arr_lut[0, slices[0], slices[1]][msk] = az_p2g[msk] + off_az
            arr_lut[1, slices[0], slices[1]][msk] = rg_p2g[msk]
            off_az += slc.lines_per_burst - 2 * H

    remove(file_dem_burst)

    with rio.open(file_lut, "w", **prof_lut) as ds_lut:
        ds_lut.write(arr_lut)


def _apply_fast_esd(
    tmp_prm_file, tmp_sec_file, min_burst, max_burst, naz, nrg, overlap
):
    """Applies an in-place phase correction to burst (complex) interferograms to mitigate phase jumps between the bursts.
    Based on ideas introduced in:
    Qin, Y.; Perissin, D.; Bai, J. A Common “Stripmap-Like” Interferometric Processing Chain for TOPS and ScanSAR Wide Swath Mode. Remote Sens. 2018, 10, 1504.
    """
    x = np.arange(naz)
    xdown, xup = overlap / 2, naz - 1 - overlap / 2

    def make_ramp(phase_diffs, idx):
        if idx == 0:
            ydown, yup = -phase_diffs[idx] / 2, phase_diffs[idx] / 2
        elif idx == len(phase_diffs):
            ydown, yup = -phase_diffs[idx - 1] / 2, phase_diffs[idx - 1] / 2
        else:
            ydown, yup = -phase_diffs[idx - 1] / 2, phase_diffs[idx] / 2
        slope = (yup - ydown) / (xup - xdown)
        off = ydown - slope * xdown
        ramp = slope * x + off
        return np.exp(1j * (ramp[:, None] + np.zeros((nrg))))

    with rio.open(tmp_prm_file, "r") as ds_prm:
        with rio.open(tmp_sec_file, "r+") as ds_sec:
            # computing cross interferograms in overlapping areas
            log.info("Fast ESD: computing cross interferograms")
            phase_diffs = []
            for burst_idx in range(min_burst, max_burst):
                first_line_tail = (burst_idx - min_burst + 1) * naz - overlap
                first_line_head = (burst_idx - min_burst + 1) * naz
                # read last lines of current burst
                tail_p = ds_prm.read(
                    indexes=1, window=Window(0, first_line_tail, nrg, overlap)
                )
                tail_s = ds_sec.read(
                    indexes=1,
                    window=Window(0, first_line_tail, nrg, overlap),
                )
                # read first lines of next burst
                head_p = ds_prm.read(
                    indexes=1, window=Window(0, first_line_head, nrg, overlap)
                )
                head_s = ds_sec.read(
                    indexes=1,
                    window=Window(0, first_line_head, nrg, overlap),
                )
                ifg1 = tail_p * tail_s.conj()
                ifg2 = head_p * head_s.conj()
                cross_ifg = ifg1 * ifg2.conj()
                dphi_clx = cross_ifg[~np.isnan(cross_ifg)]
                phase_diffs.append(np.angle(dphi_clx.mean()))

            # making phase ramps and applying to secondary
            log.info("Fast ESD: applying phase corrections")
            for burst_idx in range(min_burst, max_burst + 1):
                first_line = (burst_idx - min_burst) * naz
                arr_s = ds_sec.read(
                    indexes=1,
                    window=Window(0, first_line, nrg, naz),
                )
                esd_ramp = make_ramp(phase_diffs, burst_idx - min_burst).astype(
                    np.complex64
                )
                ds_sec.write(
                    arr_s * esd_ramp,
                    indexes=1,
                    window=Window(0, first_line, nrg, naz),
                )


def _stitch_bursts(
    file_in, file_out, lines_per_burst, burst_count, overlap, off_burst=1
):
    warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
    H = int(overlap / 2)
    naz = lines_per_burst
    with rio.open(file_in) as src:
        nrg = src.width

        if burst_count >= 2:
            siz = (naz - H) * 2 + (burst_count - 2) * (naz - 2 * H)
        elif burst_count == 1:
            siz = naz - H
        else:
            raise ValueError("Empty burst list")

        prof = src.profile.copy()
        prof.update(
            dict(width=nrg, height=siz)
            # dict(width=nrg, height=siz, compress="zstd", num_threads="all_cpus")
        )
        with rio.open(file_out, "w", **prof) as dst:

            log.info("Stitch bursts to make a continuous image")
            off_dst = 0
            for i in range(burst_count):
                if i == 0:
                    nlines = naz - H
                    off_src = 0
                elif i == burst_count - 1:
                    nlines = naz - H
                    off_src = H
                else:
                    nlines = naz - 2 * H
                    off_src = H

                for j in range(src.count):
                    arr = src.read(
                        j + 1,
                        window=Window(
                            0, (i + off_burst - 1) * naz + off_src, nrg, nlines
                        ),
                    )
                    dst.write(
                        arr, window=Window(0, off_dst, nrg, nlines), indexes=j + 1
                    )
                off_dst += nlines


def _child_process(func, args):
    # convenience function to make code prettier
    if USE_CP:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as e:
            if isinstance(args, (list, tuple)):
                res = e.submit(func, *args).result()
            elif isinstance(args, dict):
                res = e.submit(func, **args).result()
            else:
                raise ValueError(
                    "Child process arguments should be tuple, list or dict"
                )
        return res
    else:
        if isinstance(args, (list, tuple)):
            return func(*args)
        elif isinstance(args, dict):
            return func(**args)
        else:
            raise ValueError("Function arguments should be tuple, list or dict")
