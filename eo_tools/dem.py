import logging
from pystac_client.client import Client
import rioxarray as riox
from rioxarray.merge import merge_arrays
from rasterio.enums import Resampling
import planetary_computer

log = logging.getLogger(__name__)


def retrieve_dem(shp, out_file, dem_name="cop-dem-glo-30", upscale_factor=1):
    """Downloads a DEM for a given geometry from Microsoft Planetary Computer

    Args:
        shp (shapely shape): Geometry of the area of interest
        out_file (str, optional): Output file.
        dem_name (str, optional): One of the available collections ('alos-dem', 'cop-dem-glo-30', 'cop-dem-glo-90', 'nasadem'). Defaults to "cop-dem-glo-30".
        tmp_dir (str, optional): Temporary directory where the tiles to be merged and cropped will be stored. Defaults to "/tmp".
        clear_tmp_files (bool, optional): Delete original tiles. Set to False if these are to be reused.
        upscale_factor (float, optional): Upsampling factor.
    """

    log.info(f"Retrieve DEM ({dem_name})")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    data_keys = {
        "nasadem": "elevation",
        "cop-dem-glo-30": "data",
        "cop-dem-glo-90": "data",
        "alos-dem": "data",
    }

    if dem_name not in data_keys.keys():
        raise ValueError(f"Unknown DEM. Values are {list(data_keys.keys())}.")

    search = catalog.search(collections=[dem_name], intersects=shp)
    items = search.item_collection()

    to_merge = []
    for item in items:
        url = item.assets[data_keys[dem_name]].href
        da = riox.open_rasterio(url)
        to_merge.append(da)

    dem = merge_arrays(to_merge).rio.clip([shp], all_touched=True)
    if upscale_factor == 1:
        dem.rio.to_raster(out_file)
    elif upscale_factor > 0:
        log.info("Resample DEM")
        new_width = int(dem.rio.width * upscale_factor)
        new_height = int(dem.rio.height * upscale_factor)
        dem_upsampled = dem.rio.reproject(
            dem.rio.crs,
            shape=(new_height, new_width),
            resampling=Resampling.bilinear,
        )
        dem_upsampled.rio.to_raster(
            out_file, tiled=True, blockxsize=512, blockysize=512
        )
    else:
        raise ValueError("Upsampling factor must be positive.")
