import logging
from pystac_client.client import Client
import rioxarray as riox
from rioxarray.merge import merge_arrays

log = logging.getLogger(__name__)


def retrieve_dem(shp, file_out, dem_name="cop-dem-glo-30"):
    """Downloads a DEM for a given geometry from Microsoft Planetary Computer

    Args:
        shp (shapely shape): Geometry of the area of interest
        file_out (str, optional): Output file.
        dem_name (str, optional): One of the available collections ('alos-dem', 'cop-dem-glo-30', 'cop-dem-glo-90', 'nasadem'). Defaults to "cop-dem-glo-30".
        tmp_dir (str, optional): Temporary directory where the tiles to be merged and cropped will be stored. Defaults to "/tmp".
        clear_tmp_files (bool, optional): Delete original tiles. Set to False if these are to be reused.
    """

    log.info("Retrieve DEM")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

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
    dem.rio.to_raster(file_out)
