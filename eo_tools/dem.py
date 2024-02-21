from eodag import EODataAccessGateway
import rasterio
from rasterio.io import MemoryFile
import rasterio.merge
import rasterio.mask
from pathlib import Path
import logging
from .auxils import remove

log = logging.getLogger(__name__)


def retrieve_dem(
    shp, file_out, dem_name="cop-dem-glo-30", tmp_dir="/tmp", clear_tmp_files=True
):
    """Downloads a DEM for a given geometry from Microsoft Planetary Computer

    Args:
        shp (shapely shape): Geometry of the area of interest
        file_out (str, optional): Output file.
        dem_name (str, optional): One of the available collections ('3dep-seamless', 'alos-dem', 'cop-dem-glo-30', 'cop-dem-glo-90', 'nasadem'). Defaults to "cop-dem-glo-30".
        tmp_dir (str, optional): Temporary directory where the tiles to be merged and cropped will be stored. Defaults to "/tmp".
        clear_tmp_files (bool, optional): Delete original tiles. Set to False if these are to be reused.
    """

    dag = EODataAccessGateway()
    search_criteria = {
        "productType": dem_name,
        "provider": "planetary_computer",
        "geom": shp,
    }
    results = dag.search_all(**search_criteria)
    dl_dirs = dag.download_all(results, outputs_prefix=tmp_dir)
    dl_paths = [
        f"{tmp_dir}/{Path(name).stem}/{Path(name).stem}.tif" for name in dl_dirs
    ]
    to_merge = [rasterio.open(name) for name in dl_paths]
    prof = to_merge[0].profile.copy()

    log.info("Merging rasters")
    arr_merge, trans_merge = rasterio.merge.merge(to_merge)
    prof.update(
        {
            "height": arr_merge.shape[1],
            "width": arr_merge.shape[2],
            "transform": trans_merge,
            "nodata": 0,
            "count": 1,
        }
    )

    # crop without writing intermediate file
    log.info("Cropping rasters")
    with MemoryFile() as memfile:
        with memfile.open(**prof) as mem:
            # Populate the input file with numpy array
            mem.write(arr_merge)
            arr_crop, trans_crop = rasterio.mask.mask(mem, [shp], crop=True)
            prof_out = mem.profile.copy()
            prof_out.update(
                {
                    "transform": trans_crop,
                    "width": arr_crop.shape[2],
                    "height": arr_crop.shape[1],
                    "count": 1,
                }
            )

    log.info(f"Writing file {file_out}")
    with rasterio.open(file_out, "w", **prof_out) as dst:
        dst.write(arr_crop)

    if clear_tmp_files:
        log.info("---- Removing temporary files.")
        for dir_name in dl_dirs:
            remove(dir_name)
