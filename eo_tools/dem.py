from eodag import EODataAccessGateway
import rasterio
from rasterio.io import MemoryFile
from pathlib import Path


def retrieve_dem(shp, file_out, dem_name="cop-dem-glo-30", tmp_dir="/tmp"):
    """Downloads a DEM for a given geometry from Microsoft Planetary Computer

    Args:
        shp (shapely shape): Geometry of the area of interest
        file_out (str, optional): Output file.
        dem_name (str, optional): One of the available collections ('3dep-seamless', 'alos-dem', 'cop-dem-glo-30', 'cop-dem-glo-90', 'nasadem'). Defaults to "cop-dem-glo-30".
        tmp_dir (str, optional): Temporary directory where the tiles to be merged and cropped will be stored. Defaults to "/tmp".
    """

    dag = EODataAccessGateway()
    search_criteria = {
        "collection": dem_name,
        "provider": "planetary_computer",
        "geom": shp,
    }
    results = dag.search_all(**search_criteria)
    dl = dag.download_all(results, outputs_prefix=tmp_dir)
    to_merge = [
        rasterio.open(f"{tmp_dir}/{Path(name).stem}/{Path(name).stem}.tif")
        for name in dl
    ]
    prof = to_merge[0].profile.copy()
    arr_merge, trans_merge = rasterio.merge.merge(to_merge)
    print(arr_merge.shape)
    prof.update(
        {
            "height": arr_merge.shape[1],
            "width": arr_merge.shape[2],
            "transform": trans_merge,
            "nodata": 0,
            "count": 1
        }
    )

    # crop without writing intermediate file
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

    with rasterio.open(file_out, 'w', **prof_out) as dst:
        dst.write(arr_crop)
