import numpy as np
from scipy.interpolate import griddata
import xarray as xr
import rasterio


def s5p_so2_to_cog(filein, outpath="/tmp"):

    ds = xr.load_dataset(filein, group="PRODUCT")

    darr = ds.sulfurdioxide_total_vertical_column.values[0].ravel()
    lon = ds.longitude.values[0].ravel()
    lat = ds.latitude.values[0].ravel()
    minlon, maxlon = lon.min(), lon.max()
    minlat, maxlat = lat.min(), lat.max()

    # TODO: compute size based on approx resolution
    x = np.linspace(minlon, maxlon, 500)
    y = np.linspace(minlat, maxlat, 500)

    xx, yy = np.meshgrid(x, y)

    res = griddata((lon, lat), darr, (xx, yy), method="linear")[::-1, ::-1]

    transform = rasterio.transform.from_bounds(
        minlon, minlat, maxlon, maxlat, res.shape[1], res.shape[0]
    )
    crs = "EPSG:4326"

    # TODO: parse filename from input
    fileout = f"{outpath}/regridded.tif"

    # TODO: write with COG driver
    with rasterio.open(
        fileout,
        "w",
        width=res.shape[1],
        height=res.shape[0],
        transform=transform,
        crs=crs,
        count=1,
        dtype="float32",
        nodata=np.nan,
    ) as ds_out:
        ds_out.write(res, 1)

    return fileout
