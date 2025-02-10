import geopandas as gpd
from eo_tools.dem import retrieve_dem
import rioxarray as riox
from pathlib import Path
import numpy as np


def test_retrieve_dem(tmp_path):
    aoi_file = f"./data/test_dem_aoi.geojson"
    shp = gpd.read_file(aoi_file).geometry[0]

    output_file = tmp_path / "test_dem.tif"
    reference_file = Path("./data/test_dem_ref.tif")

    # Ensure the reference file exists
    assert reference_file.is_file(), "Reference file does not exist."

    retrieve_dem(shp, output_file)

    dem = riox.open_rasterio(output_file)
    ref_dem = riox.open_rasterio(reference_file)

    assert dem.shape == ref_dem.shape, "Shapes of the DEMs do not match."
    np.testing.assert_allclose(
        dem.data,
        ref_dem.data,
        rtol=1e-5,
        atol=1e-8,
        err_msg="DEM data does not match reference.",
    )
