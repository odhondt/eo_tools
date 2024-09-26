# %%
import rioxarray as riox
import shapely
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from eo_tools.S2 import s2_band_info
import warnings

# change to your data directories
path_data = "/data"
path_s2 = f"{path_data}/S2"
product_name = "S2A_MSIL1C_20240215T063511_N0510_R134_T40KCB_20240215T080808"
zip_file = f"{path_s2}/{product_name}.zip"
wkt_string = "POLYGON ((55.7236 -21.2355, 55.7044 -21.2355, 55.7044 -21.2535, 55.7236 -21.2535, 55.7236 -21.2355))"
shp = shapely.from_wkt(wkt_string)

# %%
# read files, reproject and clip

# dataframe used to extract band combinations from complicated S2 datasets
df_bands = s2_band_info()

# sub-dataframes (for convenience)
df_rgb = df_bands.loc[["B4", "B3", "B2"]]
df_swir = df_bands.loc[["B12", "B11", "B8A"]]

# read subdatasets first
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
with rasterio.open(zip_file) as ds_rio:
    sdvars = ds_rio.subdatasets

# warning: selected bands must be in the same sub-dataset
da_rgb_0 = riox.open_rasterio(sdvars[df_rgb.iloc[0].subd])
da_swir_0 = riox.open_rasterio(sdvars[df_swir.iloc[0].subd])

# extract band indices
rgb_idx = list(df_rgb.idx - 1)
swir_idx = list(df_swir.idx - 1)

# select bands, reproject in common CRS and crop
da_rgb = da_rgb_0[rgb_idx].rio.reproject("EPSG:4326").rio.clip([shp])
da_swir = da_swir_0[swir_idx].rio.reproject("EPSG:4326").rio.clip([shp])

# this is to avoid a stupid error when writing
da_swir.attrs["long_name"] = tuple(da_swir.attrs["long_name"][i] for i in swir_idx)
da_rgb.attrs["long_name"] = tuple(da_rgb.attrs["long_name"][i] for i in rgb_idx)

# use this value to normalize the data
qv_rgb = da_rgb.attrs["QUANTIFICATION_VALUE"]
qv_swir = da_swir.attrs["QUANTIFICATION_VALUE"]


# %%
# save png previews
da_swir_viz = da_swir.rio.reproject_match(
    da_rgb, resampling=Resampling.bilinear, count=da_swir.rio.count
)

(255 * (da_rgb / da_rgb.max())).astype("uint8").rio.to_raster(f"{path_data}/rgb.png")
(255 * (da_swir_viz / da_swir_viz.max())).astype("uint8").rio.to_raster(
    f"{path_data}/swir.png"
)

# %%
# save GeoTIFFs with native resolution and normalization
if da_rgb.attrs["PROCESSING_BASELINE"] > 4:
    # read radiometric offsets
    rgb_off = da_rgb.attrs["RADIO_ADD_OFFSET"]
    swir_off = da_swir.attrs["RADIO_ADD_OFFSET"]
    da_rgb_cal = ((da_rgb.astype("float32") + rgb_off) / qv_rgb).clip(0)
    da_rgb_cal.rio.to_raster(f"{path_data}/rgb.tif")
    da_swir_cal = ((da_swir.astype("float32") + swir_off) / qv_swir).clip(0)
    da_swir_cal.rio.to_raster(f"{path_data}/swir.tif")
else:
    (da_rgb.astype("float32") / qv_rgb).rio.to_raster(f"{path_data}/rgb.tif")
    (da_swir.astype("float32") / qv_swir).rio.to_raster(f"{path_data}/swir.tif")