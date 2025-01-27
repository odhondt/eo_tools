# test if provider is available and download quickview for a recent product
# %%
# Uncomment the next block to test conda imports

# import sys
# sys.path.remove("/eo_tools")
# sys.path.append("/eo_tools/") # workaround to include eo_tools_dev
# import eo_tools
# print(f"EO-Tools imported from:")
# print(f"{eo_tools.__file__=}")

from eodag import EODataAccessGateway
import geopandas as gpd
import datetime
import os

# credentials need to be stored in the following file (see EODAG docs)
confpath = "/data/eodag_config.yml"
dag = EODataAccessGateway(user_conf_file_path=confpath)
# make sure cop_dataspace will be used
dag.set_preferred_provider("cop_dataspace")

# uncomment to test free alternative provider
# dag.set_preferred_provider("geodes")

import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)
# %%
# change to your custom locations
data_dir = "/data/S1"

aoi_file = "/eo_tools/data/Bretagne_AOI.geojson"
shp = gpd.read_file(aoi_file).geometry[0]

# set dates between today and 1 month
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=30)

search_criteria = {
    "productType": "S1_SAR_SLC",
    "start": str(start_date),
    "end": str(end_date),
    "geom": shp,
}

results = dag.search(**search_criteria)

out_dir = "/tmp"
out_path = f"{out_dir}/test_preview.png"
if os.path.exists(out_path):
    os.remove(out_path)

dl = results[0].get_quicklook(filename="test_preview.png", output_dir="/tmp")
