from pyroSAR.snap.auxil import Workflow, gpt, groupbyWorkers
from s1pro.auxils import get_burst_geometry
import geopandas as gpd
import rasterio as rio
import numpy as np
from rasterio import merge, mask
from rasterio.io import MemoryFile
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from scipy.ndimage import binary_erosion

def s1_insar_proc(mst_file, slv_file, out_dir, tmp_dir, shp=None, pol='full'):
    
    # find what iw and bursts intersect AOI
    gdf_burst_mst = get_burst_geometry(mst_file, 
                    target_subswaths=['IW1', 'IW2','IW3'], 
                    polarization='VV')
    gdf_burst_slv = get_burst_geometry(slv_file, 
                    target_subswaths=['IW1', 'IW2','IW3'], 
                    polarization='VV')
    
    gdf_burst_mst = gdf_burst_mst[gdf_burst_mst.intersects(shp)]
    gdf_burst_slv = gdf_burst_slv[gdf_burst_slv.intersects(shp)]
    selected_subswaths_mst = gdf_burst_mst['subswath'].unique()
    selected_subswaths_slv = gdf_burst_slv['subswath'].unique()