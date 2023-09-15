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

    graph_path = "../graph/TOPSAR_coh_geocode_IW_to_geotiff.xml"

    # retrieve burst geometries
    gdf_burst_mst = get_burst_geometry(mst_file, 
                    target_subswaths=['IW1', 'IW2','IW3'], 
                    polarization='VV')
    gdf_burst_slv = get_burst_geometry(slv_file, 
                    target_subswaths=['IW1','IW2','IW3'], 
                    polarization='VV')

    # find what subswaths and bursts intersect AOI
    gdf_burst_mst = gdf_burst_mst[gdf_burst_mst.intersects(shp)]
    gdf_burst_slv = gdf_burst_slv[gdf_burst_slv.intersects(shp)]

    # identify corresponding subswaths
    sel_subsw_mst = gdf_burst_mst['subswath'].unique()
    sel_subsw_slv = gdf_burst_slv['subswath'].unique()
    unique_subswaths = sel_subsw_mst.append(sel_subsw_slv).unique()

    if pol == 'full':
        pol_ = ['VV', 'VH']
    elif isinstance(pol, list):
        pol_ = pol

    for subswath in unique_subswaths:
        # setting graph parameters
        wfl = Workflow(graph_path)
        wfl['Read'].parameters['file'] = mst_file
        wfl['Read(2)'].parameters['file'] = slv_file
        
        print(f"Processing subswath {subswath}")
        wfl['TOPSAR-Split'].parameters['subswath'] = subswath
        wfl['TOPSAR-Split(2)'].parameters['subswath'] = subswath

        bursts_mst = gdf_burst_mst[gdf_burst_mst['subswath']==subswath]['burst'].values
        burst_mst_min = bursts_mst.min()
        burst_mst_max = bursts_mst.max()
        wfl['TOPSAR-Split'].parameters['firstBurstIndex'] = burst_mst_min
        wfl['TOPSAR-Split'].parameters['lastBurstIndex'] = burst_mst_max
        
        bursts_slv = gdf_burst_slv[gdf_burst_slv['subswath']==subswath]['burst'].values
        burst_slv_min = bursts_slv.min()
        burst_slv_max = bursts_slv.max()
        wfl['TOPSAR-Split(2)'].parameters['firstBurstIndex'] = burst_slv_min
        wfl['TOPSAR-Split(2)'].parameters['lastBurstIndex'] = burst_slv_max

        wfl['Write'].parameters['file'] = f"{out_dir}/{subswath}_COH_geo.tif"
        wfl.write('/tmp/graph.xml')
        grp = groupbyWorkers('/tmp/graph.xml', n=1)
        gpt('/tmp/graph.xml', groups=grp, tmpdir='/data/tmp/')

        print(f"Removing dark edges after terrain correction")
        with rio.open(f"{out_dir}/{subswath}_COH_geo.tif", 'r') as src:
            prof = src.profile.copy()
            prof.update({
                "driver": "GTiff",
                "nodata": 0
                })
            struct = np.ones((15,15))
            with rio.open(f"{out_dir}/{subswath}_COH_geo_border.tif", 'w', **prof) as dst:
                for i in range(1, prof['count'] + 1):
                    band_src = src.read(i)
                    msk_src = band_src != 0
                    msk_dst = binary_erosion(msk_src, struct)
                    band_dst = band_src * msk_dst
                    dst.write(band_dst, i)

    print("Merging and cropping selected subswaths")

    # merge 
    to_merge = [rio.open(f"{out_dir}/{iw}_COH_geo_border.tif") for iw in unique_subswaths]
    arr_merge, trans_merge = merge.merge(to_merge)
    with rio.open(f"{out_dir}/{unique_subswaths[0]}_COH_geo_border.tif") as src:
            out_meta = src.meta.copy()    
    out_meta.update(
            {
                "height": arr_merge.shape[1],
                "width": arr_merge.shape[2],
                "transform": trans_merge,
                "nodata": 0
            }
        )

    # crop without writing intermediate file
    with MemoryFile() as memfile:
        with memfile.open(**out_meta) as mem:
            # Populate the input file with numpy array
            mem.write(arr_merge)
            arr_crop, trans_crop = mask.mask(mem, [shp], crop=True)
            prof_crop = mem.profile.copy() 
            prof_crop.update(
                    {
                        "transform": trans_crop,
                        "width": arr_crop.shape[2],
                        "height": arr_crop.shape[1]
                    }
                )

    # write as COG
    with MemoryFile() as memfile:
        with memfile.open(**prof_crop) as mem:
            mem.write(arr_crop)
            dst_profile = cog_profiles.get("deflate")
            cog_translate(
                mem,
                f"{out_dir}/mergedRasters.tif",
                dst_profile,
                in_memory=True,
                quiet=True,
            )
    # TODO: 
    # - loop on polarizations
    # - optional crop
    # - check polarizations in data
    # - remove temp files
    # - name files with dates (check if band name is in first geotiff)
    # - gpt options (?)
    # - subswaths as a parameter
    # - interferogram
    # - add some parameters