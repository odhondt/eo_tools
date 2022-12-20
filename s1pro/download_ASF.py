
import asf_search as asf
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path



## download specific product
def _download_product(args):
    product, path, session = args
    product.download(path=path, session=session)

def asf_downloader(shapefile, download_dir, mindate, maxdate, platform = 'Sentinel-1A', processinglevel = 'SLC', beammode = 'IW', polarization = 'VV+VH', username = None, password = None, processes = 1, **kwargs):
    gdf = gpd.read_file(shapefile)
    bounds = gdf.total_bounds
    gdf_bounds = gpd.GeoSeries([box(*bounds)])
    wkt_aoi = gdf_bounds.to_wkt().values.tolist()[0]

    results = asf.search(
        platform= platform,
        processingLevel=[processinglevel],
        start = mindate,
        end = maxdate,
        beamMode = beammode,
        polarization = polarization,
        intersectsWith = wkt_aoi,
        **kwargs
        )

    print(f'Total Images Found: {len(results)}')
    session = asf.ASFSession().auth_with_creds(username, password)
    
    dd = Path(download_dir)
    dd.mkdir(parents=True, exist_ok=True)

    print('Start download')
    if processes == 1:
        for product in tqdm(results):
            product.download(path=dd.as_posix(), session=session)
    else:
        args = [(product, dd.as_posix(), session) for product in results]
        with Pool(processes) as pool:
             max = len(results)
             with tqdm(total=max) as pbar:
                for i, _ in enumerate(pool.imap_unordered(_download_product, args)):
                    pbar.update()