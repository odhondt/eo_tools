import pyroSAR
from pyroSAR.snap.auxil import parse_recipe, parse_node, gpt, execute
from pyroSAR import Archive, identify
from spatialist.ancillary import finder
from spatialist import crsConvert, Vector, Raster, bbox, intersect
import os, shutil
import glob
import datetime
from os.path import join, basename
import datetime
from datetime import datetime as dt
from datetime import timedelta
import itertools
import asf_search as asf
import geopandas as gpd
from shapely.geometry import box, Polygon
from tqdm import tqdm
from multiprocessing import Pool
import configparser
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET
import pandas as pd

##################################################################
# Functions
#################################################################

def get_config(config_file, proc_section):
    if not os.path.isfile(config_file):
        raise FileNotFoundError("Config file {} does not exist.".format(config_file))
    
    parser = configparser.ConfigParser(allow_no_value=True, converters={'_datetime': _parse_datetime})
    parser.read(config_file)
    out_dict = {}

    try:
        proc_sec = parser[proc_section]
    except KeyError:
        raise KeyError("Section '{}' does not exist in config file {}".format(proc_section, config_file))

    for k, v in proc_sec.items():
        if k == 'download':
            if v.lower() == 'true':
                v = True
        if k == 'processes':
            v = int(v)
        if k.endswith('date'):
            v = proc_sec.get_datetime(k)
        if k == 'int_proc':
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
        if k == 'coh_proc':
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
        if k == 'ha_proc':
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
        if k == 'ext_dem_egm':
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
        if k == 'clean_tmpdir':
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
        if k == 'osvfail':
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
        if k == 'radnorm':
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
        if k == 'msk_nodatval':
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
        if k == 'ext_dem':
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
        if k == 'subset':
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
        if k == 't_crs':
            v = int(v)
        if k == 'cohwinrg':
            v = int(v)
        if k == 'cohwinaz':
            v = int(v)
        if k == 'filtersizex':
            v = int(v)
        if k == 'filtersizey':
            v = int(v)
        if k == 'ml_rglook':
            v = int(v)
        if k == 'ml_azlook':
            v = int(v)
        if k == 'decomp_win_size':
            v = int(v)
        if k == 'ext_dem_nodatval':
            v = int(v)
        if k == 'res_int':
            v = int(v)
        if k == 'res_coh':
            v = int(v)
        if k == 'res_ha':
            v = int(v)
        
        
        if k == 'outdir_int':
            if v == "None":
                v =  None
            else:
                v = v
        if k == 'outdir_coh':
            if v == "None":
                v =  None
            else:
                v = v
        if k == 'outdir_ha':
            if v == "None":
                v =  None
            else:
                v = v
        if k == 'tmpdir':
            if v == "None":
                v =  None
            else:
                v = v
        if k == 'ext_dem_file':
            if v == "None":
                v =  None
            else:
                v = v
        if k == 'osvpath':
            if v == "None":
                v =  None
            else:
                v = v
        if k == 'firstburstindex':
            if v == "None":
                v =  None
            else:
                v = int(v)
        if k == 'lastburstindex':
            if v == "None":
                v =  None
            else:
                v = int(v)
        if k == 'iws':
            v = v.split(',')
        out_dict[k] = v
        if k == 'decompfeats':
            v = v.split(',')
        out_dict[k] = v
        if k == 'gpt_paras':
            if v == "None":
                v =  None
            else:
                v = v.split(',')
    return out_dict


def _parse_datetime(s):
    """Custom converter for configparser:
    https://docs.python.org/3/library/configparser.html#customizing-parser-behaviour"""
    if 'T' in s:
        try:
            return dt.strptime(s, '%Y-%m-%dT%H:%M:%S')
        except ValueError as e:
            raise Exception("Parameters 'mindate/maxdate': Could not parse '{}' with datetime format "
                            "'%Y-%m-%dT%H:%M:%S'".format(s)) from e
    else:
        try:
            return dt.strptime(s, '%Y-%m-%d')
        except ValueError as e:
            raise Exception("Parameters 'mindate/maxdate': Could not parse '{}' with datetime format "
                            "'%Y-%m-%d'".format(s)) from e


def _download_product(args):
    product, path, session = args
    product.download(path=path, session=session)

def asf_downloader(shapefile, download_dir, mindate, maxdate, platform = 'Sentinel-1A', processinglevel = 'SLC', beammode = 'IW', polarization = 'VV+VH', username = None, password = None, archive = None, processes = 1, **kwargs):
    gdf = gpd.read_file(shapefile)
    bounds = gdf.total_bounds
    gdf_bounds = gpd.GeoSeries([box(*bounds)])
    wkt_aoi = gdf_bounds.to_wkt().values.tolist()[0]
    #start = dt.strptime(f'{mindate[0:4]}-{mindate[4:6]}-{mindate[7:8]}', '%Y-%m-%d')
    #end = dt.strptime(f'{maxdate[0:4]}-{maxdate[4:6]}-{maxdate[7:8]}', '%Y-%m-%d')
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
    
    print('Start download')
    if processes == 1:
        for product in tqdm(results):
            product.download(path=download_dir, session=session)
    else:
        args = [(product, download_dir, session) for product in results]
        with Pool(processes) as pool:
             max = len(results)
             with tqdm(total=max) as pbar:
                for i, _ in enumerate(pool.imap_unordered(_download_product, args)):
                    pbar.update()
    
    if archive != None:
        print('Creating archive')
        scenes_s1 = finder(download_dir, [r'^S1[AB].*\.zip'], regex=True, recursive=True)
        for scene in tqdm(scenes_s1):
            with Archive(archive) as a:
                scene = identify(scene)
                if a.is_registered(scene.scene):
                    print(f'{scene} already in database')
                else:
                    a.insert(scene)


def group_by_info(infiles, group= None):
##sort files by rel. orbit number
    info= pyroSAR.identify_many(infiles, sortkey= group)
    ##extract file paths of sorted files
    fps_lst=[]
    for fp in info:
        fp_str=fp.scene
        fps_lst.append(fp_str)

    ##extract and identify unique keys
    groups= []
    for o in info:
        orb= eval("o."+ group)
        groups.append(orb)

    query_group= groups.count(groups[0]) == len(groups)
    unique_groups= list(set(groups))

    out_files=[]
    if query_group == True:
        out_files=infiles
    else:
        group_idx= [] 
        #index files of key
        for a in unique_groups:
            tmp_groups=[]
            for idx, elem in enumerate(groups):
                    if(a == elem):
                        tmp_groups.append(idx)

            group_idx.append(tmp_groups) 
        ###group by same keyword 
        for i in range(0, len(group_idx)):
            fpsN= list(map(fps_lst.__getitem__, group_idx[i]))
            out_files.append(fpsN)
        
    return(out_files)

def Reproc_by_ErrorLog(dir_log, fp_S1, coh_dist):
    ##read .log files from directory
    pattern= "S1*proc*.log"
    ls_fl= glob.glob(join(dir_log, pattern))
    ##extract filedate from error log name
    d2rp= []
    for l in ls_fl:
        bname= basename(l)
        bn_split= bname.split("_")

        if bn_split[1] == "COH":
            date1= bn_split[4]
            date2= bn_split[5].split(".")[0]

            d2rp.extend([date1, date2])
        else:
            date1= bn_split[4].split(".")[0]
            d2rp.append(date1)

    d2rp_uniq=list(set(d2rp))

    ##extract dates from SLC data and sort list
    info_lst= pyroSAR.identify_many(fp, sortkey='start')
    d_S1= []
    fp_S1_sorted= []
    for i in info_lst:
        d_S1.append(i.start)
        fp_S1_sorted.append(i.scene)

    dates_idx= [] 

    for a in d2rp_uniq:
        tmp_dates=[]
        for idx, elem in enumerate(d_S1):
                if(a == elem):
                    tmp_dates.append(idx)

        dates_idx.append(tmp_dates)
    ##flatten list of SLC dates
    dates_idx= [y for x in dates_idx for y in x]
    ##match dates of SLC and error logs 
    fp2rp= list(map(fp_S1_sorted.__getitem__, dates_idx))
    ##sort matched filepaths and get datetimes
    fp2rp_sorted=[]
    fp2rp_dates=[]
    s1_info=pyroSAR.identify_many(fp2rp, sortkey="start")
    
    for s1 in s1_info:
        fp2rp_sorted.append(s1.scene)
        fp2rp_dates.append(s1.start)
    ##calculate temporal distance between scenes for matching to InSAR coherence intervals
    temp_dst= []
    for fps in range(0, len(fp2rp_sorted)-1):
        date1= fp2rp_dates[fps].split("T")[0]
        date1= dt.strptime(date1, "%Y%m%d")

        date2= fp2rp_dates[fps+1].split("T")[0]
        date2= dt.strptime(date2, "%Y%m%d")

        dt_diff=date2-date1
        temp_dst.append(dt_diff)
    ##get dates of elemets where the temporal distances exceeds coherence interval
    brk_pts= [ _ for _ in itertools.compress(fp2rp_dates, map(lambda x: x > timedelta(days=coh_dist), temp_dst)) ]

    dates_idx= [] 
    ## get index of elements where the temporal distances exceeds coherence interval
    for t in brk_pts:
        tmp_dates=[]
        for idx, elem in enumerate(fp2rp_dates):
                if(t == elem):
                    tmp_dates.append(idx)

        dates_idx.append(tmp_dates)
    ##flatten index list
    dates_idx= list(itertools.chain(*dates_idx))
   
    ##create nested list with SLC filepaths that match the coherence interval
    fp2rp_nst= []
    
    for di in range(0, len(dates_idx)):
        if di == 0:
            sublst= fp2rp_sorted[0: dates_idx[di]+1]
        elif (di != 0 and di <= len(dates_idx)-2):
            sublst= fp2rp_sorted[dates_idx[di]: dates_idx[di+1]+1]
        else:
            sublst= fp2rp_sorted[dates_idx[di]+1:]
        
        fp2rp_nst.append(sublst)
        
    return(fp2rp_nst)

def load_metadata(zip_path, subswath, polarization):
    archive = ZipFile(zip_path)
    archive_files = archive.namelist()
    regex_filter = r's1(?:a|b)-iw\d-slc-(?:vv|vh|hh|hv)-.*\.xml'
    metadata_file_list = []
    for item in archive_files:
        #if 'calibration' in item:
        #    continue
        match = re.search(regex_filter, item)
        if match:
            metadata_file_list.append(item)
    target_file = None
    for item in metadata_file_list:
        if subswath.lower() in item and polarization.lower() in item:
            target_file = item
    return archive.open(target_file)

def parse_location_grid(metadata):
    tree = ET.parse(metadata)
    root = tree.getroot()
    lines = []
    coord_list = []
    for grid_list in root.iter('geolocationGrid'):
        for point in grid_list:
            for item in point:
                lat = item.find('latitude').text
                lon = item.find('longitude').text
                line = item.find('line').text
                lines.append(line)
                coord_list.append((float(lat), float(lon)))
    total_num_bursts = len(set(lines)) - 1

    return total_num_bursts, coord_list

def parse_subswath_geometry(coord_list, total_num_bursts):
    def get_coords(index, coord_list):
        coord = coord_list[index]
        assert isinstance(coord[1], float)
        assert isinstance(coord[0], float)
        return coord[1], coord[0]

    bursts_dict = {}
    top_right_idx = 0
    top_left_idx = 20
    bottom_left_idx = 41
    bottom_right_idx = 21

    for burst_num in range(1, total_num_bursts + 1):
        burst_polygon = Polygon(
            [
                [get_coords(top_right_idx, coord_list)[0], get_coords(top_right_idx, coord_list)[1]],  # Top right
                [get_coords(top_left_idx, coord_list)[0], get_coords(top_left_idx, coord_list)[1]],  # Top left
                [get_coords(bottom_left_idx, coord_list)[0], get_coords(bottom_left_idx, coord_list)[1]],  # Bottom left
                [get_coords(bottom_right_idx, coord_list)[0], get_coords(bottom_right_idx, coord_list)[1]] # Bottom right
            ]
        )

        top_right_idx += 21
        top_left_idx += 21
        bottom_left_idx += 21
        bottom_right_idx += 21

        bursts_dict[burst_num] = burst_polygon

    return bursts_dict

def get_burst_geometry(path, target_subswaths, polarization):
    df_all = gpd.GeoDataFrame(columns=['subswath', 'burst', 'geometry'], crs='EPSG:4326')
    for subswath in target_subswaths:
        meta = load_metadata(zip_path = path, subswath = subswath, polarization = polarization)
        total_num_bursts, coord_list = parse_location_grid(meta)
        subswath_geom = parse_subswath_geometry(coord_list, total_num_bursts)
        df = gpd.GeoDataFrame(
                    {'subswath': [subswath.upper()] * len(subswath_geom),
                     'burst': [x for x in subswath_geom.keys()],
                     'geometry': [x for x in subswath_geom.values()]
                    },
                    crs='EPSG:4326'
                )
        df_all = gpd.GeoDataFrame(pd.concat([df_all, df]), crs='EPSG:4326')
    return(df_all)

def S1_INT_proc(infiles, out_dir= None, tmpdir= None, shapefile=None, t_res=20, t_crs=32633,  out_format= "GeoTIFF", gpt_paras= None, pol= 'full',\
                    IWs= ["IW1", "IW2", "IW3"], ext_DEM= False, ext_DEM_noDatVal= -9999, ext_Dem_file= None, msk_noDatVal= False,\
                    ext_DEM_EGM= True, imgResamp= "BICUBIC_INTERPOLATION", demResamp= "BILINEAR_INTERPOLATION",\
                    speckFilter= "Boxcar", filterSizeX= 5, filterSizeY= 5, ml_RgLook= 4, ml_AzLook= 1, ref_plain= "gamma",\
                    l2dB_arg= True, firstBurstIndex= None, lastBurstIndex= None, osvPath= None, clean_tmpdir= True, osvFail= False):
    
    ##define formatName for reading zip-files
    formatName= "SENTINEL-1"
    ##specify tmp output format
    tpm_format= "BEAM-DIMAP"
    ## create temp dir for intermediate .dim files
    if tmpdir is None:
        tmpdir= os.path.join(os.getcwd(), "tmp_dim")
        if os.path.isdir(tmpdir) == False:
            os.mkdir(tmpdir)
    ##check if a single IW or consecutive IWs are selected
    if isinstance(IWs, str):
        IWs= [IWs]
    if sorted(IWs) == ["IW1", "IW3"]:
        raise RuntimeError("Please select single or consecutive IW")
    
    ##extract info about files and order them by date
    ##handle length and type of infiles: str or list
    if isinstance(infiles, str):
        info= pyroSAR.identify(infiles)
        fps_lst=[info.scene]
        info_ms= info
        info= [info]
    elif isinstance(infiles, list):
        info= pyroSAR.identify_many(infiles, sortkey='start')
        ##collect filepaths sorted by date
        fps_lst=[]
        for fp in info:
            fp_str=fp.scene
            fps_lst.append(fp_str)
        
        info_ms= info[0]
    else:
        raise RuntimeError('Please provide str or list of filepaths') 
    ##query and handle polarisations, raise error if selected polarisations don't match (see Truckenbrodt et al.: pyroSAR: geocode)    
    if isinstance(pol, str):
        if pol == 'full':
            pol = info_ms.polarizations
        else:
            if pol in info_ms.polarizations:
                pol = [pol]
            else:
                raise RuntimeError('polarization {} does not exists in the source product'.format(pol))
    elif isinstance(pol, list):
        pol = [x for x in pol if x in info_ms.polarizations]
    else:
        raise RuntimeError('polarizations must be of type str or list')
    ##specify auto download DEM and handle external DEM file
    if ext_DEM == False:
        demName = 'SRTM 1Sec HGT'
        ext_DEM_file = None
    else:
        demName = "External DEM"
    ##raise error if no path to external file is provided
    if ext_DEM == True and ext_DEM_file == None:
        raise RuntimeError('No DEM file provided. Specify path to DEM-file')
    
    ##handle SNAP problem with WGS84 (EPSG: 4326) by manually constructing crs string (see Truckenbrodt et al.: pyroSAR: geocode)
    if t_crs == 4326:
        epsg= 'GEOGCS["WGS84(DD)",' \
                'DATUM["WGS84",' \
                'SPHEROID["WGS84", 6378137.0, 298.257223563]],' \
                'PRIMEM["Greenwich", 0.0],' \
                'UNIT["degree", 0.017453292519943295],' \
                'AXIS["Geodetic longitude", EAST],' \
                'AXIS["Geodetic latitude", NORTH]]'
    else:
        epsg="EPSG:{}".format(t_crs)
    ##check if correct DEM resampling methods are supplied
    reSamp_LookUp = ['NEAREST_NEIGHBOUR',
               'BILINEAR_INTERPOLATION',
               'CUBIC_CONVOLUTION',
               'BISINC_5_POINT_INTERPOLATION',
               'BISINC_11_POINT_INTERPOLATION',
               'BISINC_21_POINT_INTERPOLATION',
               'BICUBIC_INTERPOLATION']
    
    message = '{0} must be one of the following:\n- {1}'
    if demResamp not in reSamp_LookUp:
        raise ValueError(message.format('demResamplingMethod', '\n- '.join(reSamp_LookUp)))
    if imgResamp not in reSamp_LookUp:
        raise ValueError(message.format('imgResamplingMethod', '\n- '.join(reSamp_LookUp)))
     ##check if correct speckle filter option is supplied
    speckleFilter_options = ['Boxcar', 'Median', 'Frost', 'Gamma Map', 'Refined Lee', 'Lee', 'Lee Sigma']
    if speckFilter not in speckleFilter_options:
            raise ValueError(message.format('speckleFilter', '\n- '.join(speckleFilter_options)))
    ##query unique dates of files: determine if sliceAssembly is required
    dates_info= []
    for d in info:
        di= d.start.split("T")[0]
        dates_info.append(di)

    unique_dates_info= list(set(dates_info))
    unique_dates_info=sorted(unique_dates_info, key=lambda x: datetime.datetime.strptime(x, '%Y%m%d'))
    
    ##check for files of the same date and put them in sublists
    pair_dates_idx= [] 

    for a in unique_dates_info:
        tmp_dates=[]
        for idx, elem in enumerate(dates_info):
                if(a == elem):
                    tmp_dates.append(idx)

        pair_dates_idx.append(tmp_dates)
    ##selection of paired files for sliceAssembly
    for i in range(0, len(pair_dates_idx)):
        fps_grp= list(map(fps_lst.__getitem__, pair_dates_idx[i]))
        #get relative orbit number of grouped files
        info_tmp=pyroSAR.identify(fps_grp[0])
        relOrb= info_tmp.orbitNumber_rel
        sensor= info_tmp.sensor
        orbit= info_tmp.orbit
        date_str= info_tmp.start
        
        ##check availability of orbit state vector
        orbitType= "Sentinel Precise (Auto Download)"
        
        match = info_tmp.getOSV(osvType='POE', returnMatch=True, osvdir=osvPath)
        if match is None:
            info_tmp.getOSV(osvType='RES', osvdir=osvPath)
            orbitType = 'Sentinel Restituted (Auto Download)'
        
        slcAs_name= sensor +"_relOrb_"+ str(relOrb)+"_INT_"+unique_dates_info[i]+"_slcAs"
        slcAs_out= os.path.join(tmpdir, slcAs_name)

        graph_dir = f'{tmpdir}/graphs'
        isExist = os.path.exists(graph_dir)
        if not isExist:
            os.makedirs(graph_dir)
        
        ##exception handling for SNAP errors

        try:
            ## create workflow for sliceAssembly if more than 1 file is available per date
            if len(fps_grp) > 1:
                workflow_slcAs = parse_recipe("blank")

                read1 = parse_node('Read')
                read1.parameters['file'] = fps_grp[0]
                read1.parameters['formatName'] = formatName
                readers = [read1.id]

                workflow_slcAs.insert_node(read1)
                if shapefile:
                    bursts = get_burst_geometry(fps_grp[0], target_subswaths = ['iw1', 'iw2', 'iw3'], polarization = 'vv')
                    polygon = gpd.read_file(shapefile)
                    inter = bursts.overlay(polygon, how='intersection')
                    iw_list = inter['subswath'].unique()
                    iw_bursts = dict()
                    for ib in iw_list:
                        iw_inter = inter[inter['subswath'] == ib.upper()]
                        minb = iw_inter['burst'].min()
                        maxb = iw_inter['burst'].max()
                        iw_bursts[ib] =  [minb, maxb]
                    print(iw_bursts)


                for r in range(1, len(fps_grp)):
                    readn = parse_node('Read')
                    readn.parameters['file'] = fps_grp[r]
                    readn.parameters['formatName'] = formatName
                    workflow_slcAs.insert_node(readn, before= read1.id, resetSuccessorSource=False)
                    readers.append(readn.id)

                    if shapefile:
                        bursts = get_burst_geometry(fps_grp[r], target_subswaths = ['iw1', 'iw2', 'iw3'], polarization = 'vv')
                        polygon = gpd.read_file(shapefile)
                        inter = bursts.overlay(polygon, how='intersection')
                        iw_list = inter['subswath'].unique()
                        iw_bursts = dict()
                        for ib in iw_list:
                            iw_inter = inter[inter['subswath'] == ib.upper()]
                            minb = iw_inter['burst'].min()
                            maxb = iw_inter['burst'].max()
                            if ib in iw_bursts:
                                iw_bursts[ib][1]  =  iw_bursts[ib][1] + maxb
                            else:
                                iw_bursts[ib] =  [minb, maxb]
                        IWs = list(iw_bursts.keys())

                slcAs=parse_node("SliceAssembly")
                slcAs.parameters["selectedPolarisations"]= pol

                workflow_slcAs.insert_node(slcAs, before= readers)
                read1= slcAs
                last_node= slcAs.id

                write_slcAs=parse_node("Write")
                write_slcAs.parameters["file"]= slcAs_out
                write_slcAs.parameters["formatName"]= tpm_format

                workflow_slcAs.insert_node(write_slcAs, before= last_node)

                workflow_slcAs.write(f"{graph_dir}/INT_slc_prep_graph")

                gpt(f"{graph_dir}/INT_slc_prep_graph.xml", gpt_args= gpt_paras, tmpdir = tmpdir)

                INT_proc_in= slcAs_out+".dim"
            ##pass file path if no sliceAssembly required
            else:
                INT_proc_in = fps_grp[0]
                if shapefile:
                    bursts = get_burst_geometry(fps_grp[0], target_subswaths = ['iw1', 'iw2', 'iw3'], polarization = 'vv')
                    polygon = gpd.read_file(shapefile)
                    inter = bursts.overlay(polygon, how='intersection')
                    iw_list = inter['subswath'].unique()
                    iw_bursts = dict()
                    for ib in iw_list:
                        iw_inter = inter[inter['subswath'] == ib.upper()]
                        minb = iw_inter['burst'].min()
                        maxb = iw_inter['burst'].max()
                        iw_bursts[ib] =  [minb, maxb]
                    IWs = list(iw_bursts.keys())

            for p in pol:
                for iw in IWs:
                    tpm_name= sensor+"_" + p +"_INT_relOrb_"+ str(relOrb) + "_"+\
                        unique_dates_info[i]+ "_"+iw+"_2TPM"
                    tpm_out= os.path.join(tmpdir, tpm_name)
                    ##generate workflow for IW splits 
                    workflow= parse_recipe("blank")

                    read= parse_node("Read")
                    read.parameters["file"]= INT_proc_in
                    if len(fps_grp) == 1:
                        read.parameters["formatName"]= formatName
                    workflow.insert_node(read)

                    ##TOPSAR split node
                    ts=parse_node("TOPSAR-Split")
                    ts.parameters["subswath"]= iw
                    if iw_bursts[iw][0] is not None and iw_bursts[iw][1] is not None:
                        ts.parameters["firstBurstIndex"]= iw_bursts[iw][0]
                        ts.parameters["lastBurstIndex"]= iw_bursts[iw][1]
                    workflow.insert_node(ts, before=read.id)

                    aof=parse_node("Apply-Orbit-File")
                    aof.parameters["orbitType"]= orbitType
                    aof.parameters["polyDegree"]= 3
                    aof.parameters["continueOnFail"]= osvFail
                    workflow.insert_node(aof, before= ts.id)

                    cal= parse_node("Calibration")
                    cal.parameters["selectedPolarisations"]= pol
                    cal.parameters["createBetaBand"]= False
                    cal.parameters["outputBetaBand"]= True
                    cal.parameters["outputSigmaBand"]= False
                    
                    workflow.insert_node(cal, before= aof.id)

                    tpd=parse_node("TOPSAR-Deburst")
                    tpd.parameters["selectedPolarisations"]= pol
                    workflow.insert_node(tpd, before=cal.id)

                    write_tmp = parse_node("Write")
                    write_tmp.parameters["file"]= tpm_out
                    write_tmp.parameters["formatName"]= tpm_format
                    workflow.insert_node(write_tmp, before=tpd.id)

                    workflow.write(f"{graph_dir}/Int_proc_IW_graph")

                    execute(f"{graph_dir}/Int_proc_IW_graph.xml", gpt_args= gpt_paras)    

                ##load temporary files
                tpm_in= glob.glob(tmpdir+"/"+sensor+"_" + p +"_INT_relOrb_"+ str(relOrb) + "_"+\
                        unique_dates_info[i]+ "*_2TPM.dim")
                ##specify sourceBands for reference lvl beta and gamma
                if len(IWs)== 1:
                    ref= dict()
                    ref["beta"]= ["Beta0_"+IWs[0]+ "_" + p]
                    ref["gamma"]= ["Gamma0_"+IWs[0]+ "_"+ p]
                    ref["sigma"]= ["Sigma0_"+IWs[0]+ "_"+ p]
                else:
                    ref= dict()
                    ref["beta"]= ["Beta0_"+ p]
                    ref["gamma"]= ["Gamma0_"+ p]
                    ref["sigma"]= ["Sigma0_"+ p]
                ##assign sourceBands of nodes depending on reference lvl
                if ref_plain == "gamma":
                    ref_pl_ml= ref["beta"]
                    ref_pl= ref["gamma"]
                elif ref_plain == "sigma":
                    ref_pl_ml = ref["beta"]
                    ref_pl= ref["sigma"]
                    
                ## parse_workflow of INT processing
                workflow_tpm = parse_recipe("blank")

                read1 = parse_node('Read')
                read1.parameters['file'] = tpm_in[0]
                workflow_tpm.insert_node(read1)
                last_node= read1.id
                ##merge IWs if multiple IWs were selected
                if len(tpm_in) > 1:
                    readers= [read1.id]

                    for t in range(1, len(tpm_in)):
                        readn = parse_node('Read')
                        readn.parameters['file'] = tpm_in[t]
                        workflow_tpm.insert_node(readn, before= last_node, resetSuccessorSource=False)
                        readers.append(readn.id)
                    ##TOPSAR merge     
                    tpm=parse_node("TOPSAR-Merge")
                    tpm.parameters["selectedPolarisations"]=p
                    workflow_tpm.insert_node(tpm, before=readers)
                    last_node= tpm.id

                ##multi looking
                ml= parse_node("Multilook")
                ml.parameters["sourceBands"]= ref_pl_ml
                ml.parameters["nRgLooks"]= ml_RgLook
                ml.parameters["nAzLooks"]= ml_AzLook
                ml.parameters["grSquarePixel"]= True
                ml.parameters["outputIntensity"]= False
                workflow_tpm.insert_node(ml,before= last_node) 
                ##terrain flattening
                tf= parse_node("Terrain-Flattening")
                tf.parameters["sourceBands"]= ref_pl_ml
                tf.parameters["demName"]= demName
                tf.parameters["demResamplingMethod"]= demResamp
                tf.parameters["externalDEMFile"]= ext_Dem_file
                tf.parameters["externalDEMNoDataValue"]= ext_DEM_noDatVal
                tf.parameters["externalDEMApplyEGM"]= True
                tf.parameters["additionalOverlap"]= 0.1
                tf.parameters["oversamplingMultiple"]= 1.0
                if ref_plain == "sigma":
                    tf.parameters["outputSigma0"]= True

                workflow_tpm.insert_node(tf, before= ml.id)
                #speckle filtering
                sf= parse_node("Speckle-Filter")
                sf.parameters["sourceBands"]=ref_pl
                sf.parameters["filter"]= speckFilter
                sf.parameters["filterSizeX"]= filterSizeX
                sf.parameters["filterSizeY"]= filterSizeY

                workflow_tpm.insert_node(sf, before= tf.id)
                #terrain correction
                tc= parse_node("Terrain-Correction")
                tc.parameters["sourceBands"]= ref_pl
                tc.parameters["demName"]= demName
                tc.parameters["externalDEMFile"]= ext_Dem_file
                tc.parameters["externalDEMNoDataValue"]= ext_DEM_noDatVal
                tc.parameters["externalDEMApplyEGM"]= ext_DEM_EGM
                tc.parameters["imgResamplingMethod"]= imgResamp
                tc.parameters["demResamplingMethod"]= demResamp
                tc.parameters["pixelSpacingInMeter"]= t_res
                tc.parameters["mapProjection"]= t_crs
                tc.parameters["saveSelectedSourceBand"]= True
                #tc.parameters["outputComplex"]= False
                tc.parameters["nodataValueAtSea"]= msk_noDatVal

                workflow_tpm.insert_node(tc, before= sf.id)
                last_node= tc.id

                ## generate str for final output file based on selected IWs
                if len(IWs) == 1:
                    #out_name= sensor+"_"+ orbit+ "_relOrb_"+ str(relOrb) + "_INT_"+ p + "_"+ IWs[0] + "_"+\
                    #    date_str+"_Orb_Cal_Deb_ML_TF_Spk_TC"
                    out_name= sensor+"_"+ orbit+ "_relOrb_"+ str(relOrb) + "_INT_"+ p + "_"+\
                        date_str+"_Orb_Cal_Deb_ML_TF_Spk_TC"
                elif len(IWs)== 2:
                    separator = "_"
                    iw_str= separator.join(IWs)
                    out_name= sensor+"_"+ orbit+ "_relOrb_"+ str(relOrb) + "_INT_"+ p + "_"+\
                        iw_str +"_" + date_str+"_Orb_Cal_Deb_ML_TF_Spk_TC"
                else:
                    out_name= sensor+"_"+ orbit+ "_relOrb_"+ str(relOrb) + "_INT_"+ p + "_"+\
                        date_str+"_Orb_Cal_Deb_ML_TF_Spk_TC"
                ##create default output folder for each selected polarization
                if out_dir is None:
                    out_dir_p= os.path.join("INT", p)
                    if os.path.isdir(out_dir_p) == False:
                        os.makedirs(os.path.join(os.getcwd(), out_dir_p))
                else:
                    filename = sensor+"_"+ orbit+ "_relOrb_"+ str(relOrb) + "_INT_"+\
                        date_str+"_Orb_Cal_Deb_ML_TF_Spk_TC"
                    out_folder = f'{out_dir}/{filename}'
                    isExist = os.path.exists(out_folder)
                    if not isExist:
                        os.makedirs(out_folder)
                    out_dir_p = f'{out_dir}/{filename}'

                out_path= os.path.join(out_dir_p, out_name)   


                ##conversion from linear to dB if selected
                if l2dB_arg == True:
                    l2DB= parse_node("LinearToFromdB")
                    l2DB.parameters["sourceBands"]= ref_pl
                    workflow_tpm.insert_node(l2DB, before= last_node)
                    last_node= l2DB.id
                    ##change output name to reflect dB conversion
                    out_name= out_name+ "_dB"

                if shapefile:
                    if isinstance(shapefile, dict):
                        ext = shapefile
                    else:
                        if isinstance(shapefile, Vector):
                            shp = shapefile.clone()
                        elif isinstance(shapefile, str):
                            shp = Vector(shapefile)
                        else:
                            raise TypeError("argument 'shapefile' must be either a dictionary, a Vector object or a string.")
                        # reproject the geometry to WGS 84 latlon
                        shp.reproject(4326)
                        ext = shp.extent
                        shp.close()
                    # add an extra buffer of 0.01 degrees
                    buffer = 0.01
                    ext['xmin'] -= buffer
                    ext['ymin'] -= buffer
                    ext['xmax'] += buffer
                    ext['ymax'] += buffer
                    with bbox(ext, 4326) as bounds:
                        inter = intersect(info_ms.bbox(), bounds)
                        if not inter:
                            raise RuntimeError('no bounding box intersection between shapefile and scene')
                        inter.close()
                        wkt = bounds.convert2wkt()[0]

                    subset = parse_node('Subset')
                    #subset.parameters['region'] = [0, 0, 0, 0]
                    subset.parameters['geoRegion'] = wkt
                    subset.parameters['copyMetadata'] = True
                    workflow_tpm.insert_node(subset, before=last_node)
                    last_node = subset.id

                write_tpm=parse_node("Write")
                write_tpm.parameters["file"]= out_path
                write_tpm.parameters["formatName"]= out_format
                workflow_tpm.insert_node(write_tpm, before= last_node)

                ##write graph and execute it
                workflow_tpm.write(f"{graph_dir}/Int_TPM_continued_proc_graph")

                execute(f"{graph_dir}/Int_TPM_continued_proc_graph.xml", gpt_args= gpt_paras)
        #exception for SNAP errors & creating error log        
        except RuntimeError as e:
            print(str(e))
            with open("S1_INT_proc_ERROR_"+date_str+".log", "w") as logf:
                logf.write(str(e))
        ##clean tmp folder to avoid overwriting errors even if exception is valid
            files = glob.glob(os.path.join(tmpdir, '*'))
            for f in files:
                if os.path.isfile(f) or os.path.islink(f):
                    os.unlink(f)
                elif os.path.isdir(f):
                    shutil.rmtree(f)
            
            continue
            
        ##clean tmp folder to avoid overwriting errors    
        files = glob.glob(os.path.join(tmpdir, '*'))
        for f in files:
            if os.path.isfile(f) or os.path.islink(f):
                os.unlink(f)
            elif os.path.isdir(f):
                shutil.rmtree(f)
    ##delete tmp folder after processing
    if clean_tmpdir == True:
        print(tmpdir)
        #shutil.rmtree(tmpdir)

def S1_HA_proc(infiles, out_dir= None, tmpdir= None, shapefile = None, t_res=20, t_crs=32633,  out_format= "GeoTIFF", gpt_paras= None,\
                    IWs= ["IW1", "IW2", "IW3"], decompFeats= ["Alpha", "Entropy", "Anisotropy"], ext_DEM= False, ext_DEM_noDatVal= -9999, ext_Dem_file= None, msk_noDatVal= False,\
                    ext_DEM_EGM= True, imgResamp= "BICUBIC_INTERPOLATION", demResamp= "BILINEAR_INTERPOLATION",decomp_win_size= 5 ,\
                    speckFilter= "Box Car Filter", ml_RgLook= 4, ml_AzLook= 1, osvPath=None,\
                    firstBurstIndex= None, lastBurstIndex= None, clean_tmpdir= True, osvFail= False):
    
    ##define formatName for reading zip-files
    formatName= "SENTINEL-1"
    ##specify tmp output format
    tpm_format= "BEAM-DIMAP"
    ## create temp dir for intermediate .dim files
    if tmpdir is None:
        tmpdir= os.path.join(os.getcwd(), "tmp_dim")
        if os.path.isdir(tmpdir) == False:
            os.mkdir(tmpdir)
    ##check if a single IW or consecutive IWs are selected
    if isinstance(IWs, str):
        IWs= [IWs]
    if sorted(IWs) == ["IW1", "IW3"]:
        raise RuntimeError("Please select single or consecutive IW")
    
    ##extract info about files and order them by date
    ##handle length and type of infiles: str or list
    if isinstance(infiles, str):
        info= pyroSAR.identify(infiles)
        fps_lst=[info.scene]
        info_ms= info
        info= [info]
    elif isinstance(infiles, list):
        info= pyroSAR.identify_many(infiles, sortkey='start')
        ##collect filepaths sorted by date
        fps_lst=[]
        for fp in info:
            fp_str=fp.scene
            fps_lst.append(fp_str)
        
        info_ms= info[0]
    else:
        raise RuntimeError('Please provide str or list of filepaths') 
    ##query and handle polarisations, raise error if selected polarisations don't match (see Truckenbrodt et al.: pyroSAR: geocode)    
    ##specify auto download DEM and handle external DEM file
    if ext_DEM == False:
        demName = 'SRTM 1Sec HGT'
        ext_DEM_file = None
    else:
        demName = "External DEM"
    ##raise error if no path to external file is provided
    if ext_DEM == True and ext_DEM_file == None:
        raise RuntimeError('No DEM file provided. Specify path to DEM-file')
    ##raise error ifwrong decomp feature
    
    ##handle SNAP problem with WGS84 (EPSG: 4326) by manually constructing crs string (see Truckenbrodt et al.: pyroSAR: geocode)
    if t_crs == 4326:
        epsg= 'GEOGCS["WGS84(DD)",' \
                'DATUM["WGS84",' \
                'SPHEROID["WGS84", 6378137.0, 298.257223563]],' \
                'PRIMEM["Greenwich", 0.0],' \
                'UNIT["degree", 0.017453292519943295],' \
                'AXIS["Geodetic longitude", EAST],' \
                'AXIS["Geodetic latitude", NORTH]]'
    else:
        epsg="EPSG:{}".format(t_crs)
    ##check if correct DEM resampling methods are supplied
    reSamp_LookUp = ['NEAREST_NEIGHBOUR',
               'BILINEAR_INTERPOLATION',
               'CUBIC_CONVOLUTION',
               'BISINC_5_POINT_INTERPOLATION',
               'BISINC_11_POINT_INTERPOLATION',
               'BISINC_21_POINT_INTERPOLATION',
               'BICUBIC_INTERPOLATION']
    
    message = '{0} must be one of the following:\n- {1}'
    if demResamp not in reSamp_LookUp:
        raise ValueError(message.format('demResamplingMethod', '\n- '.join(reSamp_LookUp)))
    if imgResamp not in reSamp_LookUp:
        raise ValueError(message.format('imgResamplingMethod', '\n- '.join(reSamp_LookUp)))
     ##check if correct speckle filter option is supplied
    speckleFilter_options = ['Box Car Filter', 'IDAN Filter', 'Refined Lee Filter', 'Improved Lee Sigma Filter']
    if speckFilter not in speckleFilter_options:
            raise ValueError(message.format('speckleFilter', '\n- '.join(speckleFilter_options)))
    ##query unique dates of files: determine if sliceAssembly is required
    dates_info= []
    for d in info:
        di= d.start.split("T")[0]
        dates_info.append(di)

    unique_dates_info= list(set(dates_info))
    unique_dates_info=sorted(unique_dates_info, key=lambda x: datetime.datetime.strptime(x, '%Y%m%d'))
    
    ##check for files of the same date and put them in sublists
    pair_dates_idx= [] 

    for a in unique_dates_info:
        tmp_dates=[]
        for idx, elem in enumerate(dates_info):
                if(a == elem):
                    tmp_dates.append(idx)

        pair_dates_idx.append(tmp_dates)
    
    ##selection of paired files for sliceAssembly
    for i in range(0, len(pair_dates_idx)):
        fps_grp= list(map(fps_lst.__getitem__, pair_dates_idx[i]))
        #get relative orbit number of grouped files
        info_tmp=pyroSAR.identify(fps_grp[0])
        relOrb= info_tmp.orbitNumber_rel
        sensor= info_tmp.sensor
        orbit= info_tmp.orbit
        pol= info_tmp.polarizations
        date_str= info_tmp.start
        
        ##check availability of orbit state vector
        orbitType= "Sentinel Precise (Auto Download)"
        
        match = info_tmp.getOSV(osvType='POE', returnMatch=True, osvdir=osvPath)
        if match is None:
            info_tmp.getOSV(osvType='RES', osvdir=osvPath)
            orbitType = 'Sentinel Restituted (Auto Download)'
        ##exception handling of SNAP errors    
        try:
        
            slcAs_name= sensor +"_relOrb_"+ str(relOrb)+"_HA_"+unique_dates_info[i]+"_slcAs"
            slcAs_out= os.path.join(tmpdir, slcAs_name)
            graph_dir = f'{tmpdir}/graphs'
            isExist = os.path.exists(graph_dir)
            if not isExist:
                os.makedirs(graph_dir)
            ## create workflow for sliceAssembly if more than 1 file is available per date
            if len(fps_grp) > 1:

                workflow_slcAs = parse_recipe("blank")

                read1 = parse_node('Read')
                read1.parameters['file'] = fps_grp[0]
                read1.parameters['formatName'] = formatName
                readers = [read1.id]

                workflow_slcAs.insert_node(read1)
                
                if shapefile:
                    bursts = get_burst_geometry(fps_grp[0], target_subswaths = ['iw1', 'iw2', 'iw3'], polarization = 'vv')
                    polygon = gpd.read_file(shapefile)
                    inter = bursts.overlay(polygon, how='intersection')
                    iw_list = inter['subswath'].unique()
                    iw_bursts = dict()
                    for ib in iw_list:
                        iw_inter = inter[inter['subswath'] == ib.upper()]
                        minb = iw_inter['burst'].min()
                        maxb = iw_inter['burst'].max()
                        iw_bursts[ib] =  [minb, maxb]


                for r in range(1, len(fps_grp)):
                    readn = parse_node('Read')
                    readn.parameters['file'] = fps_grp[r]
                    readn.parameters['formatName'] = formatName
                    workflow_slcAs.insert_node(readn, before= read1.id, resetSuccessorSource=False)
                    readers.append(readn.id)

                    if shapefile:
                        bursts = get_burst_geometry(fps_grp[r], target_subswaths = ['iw1', 'iw2', 'iw3'], polarization = 'vv')
                        polygon = gpd.read_file(shapefile)
                        inter = bursts.overlay(polygon, how='intersection')
                        iw_list = inter['subswath'].unique()
                        iw_bursts = dict()
                        for ib in iw_list:
                            iw_inter = inter[inter['subswath'] == ib.upper()]
                            minb = iw_inter['burst'].min()
                            maxb = iw_inter['burst'].max()
                            if ib in iw_bursts:
                                iw_bursts[ib][1]  =  iw_bursts[ib][1] + maxb
                            else:
                                iw_bursts[ib] =  [minb, maxb]
                        IWs = list(iw_bursts.keys())

                slcAs=parse_node("SliceAssembly")
                slcAs.parameters["selectedPolarisations"]= pol

                workflow_slcAs.insert_node(slcAs, before= readers)
                read1= slcAs

                write_slcAs=parse_node("Write")
                write_slcAs.parameters["file"]= slcAs_out
                write_slcAs.parameters["formatName"]= tpm_format

                workflow_slcAs.insert_node(write_slcAs, before= slcAs.id)
                workflow_slcAs.write(f"{graph_dir}/HA_slc_prep_graph")
                gpt(f"{graph_dir}/HA_slc_prep_graph.xml", gpt_args= gpt_paras, tmpdir = tmpdir)

                HA_proc_in= slcAs_out+".dim"
            ##pass file path if no sliceAssembly required
            else:
                HA_proc_in = fps_grp[0]
                if shapefile:
                    bursts = get_burst_geometry(fps_grp[0], target_subswaths = ['iw1', 'iw2', 'iw3'], polarization = 'vv')
                    polygon = gpd.read_file(shapefile)
                    inter = bursts.overlay(polygon, how='intersection')
                    iw_list = inter['subswath'].unique()
                    iw_bursts = dict()
                    for ib in iw_list:
                        iw_inter = inter[inter['subswath'] == ib.upper()]
                        minb = iw_inter['burst'].min()
                        maxb = iw_inter['burst'].max()
                        iw_bursts[ib] =  [minb, maxb]
                    IWs = list(iw_bursts.keys())


            for iw in IWs:
                tpm_name= sensor +"_HA_relOrb_"+ str(relOrb) + "_"+\
                    unique_dates_info[i]+ "_"+iw+"_2TPM"
                tpm_out= os.path.join(tmpdir, tpm_name)
                ##generate workflow for IW splits 
                workflow= parse_recipe("blank")

                read= parse_node("Read")
                read.parameters["file"]= HA_proc_in
                if len(fps_grp) == 1:
                    read.parameters["formatName"]= formatName
                workflow.insert_node(read)

                aof=parse_node("Apply-Orbit-File")
                aof.parameters["orbitType"]= orbitType
                aof.parameters["polyDegree"]= 3
                aof.parameters["continueOnFail"]= osvFail
                workflow.insert_node(aof, before= read.id)
                ##TOPSAR split node
                ts=parse_node("TOPSAR-Split")
                ts.parameters["subswath"]= iw
                
                if iw_bursts[iw][0] is not None and iw_bursts[iw][1] is not None:
                    ts.parameters["firstBurstIndex"]= iw_bursts[iw][0]
                    ts.parameters["lastBurstIndex"]= iw_bursts[iw][1]
                workflow.insert_node(ts, before=aof.id)

                cal= parse_node("Calibration")
                cal.parameters["selectedPolarisations"]= pol
                cal.parameters["createBetaBand"]= False
                cal.parameters["outputBetaBand"]= False
                cal.parameters["outputSigmaBand"]= True
                cal.parameters["outputImageInComplex"]=True
                workflow.insert_node(cal, before= ts.id)

                tpd=parse_node("TOPSAR-Deburst")
                tpd.parameters["selectedPolarisations"]= pol
                workflow.insert_node(tpd, before=cal.id)

                write_tmp = parse_node("Write")
                write_tmp.parameters["file"]= tpm_out
                write_tmp.parameters["formatName"]= tpm_format
                workflow.insert_node(write_tmp, before=tpd.id)

                workflow.write("HA_proc_IW_graph")

                execute('HA_proc_IW_graph.xml', gpt_args= gpt_paras)    

            for dc in decompFeats:          
                dc_label= dc.upper()[0:3]
                ##load temporary files
                tpm_in= glob.glob(tmpdir+"/"+sensor+"_HA_relOrb_"+ str(relOrb) + "_"+\
                        unique_dates_info[i]+ "*_2TPM.dim")
                ## parse_workflow of INT processing
                workflow_tpm = parse_recipe("blank")

                read1 = parse_node('Read')
                read1.parameters['file'] = tpm_in[0]
                workflow_tpm.insert_node(read1)
                last_node= read1.id
                ##merge IWs if multiple IWs were selected
                if len(tpm_in) > 1:
                    readers= [read1.id]

                    for t in range(1, len(tpm_in)):
                        readn = parse_node('Read')
                        readn.parameters['file'] = tpm_in[t]
                        workflow_tpm.insert_node(readn, before= last_node, resetSuccessorSource=False)
                        readers.append(readn.id)
                    ##TOPSAR merge     
                    tpm=parse_node("TOPSAR-Merge")
                    tpm.parameters["selectedPolarisations"]=pol
                    workflow_tpm.insert_node(tpm, before=readers)
                    last_node= tpm.id
                ##create C2 covariance matrix
                polMat= parse_node("Polarimetric-Matrices")
                polMat.parameters["matrix"]= "C2"
                workflow_tpm.insert_node(polMat, before=last_node)
                last_node=polMat.id
                ##multi looking
                ml= parse_node("Multilook")
                ml.parameters["sourceBands"]=["C11", "C12_real", "C12_imag", "C22"]
                ml.parameters["nRgLooks"]= ml_RgLook
                ml.parameters["nAzLooks"]= ml_AzLook
                ml.parameters["grSquarePixel"]= True
                ml.parameters["outputIntensity"]= False
                workflow_tpm.insert_node(ml,before= last_node)
                last_node= ml.id

                ##polaricmetric speckle filtering
                polSpec= parse_node("Polarimetric-Speckle-Filter")
                polSpec.parameters["filter"]= speckFilter
                workflow_tpm.insert_node(polSpec, before= last_node)
                last_node= polSpec.id

                ##dual-pol H/a decomposition
                polDecp= parse_node("Polarimetric-Decomposition")
                polDecp.parameters["decomposition"]= "H-Alpha Dual Pol Decomposition"
                polDecp.parameters["windowSize"]= decomp_win_size
                polDecp.parameters["outputHAAlpha"]= True

                workflow_tpm.insert_node(polDecp, before= last_node)
                last_node= polDecp.id

                #terrain correction
                tc= parse_node("Terrain-Correction")
                tc.parameters["sourceBands"]= [dc]
                tc.parameters["demName"]= demName
                tc.parameters["externalDEMFile"]= ext_Dem_file
                tc.parameters["externalDEMNoDataValue"]= ext_DEM_noDatVal
                tc.parameters["externalDEMApplyEGM"]= ext_DEM_EGM
                tc.parameters["imgResamplingMethod"]= imgResamp
                tc.parameters["demResamplingMethod"]= demResamp
                tc.parameters["pixelSpacingInMeter"]= t_res
                tc.parameters["mapProjection"]= t_crs
                tc.parameters["saveSelectedSourceBand"]= True
               # tc.parameters["outputComplex"]= False
                tc.parameters["nodataValueAtSea"]= msk_noDatVal

                workflow_tpm.insert_node(tc, before= last_node)
                last_node= tc.id

                ## generate str for final output file based on selected IWs
                #if len(IWs) == 1:
                #out_name= sensor+"_"+ orbit+ "_relOrb_"+ str(relOrb) + "_HA_"+ dc_label + "_"+ IWs[0] + "_"+\
                #        date_str+"_Orb_Cal_Deb_ML_TF_Spk_TC"
                #elif len(IWs)== 2:
                #    separator = "_"
                #    iw_str= separator.join(IWs)
                #    out_name= sensor+"_"+ orbit+ "_relOrb_"+ str(relOrb) + "_HA_"+ dc_label + "_"+\
                #        iw_str +"_" + date_str+"_Orb_Cal_Deb_ML_Spk_TC"
                #else:
                out_name= sensor+"_"+ orbit+ "_relOrb_"+ str(relOrb) + "_HA_"+ dc_label + "_"+\
                    date_str+"_Orb_Cal_Deb_ML_Spk_TC"
                ##create default output folder for each selected polarization
                if out_dir is None:
                    out_dir_fp= os.path.join("HA", dc_label)
                    if os.path.isdir(out_dir_fp) == False:
                        os.makedirs(os.path.join(os.getcwd(), out_dir_fp))
                elif os.path.isdir(out_dir):
                    out_dir_fp = out_dir
                else:
                    raise RuntimeError("Please provide a valid filepath")

                if shapefile:
                    if isinstance(shapefile, dict):
                        ext = shapefile
                    else:
                        if isinstance(shapefile, Vector):
                            shp = shapefile.clone()
                        elif isinstance(shapefile, str):
                            shp = Vector(shapefile)
                        else:
                            raise TypeError("argument 'shapefile' must be either a dictionary, a Vector object or a string.")
                        # reproject the geometry to WGS 84 latlon
                        shp.reproject(4326)
                        ext = shp.extent
                        shp.close()
                    # add an extra buffer of 0.01 degrees
                    buffer = 0.01
                    ext['xmin'] -= buffer
                    ext['ymin'] -= buffer
                    ext['xmax'] += buffer
                    ext['ymax'] += buffer
                    with bbox(ext, 4326) as bounds:
                        inter = intersect(info_ms.bbox(), bounds)
                        if not inter:
                            raise RuntimeError('no bounding box intersection between shapefile and scene')
                        inter.close()
                        wkt = bounds.convert2wkt()[0]

                    subset = parse_node('Subset')
                    #subset.parameters['region'] = [0, 0, 0, 0]
                    subset.parameters['geoRegion'] = wkt
                    subset.parameters['copyMetadata'] = True
                    workflow_tpm.insert_node(subset, before=last_node)
                    last_node = subset.id 

                out_path= os.path.join(out_dir_fp, out_name)   

                write_tpm=parse_node("Write")
                write_tpm.parameters["file"]= out_path
                write_tpm.parameters["formatName"]= out_format
                workflow_tpm.insert_node(write_tpm, before= last_node)

                ##write graph and execute it
                workflow_tpm.write(f"{graph_dir}/HA_TPM_continued_proc_graph")
                execute(f"{graph_dir}/HA_TPM_continued_proc_graph.xml", gpt_args= gpt_paras)
        #exception for SNAP errors & creating error log    
        except RuntimeError as e:
            print(str(e))
            with open("S1_HA_proc_ERROR_"+date_str+".log", "w") as logf:
                logf.write(str(e))
            
            ##clean tmp folder to avoid overwriting errors even if exception is valid
            files = glob.glob(os.path.join(tmpdir, '*'))
            for f in files:
                if os.path.isfile(f) or os.path.islink(f):
                    os.unlink(f)
                elif os.path.isdir(f):
                    shutil.rmtree(f)
            
            continue     
            
        ##clean tmp folder to avoid overwriting errors    
        files = glob.glob(os.path.join(tmpdir, '*'))
        for f in files:
            if os.path.isfile(f) or os.path.islink(f):
                os.unlink(f)
            elif os.path.isdir(f):
                shutil.rmtree(f)
    ##delete tmp folder after processing
    if clean_tmpdir == True:
        shutil.rmtree(tmpdir)

def S1_InSAR_coh_proc(infiles, out_dir= "default", tmpdir= None, t_res=20, t_crs=32633,  out_format= "GeoTIFF",gpt_paras= None, pol= 'full',\
                   IWs= ["IW1", "IW2", "IW3"], ext_DEM= False, ext_DEM_noDatVal= -9999, ext_Dem_file= None, msk_noDatVal= False,\
                   ext_DEM_EGM= True, BGC_demResamp= "BICUBIC_INTERPOLATION", TC_demResamp= "BILINEAR_INTERPOLATION", osvPath= None,\
                   cohWinRg= 11, cohWinAz= 3, ml_RgLook= 4, ml_AzLook= 1, firstBurstIndex= None, lastBurstIndex= None, clean_tmpdir= True, osvFail= False):
    
    ##define formatName for reading zip-files
    formatName= "SENTINEL-1"
    ##list of abbreviated month for creation of source Bands string
    month_list= ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ##specify tmp output format
    tpm_format= "BEAM-DIMAP"
    ## create temp dir for intermediate .dim files
    if tmpdir is None:
        tmpdir= os.path.join(os.getcwd(), "tmp_dim")
        if os.path.isdir(tmpdir) == False:
            os.mkdir(tmpdir)
    ##queck if at least two files are loaded for coh estiamtion
    if len(infiles)==1:
        raise RuntimeError("At least 2 scenes needed for coherence estimation")
    
    ##check if a single IW or consecutive IWs are selected
    if isinstance(IWs, str):
        IWs= [IWs]
    if sorted(IWs) == ["IW1", "IW3"]:
        raise RuntimeError("Please select single or consecutive IW")
    ##extract info about files and order them by date
    info= pyroSAR.identify_many(infiles, sortkey='start')
    ##collect filepaths sorted by date
    fps_lst=[]
    for fp in info:
        fp_str=fp.scene
        fps_lst.append(fp_str)

    ##check if all files are of the same relative orbit
    relOrbs= []
    for o in info:
        orb= o.orbitNumber_rel
        relOrbs.append(orb)
        
    query_orb= relOrbs.count(relOrbs[0]) == len(relOrbs)
    ##raise error if different rel. orbits are detected
    if query_orb == False:
        raise RuntimeError(message.format("Files of different relative orbits detected"))
    ##query and handle polarisations, raise error if selected polarisations don't match (see Truckenbrodt et al.: pyroSAR: geocode)    
    info_ms= info[0]
    orbit= info_ms.orbit
    
    if isinstance(pol, str):
        if pol == 'full':
            pol = info_ms.polarizations
        else:
            if pol in info_ms.polarizations:
                pol = [pol]
            else:
                raise RuntimeError('polarization {} does not exists in the source product'.format(pol))
    elif isinstance(pol, list):
        pol = [x for x in pol if x in info_ms.polarizations]
    else:
        raise RuntimeError('polarizations must be of type str or list')
    ##specify auto download DEM and handle external DEM file
    if ext_DEM == False:
        demName = 'SRTM 1Sec HGT'
        ext_DEM_file = None
    else:
        demName = "External DEM"
    ##raise error if no path to external file is provided
    if ext_DEM == True and ext_DEM_file == None:
        raise RuntimeError('No DEM file provided. Specify path to DEM-file')
    
    ##handle SNAP problem with WGS84 (EPSG: 4236) by manually constructing crs string (see Truckenbrodt et al.: pyroSAR: geocode)
    if t_crs == 4326:
        epsg= 'GEOGCS["WGS84(DD)",' \
                'DATUM["WGS84",' \
                'SPHEROID["WGS84", 6378137.0, 298.257223563]],' \
                'PRIMEM["Greenwich", 0.0],' \
                'UNIT["degree", 0.017453292519943295],' \
                'AXIS["Geodetic longitude", EAST],' \
                'AXIS["Geodetic latitude", NORTH]]'
    else:
        epsg="EPSG:{}".format(t_crs)
    ##check if correct DEM resampling methods are supplied
    reSamp_LookUp = ['NEAREST_NEIGHBOUR',
               'BILINEAR_INTERPOLATION',
               'CUBIC_CONVOLUTION',
               'BISINC_5_POINT_INTERPOLATION',
               'BISINC_11_POINT_INTERPOLATION',
               'BISINC_21_POINT_INTERPOLATION',
               'BICUBIC_INTERPOLATION']
    
    message = '{0} must be one of the following:\n- {1}'
    if BGC_demResamp not in reSamp_LookUp:
        raise ValueError(message.format('demResamplingMethod', '\n- '.join(reSamp_LookUp)))
    if TC_demResamp not in reSamp_LookUp:
        raise ValueError(message.format('imgResamplingMethod', '\n- '.join(reSamp_LookUp)))
        
    ##query unique dates of files: selection of paired images for coherence estimation
    dates_info= []
    for d in info:
        di= d.start.split("T")[0]
        dates_info.append(di)

    unique_dates_info= list(set(dates_info))
    unique_dates_info=sorted(unique_dates_info, key=lambda x: datetime.datetime.strptime(x, '%Y%m%d'))
    ##raise error if only one unique date is supplied
    if len(unique_dates_info) == 1:
        raise RuntimeError("Please supply images from 2 different dates")

    ##check for files of the same date and put them in separate lists
    pair_dates_idx= [] 

    for a in unique_dates_info:
        tmp_dates=[]
        for idx, elem in enumerate(dates_info):
                if(a == elem):
                    tmp_dates.append(idx)

        pair_dates_idx.append(tmp_dates)
        
    ##selection of paired files for coherence estimation
    for i in range(0, len(pair_dates_idx)-1):
        fps1= list(map(fps_lst.__getitem__, pair_dates_idx[i])) 
        fps2= list(map(fps_lst.__getitem__, pair_dates_idx[i+1]))  
        
        fps_paired= [fps1, fps2]
        
        
        info_lst=[pyroSAR.identify(fps1[0]),\
                  pyroSAR.identify(fps2[0])]
        
        ##check availability of orbit state vector file 
        orbitType= "Sentinel Precise (Auto Download)"
        match = info_lst[0].getOSV(osvType='POE', returnMatch=True, osvdir=osvPath)
        match2= info_lst[1].getOSV(osvType='POE', returnMatch=True, osvdir=osvPath)
        if match is None or match2 is None:
            info_lst[0].getOSV(osvType='RES', osvdir=osvPath)
            info_lst[1].getOSV(osvType='RES', osvdir=osvPath)
            orbitType = 'Sentinel Restituted (Auto Download)'
        
        ##build sourceBands string for coherence estimation
        dates= []
        for i in info_lst:
            date=i.start.split("T")[0]
            date_int= int(date[4:6])
            month= month_list[date_int-1]
            date_tmp= date[6:8]+month+date[0:4]
            dates.append(date_tmp)

        ##extract dates as str from filename for the day and the full datetime
        date1= info_lst[0].start.split("T")[0]
        date2= info_lst[1].start.split("T")[0]
        
        datetime1= info_lst[0].start
        datetime2= info_lst[1].start
        ##exception handling against SNAP errors
        try:

            date_uniq=[date1, date2]
            ##manage numbers of scenes needed per time step to estimate coherence, initiate sliceAssembly if necessary
            if len(fps1)== 1 and len(fps2) == 1:
                slcAs_fps_slv= fps1[0]
                slcAs_fps_ms= fps2[0]
            else:
                if len(fps1) == 1 and len(fps2) > 1: 
                    slcAs_fps_slv= fps1[0]
                    idx_start= 1
                    idx_stop= len(fps_paired)
                elif len(fps1) > 1 and len(fps2) == 1:
                    slcAs_fps_ms= fps2[0]
                    idx_start= 0
                    idx_stop= len(fps_paired)-1
                else: 
                    idx_start= 0
                    idx_stop= len(fps_paired)
              ## initiate sliceAssembly where the time step consists of more than one scene  
                for fp in range(idx_start, idx_stop):
                    if fp == 0:
                        slcAs_name= "S1_relOrb_"+ str(relOrbs[0])+"_COH_"+date_uniq[fp]+"_SLC_slv"
                        slcAs_out= os.path.join(tmpdir, slcAs_name)
                    else:
                        slcAs_name= "S1_relOrb_"+ str(relOrbs[0])+"_COH_"+date_uniq[fp]+"_SLC_ms"
                        slcAs_out= os.path.join(tmpdir, slcAs_name)

                    workflow_slcAs = parse_recipe("blank")

                    read1 = parse_node('Read')
                    read1.parameters['file'] = fps_paired[fp][0]
                    read1.parameters['formatName'] = formatName
                    readers = [read1.id]

                    workflow_slcAs.insert_node(read1)

                    for r in range(1, len(fps_paired[fp])):
                        readn = parse_node('Read')
                        readn.parameters['file'] = fps_paired[fp][r]
                        readn.parameters['formatName'] = formatName
                        workflow_slcAs.insert_node(readn, before= read1.id, resetSuccessorSource=False)
                        readers.append(readn.id)

                    slcAs=parse_node("SliceAssembly")
                    slcAs.parameters["selectedPolarisations"]= pol

                    workflow_slcAs.insert_node(slcAs, before= readers)
                    read1= slcAs

                    write_slcAs=parse_node("Write")
                    write_slcAs.parameters["file"]= slcAs_out
                    write_slcAs.parameters["formatName"]= tpm_format

                    workflow_slcAs.insert_node(write_slcAs, before= slcAs.id)

                    workflow_slcAs.write("Coh_slc_prep_graph")

                    gpt('Coh_slc_prep_graph.xml', gpt_args= gpt_paras, tmpdir= tmpdir)

                ###import sliceAssemblies according to how many files per time step are needed   
                if len(fps1) > 1 and len(fps2) == 1:
                    slcAs_fps_slv= glob.glob(os.path.join(tmpdir, "*"+"_SLC_slv.dim"))
                elif len(fps1) == 1 and len(fps2) > 1:
                    slcAs_fps_ms= glob.glob(os.path.join(tmpdir, "*"+"_SLC_ms.dim"))
                elif len(fps1) > 1 and len(fps2) > 1: 
                    slcAs_fps_slv= glob.glob(os.path.join(tmpdir, "*"+"_SLC_slv.dim"))
                    slcAs_fps_ms= glob.glob(tmpdir+"/"+"*"+"_SLC_ms.dim")

            ##start coherence estimation for each IW
            for p in pol:
                for iw in IWs:
                    #my_source = "coh_"+ iw + "_"+ p+ "_"+ dates[1] +"_"+ dates[0]

                    ##create out_name
                    out_name= "S1_relOrb_"+ str(relOrbs[0])+ "_"+ iw +"_COH_"+ p + "_"+ date2+"_"+ date1+"_TPD"
                    tmp_out= os.path.join(tmpdir, out_name)

                    ##parse_workflows   
                    ##coherence calculation per IW
                    workflow_coh=parse_recipe("blank")

                    read1= parse_node("Read")
                    read1.parameters["file"]= slcAs_fps_ms
                    if len(fps2) == 1:
                        read1.parameters["formatName"]= formatName

                    workflow_coh.insert_node(read1)

                    aof=parse_node("Apply-Orbit-File")
                    aof.parameters["orbitType"]= orbitType
                    aof.parameters["polyDegree"]= 3
                    aof.parameters["continueOnFail"]= osvFail

                    workflow_coh.insert_node(aof, before= read1.id)

                    ts=parse_node("TOPSAR-Split")
                    ts.parameters["subswath"]= iw
                    ts.parameters["selectedPolarisations"]= p
                    #ts.parameters["firstBurstIndex"]= burst_span1[0]
                    #ts.parameters["lastBurstIndex"]= burst_span1[1]

                    workflow_coh.insert_node(ts, before= aof.id)

                    read2 = parse_node('Read')
                    read2.parameters['file'] = slcAs_fps_slv
                    if len(fps1) == 1:
                        read2.parameters['formatName'] = formatName

                    workflow_coh.insert_node(read2)

                    aof2= parse_node("Apply-Orbit-File")
                    aof2.parameters["orbitType"]= orbitType #'Sentinel Restituted (Auto Download)' Sentinel Precise (Auto Download)
                    aof2.parameters["polyDegree"]= 3
                    aof2.parameters["continueOnFail"]= osvFail

                    workflow_coh.insert_node(aof2, before=read2.id)

                    ts2=parse_node("TOPSAR-Split")
                    ts2.parameters["subswath"]= iw
                    ts2.parameters["selectedPolarisations"]= p
                    #ts2.parameters["firstBurstIndex"]= burst_span2[0]
                    #ts2.parameters["lastBurstIndex"]= burst_span2[1]

                    workflow_coh.insert_node(ts2, before= aof2.id)

                    bgc= parse_node("Back-Geocoding")
                    bgc.parameters["demName"]= demName
                    bgc.parameters["demResamplingMethod"]= BGC_demResamp
                    bgc.parameters["externalDEMFile"]= ext_Dem_file
                    bgc.parameters["externalDEMNoDataValue"]= ext_DEM_noDatVal
                    bgc.parameters["resamplingType"]= "BISINC_5_POINT_INTERPOLATION"
                    bgc.parameters["maskOutAreaWithoutElevation"]=msk_noDatVal

                    workflow_coh.insert_node(bgc, before= [ts.id, ts2.id])

                    coh= parse_node("Coherence")
                    coh.parameters["subtractFlatEarthPhase"]= True
                    coh.parameters["singleMaster"]= True
                    coh.parameters["cohWinRg"]= cohWinRg
                    coh.parameters["cohWinAz"]= cohWinAz
                    coh.parameters["demName"]= demName
                    coh.parameters["subtractTopographicPhase"]= True
                    coh.parameters["externalDEMFile"]= ext_Dem_file
                    coh.parameters["externalDEMNoDataValue"]= ext_DEM_noDatVal
                    coh.parameters["externalDEMApplyEGM"]= True

                    workflow_coh.insert_node(coh, before= bgc.id)

                    tpd=parse_node("TOPSAR-Deburst")
                    tpd.parameters["selectedPolarisations"]= p
                    workflow_coh.insert_node(tpd, before=coh.id)

                    write_coh=parse_node("Write")
                    write_coh.parameters["file"]= tmp_out
                    write_coh.parameters["formatName"]= tpm_format

                    workflow_coh.insert_node(write_coh, before= tpd.id)

                    ##write graph
                    workflow_coh.write("Coh_tmp_prep_graph")


                    ##execute graph via gpt
                    execute('Coh_tmp_prep_graph.xml', gpt_args= gpt_paras)

                ###combining the IWs
                ##filepaths of temporary files
                #search_criteria = "S1_relOrb_"+ str(info[0].orbitNumber_rel)+ "*"+p +"_"+ date2+"_"+ date1+"_TPD.dim"
                #dirpath= os.getcwd()
                #q = os.path.join(dirpath, search_criteria)
                tmp_fps= glob.glob(tmpdir+"/"+"S1_relOrb_"+ str(relOrbs[0])+"*"+p +"_"+ date2+"_"+ date1+"_TPD.dim")

                if len(IWs) == 1:
                    tpm_source= "coh_"+ IWs[0]+ "_"+ p+ "_"+ dates[1] +"_"+ dates[0]
                else:
                    tpm_source = "coh_"+ p+ "_"+ dates[1] +"_"+ dates[0]
                ##create outputname based on the number of selected IWs
                if len(IWs) == 3:
                    tpm_name= "S1_"+ orbit+"_relOrb_"+ str(relOrbs[0])+"_COH_"+ p + "_"+ datetime2+"_"+ datetime1
                else:
                    separator = "_"
                    iw_str= separator.join(IWs)
                    tpm_name= "S1_"+ orbit+"_relOrb_"+ str(relOrbs[0])+"_COH_"+ iw_str+"_"+ p + "_"+ datetime2+"_"+ datetime1
                ##create default output folder based on selected polarizations
                if out_dir is None:
                    out_dir_p= "COH/"+ p
                    if os.path.isdir(out_dir_p) == False:
                        os.makedirs(os.path.join(os.getcwd(), out_dir_p))
                elif os.path.isdir(out_dir):
                    out_dir_fp= out_dir
                else:
                    raise RuntimeError("Please provide a valid filepath")

                final_out_fp=os.path.join(out_dir_p, tpm_name)

                ##create workflow for merging
                workflow_tpm = parse_recipe("blank")

                read1 = parse_node('Read')
                read1.parameters['file'] = tmp_fps[0]
                workflow_tpm.insert_node(read1)
                ##handling multiple vs single IW
                if len(tmp_fps) > 1:
                    readers = [read1.id]

                    for t in range(1, len(tmp_fps)):
                        readn = parse_node('Read')
                        readn.parameters['file'] = tmp_fps[t]
                        workflow_tpm.insert_node(readn, before= read1.id, resetSuccessorSource=False)
                        readers.append(readn.id)

                    tpm=parse_node("TOPSAR-Merge")
                    tpm.parameters["selectedPolarisations"]=p

                    workflow_tpm.insert_node(tpm, before=readers)
                    last_id= tpm.id

                else:
                    last_id = read1.id
                
                if shapefile:
                    if isinstance(shapefile, dict):
                        ext = shapefile
                    else:
                        if isinstance(shapefile, Vector):
                            shp = shapefile.clone()
                        elif isinstance(shapefile, str):
                            shp = Vector(shapefile)
                        else:
                            raise TypeError("argument 'shapefile' must be either a dictionary, a Vector object or a string.")
                        # reproject the geometry to WGS 84 latlon
                        shp.reproject(4326)
                        ext = shp.extent
                        shp.close()
                    # add an extra buffer of 0.01 degrees
                    buffer = 0.01
                    ext['xmin'] -= buffer
                    ext['ymin'] -= buffer
                    ext['xmax'] += buffer
                    ext['ymax'] += buffer
                    with bbox(ext, 4326) as bounds:
                        inter = intersect(info_ms.bbox(), bounds)
                        if not inter:
                            raise RuntimeError('no bounding box intersection between shapefile and scene')
                        inter.close()
                        wkt = bounds.convert2wkt()[0]

                    subset = parse_node('Subset')
                    #subset.parameters['region'] = [0, 0, 0, 0]
                    subset.parameters['geoRegion'] = wkt
                    subset.parameters['copyMetadata'] = True
                    workflow_tpm.insert_node(subset, before=last_id)
                    last_id = subset.id
                
                ##multi looking for either one IW or multiple ones
                ml= parse_node("Multilook")
                ml.parameters["sourceBands"]=tpm_source
                ml.parameters["nRgLooks"]= ml_RgLook
                ml.parameters["nAzLooks"]= ml_AzLook
                ml.parameters["grSquarePixel"]= True
                ml.parameters["outputIntensity"]= False

                workflow_tpm.insert_node(ml,before= last_id)

                tc= parse_node("Terrain-Correction")
                tc.parameters["sourceBands"]= tpm_source
                tc.parameters["demName"]= demName
                tc.parameters["externalDEMFile"]= ext_Dem_file
                tc.parameters["externalDEMNoDataValue"]= ext_DEM_noDatVal
                tc.parameters["externalDEMApplyEGM"]= ext_DEM_EGM
                tc.parameters["demResamplingMethod"]= TC_demResamp
                tc.parameters["imgResamplingMethod"]= TC_demResamp
                tc.parameters["pixelSpacingInMeter"]= t_res
                tc.parameters["mapProjection"]= t_crs
                tc.parameters["saveSelectedSourceBand"]= True
                tc.parameters["outputComplex"]= False
                tc.parameters["nodataValueAtSea"]= msk_noDatVal

                workflow_tpm.insert_node(tc, before= ml.id)

                write_tpm=parse_node("Write")
                write_tpm.parameters["file"]= final_out_fp
                write_tpm.parameters["formatName"]= out_format

                workflow_tpm.insert_node(write_tpm, before= tc.id)

                ##write graph and execute graph
                workflow_tpm.write("Coh_TPM_continued_proc_graph")

                execute('Coh_TPM_continued_proc_graph.xml', gpt_args= gpt_paras)
        #exception for SNAP errors & creating error log     
        except RuntimeError as e:
            print(str(e))
            with open("S1_COH_proc_ERROR_"+datetime1+"_"+datetime2+".log", "w") as logf:
                logf.write(str(e))
            ##clean tmp folder to avoid overwriting errors even if exception is valid
            files = glob.glob(os.path.join(tmpdir, '*'))
            for f in files:
                if os.path.isfile(f) or os.path.islink(f):
                    os.unlink(f)
                elif os.path.isdir(f):
                    shutil.rmtree(f)
            
            continue
        
        ##clean tmp folder to avoid overwriting errors 
        files = glob.glob(os.path.join(tmpdir, '*'))
        for f in files:
            if os.path.isfile(f) or os.path.islink(f):
                os.unlink(f)
            elif os.path.isdir(f):
                shutil.rmtree(f)
    
    if clean_tmpdir == True:
        shutil.rmtree(tmpdir)

def S1_SLC_proc(data, maxdate = None, mindate = None , shapefile = None, int_proc = False, coh_proc= False, ha_proc= False, INT_Test= False, outdir_int= None, outdir_coh= None, outdir_ha= None, INT_test_dir= None, tmpdir= None, res_int= 20, res_coh= 20, res_ha= 20, t_crs= 4326, out_format= "GeoTIFF",\
                    gpt_paras= None, pol= 'full', iws= ["IW1", "IW2", "IW3"], ext_dem= False, ext_dem_nodatval= -9999, ext_dem_file= None, msk_nodatval= False, ext_dem_egm= True,\
                    decompfeats= ["Alpha", "Entropy", "Anisotropy"], ha_speckfilter= "Box Car Filter", decomp_win_size= 5, osvpath= None,\
                    imgresamp= "BICUBIC_INTERPOLATION", demresamp= "BILINEAR_INTERPOLATION", bgc_demresamp= "BICUBIC_INTERPOLATION", tc_demresamp= "BILINEAR_INTERPOLATION", \
                    cohwinrg= 11, cohwinaz= 3, speckfilter= "Boxcar", filtersizex= 5, filtersizey= 5, ml_rglook= 4, ml_azlook= 1,\
                    l2db_arg= True, firstburstindex= None, lastburstindex= None, ref_plain= "gamma",clean_tmpdir= True, osvfail= False, radnorm= False):
    
    site = Vector(shapefile)
    scenes = finder(data, [r'^S1[AB].*(SAFE|zip)$'],
                    regex=True, recursive=True, foldermode=1)
    
    database_path = f'{tmpdir}/scene.db'

    with Archive(dbfile= database_path) as archive:
        archive.insert(scenes)
        lst = archive.select(vectorobject=site,
                                   product='SLC', acquisition_mode='IW',
                                   mindate=mindate, maxdate=maxdate)
    
    slc_lst = []
    for slc in lst:
        bursts = get_burst_geometry(slc, target_subswaths = ['iw1', 'iw2', 'iw3'], polarization = 'vv')
        polygon = gpd.read_file(shapefile)
        inter = bursts.overlay(polygon, how='intersection')
        if not inter.empty:
            slc_lst.append(slc)

    
    print(f'Found {str(len(slc_lst))} scenes')
    if isinstance(slc_lst, str):
        ##handling one file being passed down
        grp_by_orb= [slc_lst]
    else:
        ##group files by orbit: ascending/descending
        grp_by_orb= group_by_info(slc_lst, group= "orbit")
        ##if only one orbit is detected
        if isinstance(grp_by_orb[0], str):
            grp_by_orb= [grp_by_orb]
    
    for orb in range(0,len(grp_by_orb)):
        ##handling one file being passed down
        if len(grp_by_orb) == 1 and isinstance(grp_by_orb[0], str):
            grp_by_relOrb= grp_by_orb
        else:
            ##group files by their relative orbit
            grp_by_relOrb= group_by_info(grp_by_orb[orb], group="orbitNumber_rel")
            ##if only one rel orbit is detected
            if isinstance(grp_by_relOrb[0], str):
                grp_by_relOrb= [grp_by_relOrb]      
        for ro in range(0, len(grp_by_relOrb)):
        ##selected options for features to be processed
            if int_proc == True:
                S1_INT_proc(infiles= grp_by_relOrb[ro], out_dir= outdir_int, shapefile=shapefile, t_res= res_int, tmpdir= tmpdir, t_crs= t_crs, out_format=out_format, gpt_paras= gpt_paras, pol=pol,\
                        IWs=iws, ext_DEM=ext_dem, ext_DEM_noDatVal= ext_dem_nodatval, ext_Dem_file= ext_dem_file, msk_noDatVal= msk_nodatval, ext_DEM_EGM= ext_dem_egm,\
                        imgResamp= imgresamp, demResamp=demresamp, speckFilter=speckfilter, osvPath= osvpath, ref_plain= ref_plain,\
                        filterSizeX= filtersizex, filterSizeY=filtersizey, ml_RgLook= ml_rglook, ml_AzLook=ml_azlook, l2dB_arg= l2db_arg,\
                        firstBurstIndex=firstburstindex, lastBurstIndex=lastburstindex, clean_tmpdir=clean_tmpdir, osvFail= osvfail)

            if coh_proc == True:
                S1_InSAR_coh_proc(infiles= grp_by_relOrb[ro], out_dir= outdir_coh, t_res= res_coh, tmpdir=tmpdir, t_crs= t_crs,  out_format=out_format, gpt_paras=gpt_paras,\
                                  pol= pol, IWs= iws, ext_DEM= ext_dem, ext_DEM_noDatVal=ext_dem_nodatval, ext_Dem_file=ext_dem_file, msk_noDatVal=msk_nodatval,\
                                  ext_DEM_EGM= ext_dem_egm, BGC_demResamp= bgc_demresamp, TC_demResamp= tc_demresamp, cohWinRg= cohwinrg, cohWinAz=cohwinaz, osvPath= osvpath,\
                                  ml_RgLook= ml_rglook, ml_AzLook= ml_azlook, firstBurstIndex= firstburstindex, lastBurstIndex= lastburstindex, clean_tmpdir=clean_tmpdir, osvFail= osvfail)
            if ha_proc == True:
                S1_HA_proc(infiles= grp_by_relOrb[ro], out_dir= outdir_ha, shapefile=shapefile, t_res= res_ha, tmpdir= tmpdir, t_crs= t_crs, out_format=out_format, gpt_paras= gpt_paras,\
                        IWs=iws, ext_DEM=ext_dem, ext_DEM_noDatVal= ext_dem_nodatval, ext_Dem_file= ext_dem_file, msk_noDatVal= msk_nodatval, ext_DEM_EGM= ext_dem_egm,\
                        imgResamp= imgresamp, demResamp=demresamp, speckFilter= ha_speckfilter, decomp_win_size= decomp_win_size, decompFeats=decompfeats,\
                        ml_RgLook= ml_rglook, ml_AzLook=ml_azlook,osvPath= osvpath, osvFail= osvfail,\
                        firstBurstIndex=firstburstindex, lastBurstIndex=lastburstindex, clean_tmpdir=clean_tmpdir)