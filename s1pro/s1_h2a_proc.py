import pyroSAR
from pyroSAR.snap.auxil import parse_recipe, parse_node, gpt, execute
from pyroSAR import  identify_many
from spatialist.ancillary import finder
from spatialist import crsConvert, Vector, Raster
import os
import glob
import datetime
import geopandas as gpd
from spatialist import gdalwarp

from auxils import get_burst_geometry, remove

def S1_HA_proc(infiles, out_dir= None, tmpdir= None, shapefile = None, t_res=20, t_crs=32633,  out_format= "GeoTIFF", gpt_paras= None,\
                    IWs= ["IW1", "IW2", "IW3"], decompFeats= ["Alpha", "Entropy", "Anisotropy"], ext_DEM= False, ext_DEM_noDatVal= -9999, ext_Dem_file= None, msk_noDatVal= False,\
                    ext_DEM_EGM= True, imgResamp= "BICUBIC_INTERPOLATION", demResamp= "BILINEAR_INTERPOLATION",decomp_win_size= 5 ,\
                    speckFilter= "Box Car Filter", ml_RgLook= 4, ml_AzLook= 1, osvPath=None,\
                    tpm_format= "BEAM-DIMAP", clean_tmpdir= True, osvFail= False):
    
    """[S1_HA_proc]
    function for processing H-alpha features (Alpha, Entropy, Anisotropy) from S-1 SLC files in SNAP
    Parameters
    ----------
        infiles: list or str
            filepaths of SLC zip files
        out_dir: str or None
            output folder if None a default folder structure is provided: "INT/decompFeat/"
        tmpdir: str
            temporary dir for intermediate processing steps, its automatically created at cwd if none is provided
        t_res: int, float
            resolution in meters of final product, default is 20
        t_crs: int
            EPSG code of target coordinate system, default is 4326
        out_format: str
            format of final output, formats supported by SNAP, default is GeoTiff
        gpt_paras: none or list
            a list of additional arguments to be passed to the gpt call
        decompFeats: list of str
            containing H/a decompostion features: Alpha, Entropy and Anisotropy
        decomp_win_size: int
            size of moving window in H/a decomposition in pixel, default is 5
        IWs: str or list
            selected subswath for processing, default is all 3
        extDEM: bool
            set to true if external DEM should be used in processing
        ext_DEM_noDatVal: int or float
            dependent on external DEM, default False
        ext_DEM_file: str
            path to file of external DEM, must be a format that SNAP can handle
        msk_NoDatVal: bool
            if true No data values of DEM, especially at sea, are masked out
        ext_DEM_EGM: bool
            apply earth gravitational model to external DEM, default true
        imgResamp: str
            image resampling method, must be supported by SNAP
        demResamp: str
            DEM resampling method, must be supported by SNAP
        speckFilter: str
            type of speckle filtering approach, default is Box Car Filter
        ml_RgLook: int
            number of looks in range, default is 4
        ml_AzLook: int
            number of looks in azimuth, default is 1
        clean_tmpdir, bool
            delete tmpdir, default true
        osvPath: None
            specify path to locally stored OSVs, if none default OSV path of SNAP is set
        tpm_format: str
            specify the SNAP format for temporary files: "BEAM-DIMAP" or "ZNAP". "BEAM-DIMAP" default.
        Returns
        -------
        Raster files of selected output format for selected H-alpha features
        Note
        ----
        Only set first and last burstindex if all files you are processing have the same number of bursts
        Examples
        --------
        process all H-alpha features for given SLC file
        
        >>> filename= 'S1A_IW_GRDH_1SDV_20180829T170656_20180829T170721_023464_028DE0_F7BD.zip'
        >>> gpt_paras = ["-e", "-x", "-c","35G", "-q", "16", "-J-Xms25G", "-J-Xmx75G"]
        >>> decompFeats= ["Alpha", "Entropy", "Anisotropy"]
        >>> S1_HA_proc(infiles= filename, gtp_paras= gpt_paras, decompFeats= decompFeats)
    """

    ##define formatName for reading zip-files
    formatName= "SENTINEL-1"
    ##specify ending of tmp-files
    if tpm_format == "ZNAP":
        file_end = ".znap.zip"
    elif tpm_format == "BEAM-DIMAP":
        file_end = ".dim"
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
        info= [info]
    elif isinstance(infiles, list):
        info= pyroSAR.identify_many(infiles, sortkey='start')
        ##collect filepaths sorted by date
        fps_lst=[]
        for fp in info:
            fp_str=fp.scene
            fps_lst.append(fp_str)
        
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
            timea = datetime.datetime.now()
            slcAs_name= sensor +"_relOrb_"+ str(relOrb)+"_HA_"+unique_dates_info[i]+"_slcAs"
            slcAs_out= os.path.join(tmpdir, slcAs_name)
            graph_dir = f'{tmpdir}/graphs'
            isExist = os.path.exists(graph_dir)
            if not isExist:
                os.makedirs(graph_dir)
            ## create workflow for sliceAssembly if more than 1 file is available per date
            iw_bursts = None
            
            if len(fps_grp) > 1:

                workflow = parse_recipe("blank")

                read1 = parse_node('Read')
                read1.parameters['file'] = fps_grp[0]
                read1.parameters['formatName'] = formatName
                readers = [read1.id]

                workflow.insert_node(read1)
                
                if shapefile:
                    bursts = get_burst_geometry(fps_grp[0], target_subswaths= [ x.lower() for x in IWs], polarization = 'vv')
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
                    workflow.insert_node(readn, before= read1.id, resetSuccessorSource=False)
                    readers.append(readn.id)

                    if shapefile:
                        bursts = get_burst_geometry(fps_grp[r], target_subswaths = [ x.lower() for x in IWs], polarization = "vv")
                        polygon = gpd.read_file(shapefile)
                        inter = bursts.overlay(polygon, how='intersection')
                        iw_list = inter['subswath'].unique()
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

                workflow.insert_node(slcAs, before= readers)
                read1= slcAs

                write_slcAs=parse_node("Write")
                write_slcAs.parameters["file"]= slcAs_out
                write_slcAs.parameters["formatName"]= tpm_format

                workflow.insert_node(write_slcAs, before= slcAs.id)
                workflow.write(f"{graph_dir}/HA_slc_prep_graph")
                gpt(f"{graph_dir}/HA_slc_prep_graph.xml", gpt_args= gpt_paras, tmpdir = tmpdir)

                HA_proc_in= slcAs_out+ file_end
            ##pass file path if no sliceAssembly required
            else:
                HA_proc_in = fps_grp[0]
                if shapefile:
                    bursts = get_burst_geometry(fps_grp[0], target_subswaths = [ x.lower() for x in IWs], polarization = "vv")
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
                print(f'IW : {iw}')
                tpm_name= sensor +"_HA_relOrb_"+ str(relOrb) + "_"+\
                    unique_dates_info[i]+ "_"+iw+"_2TPM"
                tpm_out= os.path.join(tmpdir, tpm_name)
                ##generate workflow for IW splits 
                workflow= parse_recipe("blank")

                read= parse_node("Read")
                read.parameters["file"]= HA_proc_in
                workflow.insert_node(read)

                aof=parse_node("Apply-Orbit-File")
                aof.parameters["orbitType"]= orbitType
                aof.parameters["polyDegree"]= 3
                aof.parameters["continueOnFail"]= osvFail
                workflow.insert_node(aof, before= read.id)
                ##TOPSAR split node
                ts=parse_node("TOPSAR-Split")
                ts.parameters["subswath"]= iw 
                if shapefile:
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
            
            
            if tpm_format == "BEAM-DIMAP":
                    file_end= ".dim"
            elif tpm_format == "ZNAP":
                    file_end= ".znap.zip"

            for dc in decompFeats:          
                dc_label= dc.upper()[0:3]
                ##load temporary files
                tpm_in= glob.glob(tmpdir+"/"+sensor+"_HA_relOrb_"+ str(relOrb) + "_"+\
                        unique_dates_info[i]+ "*_2TPM"+file_end)
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

                out = sensor+"_"+ orbit+ "_relOrb_"+ str(relOrb) + "_HA_" + date_str + "_Orb_Cal_Deb_ML_Spk_TC"
                if shapefile is not None:
                    out_folder = f'{tmpdir}/{out}'
                else:
                    out_folder = f'{out_dir}/{out}'
                isExist = os.path.exists(out_folder)
                if not isExist:
                    os.makedirs(out_folder)
                
                out_name= sensor+"_"+ orbit+ "_relOrb_"+ str(relOrb) + "_HA_"+ dc_label + "_"+ date_str+"_Orb_Cal_Deb_ML_Spk_TC"
                out_path= os.path.join(out_folder, out_name) 

                write_tpm=parse_node("Write")
                write_tpm.parameters["file"]= out_path
                write_tpm.parameters["formatName"]= out_format
                workflow_tpm.insert_node(write_tpm, before= last_node)

                ##write graph and execute it
                workflow_tpm.write(f"{graph_dir}/HA_TPM_continued_proc_graph")
                execute(f"{graph_dir}/HA_TPM_continued_proc_graph.xml", gpt_args= gpt_paras)

                if shapefile is not None:
                    aoiname = os.path.splitext(os.path.basename(shapefile))[0]
                    out_folder = f'{out_dir}/{aoiname}/{out}'
                    isExist = os.path.exists(out_folder)
                    if not isExist:
                        os.makedirs(out_folder)
                
                    out_name = sensor+"_"+ orbit+ "_relOrb_"+ str(relOrb) + "_HA_" + dc_label + "_"+ date_str+"_Orb_Cal_Deb_ML_Spk_TC"
                    out_path_aoi= os.path.join(out_folder, out_name)
                    shp = Vector(shapefile)
                    shp.reproject(epsg)
                    extent = shp.extent
                    bounds = [extent['xmin'], extent['ymin'], extent['xmax'], extent['ymax']]
                    with Raster(f'{out_path}.tif', list_separate=False) as ras:
                        source = ras.filename
                    gdalwarp(src=source, dst=out_path_aoi,
                        options={'format': 'GTiff',
                        'outputBounds': bounds})
                   
            timeb =  datetime.datetime.now()
            proc_time = timeb - timea
            print(f'Processing time: {proc_time}')
            
        #exception for SNAP errors & creating error log    
        except RuntimeError as e:
            isExist = os.path.exists(f'{tmpdir}/error_logs')
            if not isExist:
                os.makedirs(f'{tmpdir}/error_logs')
            with open(f'{tmpdir}/error_logs/S1_HA_proc_ERROR_{date_str}.log', 'w') as logf:
                logf.write(str(e))
            
        ##clean tmp folder to avoid overwriting errors even if exception is valid
        if clean_tmpdir: 
            files = glob.glob(f'{tmpdir}/*.data') + glob.glob(f'{tmpdir}/*.dim')
            for fi in files:
                remove(fi)