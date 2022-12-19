
import pyroSAR
from pyroSAR import identify_many
import glob
from os.path import join, basename
from datetime import datetime as dt
from datetime import timedelta
import itertools

"""[Reproc_by_ErrorLog]
    function to compile a list of files to be processed again based on error logs
    ----------
        dir_log: str
            directory where logs are stored
        fp_S1: list 
            filepaths of raw S1-data
        coh_dist: int
            if coherence was processed provide days of repetition rate (e.g. for S1A+B: 6)
        
        Returns
        -------
        nested list of filepaths for reprocesssing
        ----
    """

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
    info_lst= pyroSAR.identify_many(fp_S1, sortkey='start')
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