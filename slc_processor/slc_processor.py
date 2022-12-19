import glob
import os
import logging
from functions import S1_SLC_proc, asf_downloader, get_config

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

path_current_directory = os.path.dirname(__file__)
config_file = os.path.join(path_current_directory, 'config.ini')

general = get_config(config_file, 'General')
download = get_config(config_file, 'Download')
if download['download'] == True:
    download.pop('download', None)
    download.update(general)
    asf_downloader(**download)

proc = get_config(config_file, 'Processing')
proc.update(general)
S1_SLC_proc(**proc)
