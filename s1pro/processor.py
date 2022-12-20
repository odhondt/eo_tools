import logging
from .download_ASF import asf_downloader
from .s1_slc_proc import S1_SLC_proc
from .auxil import get_config

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def process(config_file):
    general = get_config(config_file, 'General')
    download = get_config(config_file, 'Download')
    if download['download'] == True:
        download.pop('download', None)
        download.update(general)
        asf_downloader(**download)

    proc = get_config(config_file, 'Processing')
    proc.update(general)
    S1_SLC_proc(**proc)
