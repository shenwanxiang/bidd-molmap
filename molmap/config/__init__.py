from molmap.utils.logtools import print_info
import gdown
from mega import Mega
import pandas as pd
import os

#!pip install mega.py==1.0.8

# download from google drive
googleids = {'fingerprint_correlation.cfg.gzip': 'https://drive.google.com/uc?id=1E0QRu8e0_bqqnrC894GcbJHgfLR57QEo',
             'fingerprint_cosine.cfg.gzip': 'https://drive.google.com/uc?id=1dHgpol9K7Wo72rs7g0UrToGTUaouoT8a', 
             'fingerprint_jaccard.cfg.gzip': 'https://drive.google.com/uc?id=1EMYbw3wGguX-KSWjRGVhYZdwMdmisb2p'}




# download from mega drive
megaids = {'fingerprint_correlation.cfg.gzip': 'https://mega.nz/file/3uBziIzQ#QfvdlG1CRbuO-rADH6eqlJ361caqkvzq1_gyDDtLYmc',
           'fingerprint_cosine.cfg.gzip': 'https://mega.nz/file/S6IR0I4K#nk-vn3wm8makWoqg0VoNCcPSKpPxx_oq14JhW1hJHwg', 
           'fingerprint_jaccard.cfg.gzip': 'https://mega.nz/file/L7QTFIyQ#bnlWIcuAUBUB0EUmb8Y29vY3ExGNGTJTZeU-nl3OeM0'}







def load_config(ftype = 'descriptor', metric = 'cosine'):
    
    name = '%s_%s.cfg.gzip' % (ftype, metric)
    
    dirf = os.path.dirname(__file__)
    filename = os.path.join(dirf, name)
    
    if os.path.exists(filename):
        df = pd.read_pickle(filename, compression = 'gzip')
    else:
        
        name = '%s_%s.cfg.gzip' % (ftype, metric)
        filename = os.path.join(dirf, name)
        
        try:
            url = googleids.get(name)        
            print_info('downloading config file from google drive: %s' % url)
            filename = gdown.download(url, filename, quiet = False)
            print_info('finished...')
            
        except:
            url = megaids.get(name) 
            print_info('downloading config file from Mega drive: %s' % url)
            
            mega  = Mega()
            mega.download_url(url, dest_filename = filename)
            print_info('finished...')

        df = pd.read_pickle(filename,compression = 'gzip')
    return df