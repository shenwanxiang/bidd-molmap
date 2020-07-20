from molmap.utils.logtools import print_info
import gdown
import pandas as pd
import os

# download from google drive
googleids = {'fingerprint_correlation.cfg.gzip': 'https://drive.google.com/uc?id=1E0QRu8e0_bqqnrC894GcbJHgfLR57QEo',
             'fingerprint_cosine.cfg.gzip': 'https://drive.google.com/uc?id=1dHgpol9K7Wo72rs7g0UrToGTUaouoT8a', 
             'fingerprint_jaccard.cfg.gzip': 'https://drive.google.com/uc?id=1EMYbw3wGguX-KSWjRGVhYZdwMdmisb2p'}


# download from bidd
biddids =  {'fingerprint_correlation.cfg.gzip': 'http://bidd.group/download/fingerprint_correlation.cfg.gzip',
           'fingerprint_cosine.cfg.gzip': 'http://bidd.group/download/fingerprint_cosine.cfg.gzip',
           'fingerprint_jaccard.cfg.gzip': 'http://bidd.group/download/fingerprint_jaccard.cfg.gzip'}

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
            print('Max retries exceeded for Google Drive, will try down it from BIDD ...')
            url = biddids.get(name) 
            print_info('downloading config file from bidd website: %s' % url)
            filename = gdown.download(url, filename, quiet = False)
            print_info('finished...')

        df = pd.read_pickle(filename,compression = 'gzip')
    return df
