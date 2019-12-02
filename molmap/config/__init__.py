from molmap.utils.logtools import print_info
import gdown
import pandas as pd
import os

fids = {'fingerprint_correlation.cfg.gzip': '1E0QRu8e0_bqqnrC894GcbJHgfLR57QEo',
        'fingerprint_cosine.cfg.gzip': '1dHgpol9K7Wo72rs7g0UrToGTUaouoT8a', 
        'fingerprint_jaccard.cfg.gzip': '1EMYbw3wGguX-KSWjRGVhYZdwMdmisb2p'}



def load_config(ftype = 'descriptor', metric = 'cosine'):
    
    name = '%s_%s.cfg.gzip' % (ftype, metric)
    
    dirf = os.path.dirname(__file__)
    filename = os.path.join(dirf, name)
    
    if os.path.exists(filename):
        df = pd.read_pickle(filename, compression = 'gzip')
    else:
        print_info('downloading config file from google drive...')
        name = '%s_%s.cfg.gzip' % (ftype, metric)
        filename = os.path.join(dirf, name)
        fid = fids.get(name)
        url = "https://drive.google.com/uc?id=%s" % fid
        filename = gdown.download(url, filename, quiet = False)
        print_info('finished...')
        df = pd.read_pickle(filename,compression = 'gzip')
    return df