from molmap.utils.logtools import print_info
import gdown
import pandas as pd
import os

fids = {'fingerprint_correlation.cfg.gzip': '1r-0F6ucqEoKfkpCinnRwIjREWU26KvEQ',
        'fingerprint_cosine.cfg.gzip': '14IkgTXbzMj7KutctIDnasLL4QDGhCiRV', 
        'fingerprint_jaccard.cfg.gzip': '1eBBrJDeCUlRX_FR6rMUXi4PLR956G9dO'}


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