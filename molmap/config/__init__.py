from molmap.utils.logtools import print_info
import gdown
import pandas as pd
import os


# download from bidd
biddids =  {
            'descriptor_correlation.cfg.gzip': 'http://bidd.group/download/molmap/descriptor_correlation.cfg.gzip',
            'descriptor_cosine.cfg.gzip': 'http://bidd.group/download/molmap/descriptor_cosine.cfg.gzip',
            'descriptor_scale.cfg.gzip': 'http://bidd.group/download/molmap/descriptor_scale.cfg.gzip',    

            'fingerprint_correlation.cfg.gzip': 'http://bidd.group/download/molmap/fingerprint_correlation.cfg.gzip',
            'fingerprint_cosine.cfg.gzip': 'http://bidd.group/download/molmap/fingerprint_cosine.cfg.gzip',
            'fingerprint_scale.cfg.gzip': 'http://bidd.group/download/molmap/fingerprint_scale.cfg.gzip',
            'fingerprint_jaccard.cfg.gzip': 'http://bidd.group/download/molmap/fingerprint_jaccard.cfg.gzip'}



# download from google drive
googleids = {
            'descriptor_correlation.cfg.gzip': 'https://drive.google.com/uc?id=1yybtCK_WjLck6QA0GNs-dbdxphbdP_-P',
            'descriptor_cosine.cfg.gzip': 'https://drive.google.com/uc?id=1WTUwSbMkvp8Q6j3e_kfWO8Bnq36DwGue',
            'descriptor_scale.cfg.gzip': 'https://drive.google.com/uc?id=13fMh22lMInnYzyj0M8AX89OwqBdE_iky', 

            'fingerprint_correlation.cfg.gzip': 'https://drive.google.com/uc?id=1a1mClJvczFctZzTzEry__ij3LCM5avfj',
            'fingerprint_cosine.cfg.gzip': 'https://drive.google.com/uc?id=1RrFq1JPJrKuJR3sENlWO27SpozwO03Uj', 
            'fingerprint_scale.cfg.gzip': 'https://drive.google.com/uc?id=1u3FvHukzcjp2152J9fV9pd1fWjD05kEp',
            'fingerprint_jaccard.cfg.gzip': 'https://drive.google.com/uc?id=1BRAY0o7YN4DUQtEUBKKxXtKQn9fU0pe7',}



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
            print('try to down it from Google drive ...')
            url = googleids.get(name)        
            print_info('downloading config file from google drive: %s' % url)
            filename = gdown.download(url, filename, quiet = False)
            print_info('finished...')

        except:
            print('Max retries exceeded for Google Drive, will try to down it from bidd.group...')
            url = biddids.get(name) 
            print_info('downloading config file from bidd website: %s' % url)
            filename = gdown.download(url, filename, quiet = False)
            print_info('finished...')
            
        df = pd.read_pickle(filename,compression = 'gzip')
    return df
