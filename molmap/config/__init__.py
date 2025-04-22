from molmap.utils.logtools import print_info
import gdown
import pandas as pd
import os
import requests

# download from bidd
biddids = {
    'descriptor_correlation.cfg.gzip': 'http://bidd.group/download/molmap/descriptor_correlation.cfg.gzip',
    'descriptor_cosine.cfg.gzip': 'http://bidd.group/download/molmap/descriptor_cosine.cfg.gzip',
    'descriptor_scale.cfg.gzip': 'http://bidd.group/download/molmap/descriptor_scale.cfg.gzip',

    'fingerprint_correlation.cfg.gzip': 'http://bidd.group/download/molmap/fingerprint_correlation.cfg.gzip',
    'fingerprint_cosine.cfg.gzip': 'http://bidd.group/download/molmap/fingerprint_cosine.cfg.gzip',
    'fingerprint_scale.cfg.gzip': 'http://bidd.group/download/molmap/fingerprint_scale.cfg.gzip',
    'fingerprint_jaccard.cfg.gzip': 'http://bidd.group/download/molmap/fingerprint_jaccard.cfg.gzip'
}

# download from google drive (only ID is needed for gdown)
googleids = {
    'descriptor_correlation.cfg.gzip': '1yybtCK_WjLck6QA0GNs-dbdxphbdP_-P',
    'descriptor_cosine.cfg.gzip': '1WTUwSbMkvp8Q6j3e_kfWO8Bnq36DwGue',
    'descriptor_scale.cfg.gzip': '13fMh22lMInnYzyj0M8AX89OwqBdE_iky',

    'fingerprint_correlation.cfg.gzip': '1a1mClJvczFctZzTzEry__ij3LCM5avfj',
    'fingerprint_cosine.cfg.gzip': '1RrFq1JPJrKuJR3sENlWO27SpozwO03Uj',
    'fingerprint_scale.cfg.gzip': '1u3FvHukzcjp2152J9fV9pd1fWjD05kEp',
    'fingerprint_jaccard.cfg.gzip': '1BRAY0o7YN4DUQtEUBKKxXtKQn9fU0pe7',
}


def load_config(ftype='descriptor', metric='cosine'):
    name = f'{ftype}_{metric}.cfg.gzip'

    dirf = os.path.dirname(__file__)
    filepath = os.path.join(dirf, name)

    if os.path.exists(filepath):
        print_info(f"Found cached config: {filepath}")
        return pd.read_pickle(filepath, compression='gzip')

    # Try Google Drive first
    gdrive_id = googleids.get(name)
    if gdrive_id:
        try:
            print_info('Trying to download from Google Drive...')
            filepath = gdown.download(id=gdrive_id, output=filepath, quiet=False)
            print_info('Download from Google Drive successful.')
            return pd.read_pickle(filepath, compression='gzip')
        except Exception as e:
            print_info(f'Google Drive download failed: {e}')

    # Try fallback: download from bidd.group via requests
    bidd_url = biddids.get(name)
    if bidd_url:
        try:
            print_info(f'Trying to download from bidd.group: {bidd_url}')
            r = requests.get(bidd_url)
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(r.content)
            print_info('Download from bidd.group successful.')
            return pd.read_pickle(filepath, compression='gzip')
        except Exception as e:
            print_info(f'bidd.group download failed: {e}')

    raise FileNotFoundError(f"❌ Could not download config file for '{name}' from either Google Drive or bidd.group.")
