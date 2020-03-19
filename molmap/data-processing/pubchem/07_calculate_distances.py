import sys

from molmap.utils import distances, feature, calculator

import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
tqdm.pandas(ascii=True)


def loadnpy(filename, N, dtype, mode = 'r'):
    f = np.memmap(filename, mode = mode, 
                  dtype = dtype)
    M = int(len(f) / N)
    print(M, N)
    f = f.reshape(M, N)
    return f


def caldis(data, idx, tag, methods = ['correlation', 'cosine', 'jaccard']):
    
    
    ##############################################################
    ## to test one can use smaller number of compounds ##
    #############################################################
    
    
    for method in methods:
        res = calculator.pairwise_distance(data, n_cpus=16, method=method)
        res = np.nan_to_num(res,copy=False)
        df = pd.DataFrame(res,index=idx,columns=idx)
        df.to_pickle('./data/%s_%s.cfg' % (tag, method))





if __name__ == '__main__':
    
    #discriptors distance
    Nd = len(feature.descriptor.Extraction().bitsinfo)
    idx = feature.descriptor.Extraction().bitsinfo.IDs.tolist()
    data = loadnpy('./data/descriptors_8206960.npy', N = Nd, dtype = np.float)
    
    tag = 'descriptor'
    caldis(data, idx, tag, methods = ['correlation', 'cosine'])
    
    
#     #fingerprint distance
#     Nf = len(feature.fingerprint.Extraction().bitsinfo)
#     idx = feature.fingerprint.Extraction().bitsinfo.IDs.tolist()
#     data = loadnpy('./data/fingerprint_8206960.npy', N = Nf, dtype = np.bool)
#     tag = 'fingerprint'
#     caldis(data, idx, tag, methods = ['correlation', 'cosine', 'jaccard'])
