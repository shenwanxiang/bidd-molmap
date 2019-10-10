
from molmap import feature
from molmap.utils import summary
import sys

import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
tqdm.pandas(ascii=True)

def savenpy(filename, data):
    f = np.memmap(filename, mode = 'w+', 
                  shape = data.shape, 
                  dtype = data.dtype)
    f[:] = data
    f.flush()
    del f
    

def loadnpy(filename, N, dtype):
    f = np.memmap(filename, mode = 'r', 
                  dtype = dtype)
    M = int(len(f) / N)
    print(M, N)
    f = f.reshape(M, N)
    return f

Nd = len(feature.descriptor.Extraction().bitsinfo)
Nf = len(feature.fingerprint.Extraction().bitsinfo)


Ddata = loadnpy('./data/descriptors_8206960.npy', N = Nd, dtype = np.float)

S = summary.Summary(n_jobs = 10)

res= []
for i in tqdm(range(Ddata.shape[1])):
    r = S._statistics_one(Ddata, i)
    res.append(r)
    
df = pd.DataFrame(res)
df.index = feature.descriptor.Extraction().bitsinfo.IDs
df.to_pickle('./data/descriptor_scale.cfg')

Fdata = loadnpy('./data/fingerprint_8206960.npy', N = Nf, dtype = np.bool)
res= []
for i in tqdm(range(Fdata.shape[1])):
    r = S._statistics_one(Fdata, i)
    res.append(r)
    
df = pd.DataFrame(res)
df.index = feature.fingerprint.Extraction().bitsinfo.IDs
df.to_pickle('./data/fingerprint_scale.cfg')