
from molmap.feature.fingerprint import Extraction as fext
from molmap.feature.descriptor import Extraction as dext


import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(ascii=True)



def calculate(npy_file_name, batch_cal_fuc, M, N, chunksize, n_jobs, dtype):
    
    datapath  = './data/cid_smiles.sample.csv'
    iterator = pd.read_csv(datapath, index_col = 'index', iterator=True, chunksize=chunksize)


    f = np.memmap(npy_file_name, 
                  dtype = dtype, mode='w+',
                  shape = (M, N))

    r = M // chunksize
    with tqdm(total = r) as pbar:
        start = 0
        for i,df in tqdm(enumerate(iterator), ascii=True):
            end = start + len(df)
            npy = batch_cal_fuc(df.smiles.tolist(), n_jobs = n_jobs)
            f[start:end, ] = npy
            f.flush()
            print(start,end)
            start = end
            pbar.update(1)
            
            
if __name__ == '__main__':
    
    F = fext()
    D = dext()
    
    M = 8506205
    
    
    Nd = len(D.bitsinfo)
    Nf = len(F.bitsinfo)

    chunksize = 50000
    n_jobs = 16
    
    print('calculating descriptors... ')
    calculate('/raid/shenwanxiang/descriptors_8506205.npy', D.batch_transform, M, Nd, chunksize,n_jobs, dtype=np.float)
    
    print('calculating fingerprint... ')
    calculate('/raid/shenwanxiang/fingerprint_8506205.npy', F.batch_transform, M, Nf, chunksize, n_jobs, dtype = np.bool)
    