import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(ascii=True)
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from joblib import Parallel, delayed

def ecfp4(smile, bitsize = 2048):
    arr = np.zeros((bitsize,),  dtype=np.bool)
    try:
        mol = Chem.MolFromSmiles(smile)
        bitInfo={}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 
                                                   radius=4, 
                                                   bitInfo=bitInfo, 
                                                   nBits = bitsize)
    
        DataStructs.ConvertToNumpyArray(fp, arr)
    except:
        pass
    return arr    



def batch_cal(smiles, bitsize = 2048, n_jobs = 8):
    P = Parallel(n_jobs=n_jobs)
    res = P(delayed(ecfp4)(i, bitsize) for i in smiles)
    return np.stack(res)


if __name__ == '__main__':
    
    chunksize = 8192
    bitsize = 2048
    total_sample = 96117900
    n_jobs = 16
    
    f = np.memmap('./data/%s_%s.npy' % (total_sample, bitsize), 
                  dtype = np.bool, mode='w+',
                  shape = (total_sample, bitsize))

    iterator = pd.read_csv('./data/cid.smiles.nodupstring.96117900.csv', 
                             iterator=True, index_col = 0,
                             chunksize= chunksize)
    
    r = total_sample // chunksize
    
    with tqdm(total = r) as pbar:
        start = 0
        for i,df in tqdm(enumerate(iterator), ascii=True):
            end = start + len(df)
            npy = batch_cal(df.smiles.tolist(), bitsize = bitsize, n_jobs = n_jobs)
            f[start:end, ] = npy
            f.flush()
            
            print(start,end)
            start = end
            pbar.update(1)