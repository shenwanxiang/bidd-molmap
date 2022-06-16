
# import pandas as pd
# from tqdm import tqdm
# tqdm.pandas(ascii=True)


# total = 138857329

# data_path = './data/CID-SMILES'
# df = pd.read_csv(data_path, header=None, sep = '\t',index_col=0)
# df = df.drop_duplicates(1)
# df.index.name = 'cid'
# df = df.rename(columns={1:'smiles'})
# df = df.reset_index()

# print(len(df))
# df.to_csv('./data/cid.smiles.nodupstring.96117890.csv')


import rdkit
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import warnings
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

tqdm.pandas(ascii=True)

rdkit.RDLogger.DisableLog('rdApp.*')
warnings.warn("deprecated", DeprecationWarning)


total = 138857329
data_path = './data/CID-SMILES'
df = pd.read_csv(data_path, header=None, sep = '\t',index_col=0)
df = df.drop_duplicates(1)
print(len(df))


def calcanonicalstrings(x):
    try:
        #res = Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=True)
        res =  Chem.MolToInchiKey(Chem.MolFromSmiles(x))
    except:
        res = None
    return res


def batch_cal(smiles,  n_jobs = 12):
    P = Parallel(n_jobs=n_jobs)
    res = P(delayed(calcanonicalstrings)(i) for i in tqdm(smiles,ascii = True))
    return res

df.index.name = 'cid'
df = df.rename(columns={1:'smiles'})


len(df)

canonicalstrings = batch_cal(df.smiles.tolist())

df['canonicalstrings'] = canonicalstrings

df = df.dropna()
df = df.drop_duplicates('canonicalstrings')
df = df['smiles']

print(len(df))  
#output: 110913349

df.to_csv('./data/cid.smiles.nodupstring.%s.csv' % len(df))