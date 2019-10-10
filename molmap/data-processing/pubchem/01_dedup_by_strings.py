
import pandas as pd
from tqdm import tqdm
tqdm.pandas(ascii=True)


total = 138857329

data_path = './data/CID-SMILES'
df = pd.read_csv(data_path, header=None, sep = '\t',index_col=0)
df = df.drop_duplicates(1)
df.index.name = 'cid'
df = df.rename(columns={1:'smiles'})
df = df.reset_index()

print(len(df))
df.to_csv('./data/cid.smiles.nodupstring.96117890.csv')