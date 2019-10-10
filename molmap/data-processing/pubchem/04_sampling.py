import numpy as np
import pandas as pd
import pqkmeans
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain
import seaborn as sns
sns.set()
tqdm.pandas(ascii=True)


def _sample_in_group(df):
    '''
    sample many groups(bins) with corresponding proportions
    '''
    if len(df) == 0:
        return []
    p = df.proportion.iloc[0]
    sdf = df.sample(frac = p, random_state = 42)
    if len(sdf) == 0:
        sdf = df.iloc[[0]]
    return list(sdf.index)



def sample(tsn, bins_2_smaple = 100, tag = 'sample'):
    
    '''
    sample training and test Set indexs
    
    tsn: Series for example:
            0    30
            1    29
            2    27
            ... ...
    '''
    bins = pd.cut(tsn,bins=bins_2_smaple, include_lowest=True)
    X1 = bins.value_counts()
    prop = X1/len(tsn)
    proportions = prop.to_dict()
    bins_df = bins.to_frame(name='bins')
    bins_df['proportion'] = bins_df.bins.map(proportions)
    df = tsn.to_frame(name = 'NumOnbits')
    df = df.join(bins_df)
    df.index.name = 'NpyFileIdx'
    samples_idx = df.groupby('bins')[['proportion']].progress_apply(_sample_in_group)
    samples_idx = list(chain(*samples_idx.tolist()))
    print('sample from %s to %s for %s' % (len(tsn), len(samples_idx), tag))
    
    df[tag] = df.index.isin(samples_idx)
    return df


if __name__ == '__main__':
    
    
    
    bitsize = 2048
    total_sample = 96117900
    bins_2_smaple = 100
    tsn_all = pd.read_pickle('./data/96117900_2048_NumOnBits.pkl')

    ## sample
    df_sample = sample(tsn_all, tag = 'sample')

    
    # save distribute figures
    fig, (ax1, ax2) = plt.subplots(2, sharex=True,  figsize=(12,10), gridspec_kw={'hspace': 0.05})

    alldata = df_sample['NumOnbits']
    ax1 = alldata.hist(bins = 100, edgecolor = 'black', ax = ax1, ylabelsize = 14, color = 'lightblue')
    ax1.text(410, 10000000, 'Overall: %s' % len(alldata), fontsize=16)
    ax1.set_ylabel('Number of compunds', fontsize=14)

    sampled = df_sample[df_sample['sample'] == True]['NumOnbits']
    ax2 = sampled.hist(bins = 100, edgecolor = 'black', ax = ax2, ylabelsize = 14,  color = 'lightblue')
    ax2.text(410, 1280000, 'Sample: %s' % len(sampled), fontsize=16)
    ax2.set_ylabel('Number of compunds', fontsize=14)
    fig.savefig('./data/sample_result.pdf')
    
    # save sampled cid & smiles
    sidx = sampled.index
    dfsmiles = pd.read_csv('./data/cid.smiles.nodupstring.96117900.csv', index_col = 0)
    
    s = dfsmiles.iloc[sidx]
    s.index.name = 'index'
    s.to_csv('./data/cid_smiles.sample.csv')