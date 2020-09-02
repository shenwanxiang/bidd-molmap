import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import os

bitsize = 1024
total_sample = 110913349

data_save_folder = './data'

file = './data/%s_%s.npy' % (total_sample, bitsize)

f = np.memmap(file, dtype = np.bool,  shape = (total_sample, bitsize))


def _sum(memmap, x):
    return memmap[x, ].sum()

P = Parallel(n_jobs=16) #
res = P(delayed(_sum)(f, i) for i in tqdm(range(total_sample)))


pd.Series(res).to_pickle('./data/%s_%s_NumOnBits.pkl' % (total_sample, bitsize))


print('Done!')