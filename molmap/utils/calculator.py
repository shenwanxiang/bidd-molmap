#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:54:12 2019

@author: wanxiang.shen@u.nus.edu

@usecase: calculate varies distances
"""
from molmap.utils.distances import named_distances
from molmap.utils.multiproc import MultiProcessUnorderedBarRun

import numpy as np
from tqdm import tqdm


def _yield_combinations(N):
    for i1 in range(N):
        for i2 in range(i1):
            yield (i1,i2)
            
def _calculate(i1, i2):
    x1 = data[:, i1]
    x2 = data[:, i2]
    ## dropna
    X = np.stack([x1,x2], axis=1)
    X = X[~np.isnan(X).any(axis=1)]
    x1 = X[:, 0]
    x2 = X[:, 1]
    
#     x1 = np.nan_to_num(x1)
#     x2 = np.nan_to_num(x2)
    if x1.any():
        dist = func(x1, x2)
    else:
        dist = np.nan
    return (i1, i2, dist)

def _fuc(x):
    i1, i2  = x
    return _calculate(i1, i2)


def pairwise_distance(npydata, n_cpus=8, method='correlation'):
    """
    parameters
    ---------------
    method: {'euclidean', 'manhattan', 'canberra', 'chebyshev', 
             'cosine', 'braycurtis', 'correlation',
             'jaccard', 'rogerstanimoto', 'hamming', 'dice', 'kulsinski', 'sokal_sneath'}
    npydata: np.array or np.memmap, Note that the default we will calcuate the vector's distances instead of sample's distances, if you wish to calculate distances between samples, you can pass data.T instead of data

    Usage
    --------------
    >>> import numpy as np
    >>> data = np.random.random_sample(size=(10000,10)
    >>> dist_matrix = pairwise_distance(data)
    >>> dist_matrix.shape
    >>> (10,10)  
    """    
    global data, func
    
    func = named_distances.get(method)
    data = npydata
    N = data.shape[1]
    lst = list(_yield_combinations(N))
    res = MultiProcessUnorderedBarRun(_fuc, lst, n_cpus=n_cpus)
    dist_matrix = np.zeros(shape = (N,N))
    for x,y,v in tqdm(res,ascii=True):
        dist_matrix[x,y] = v
        dist_matrix[y,x] = v
    return dist_matrix

    
    
if __name__ == '__main__':
    
    import numpy as np
    import pandas as pd
    from umap import UMAP
    import matplotlib.pyplot as plt
    
    X = np.random.random_sample(size=(1000000,40))
    distmatrix = pairwise_distance(X, n_cpus=6)
    embedding = UMAP(metric = 'precomputed',random_state=10)
    df = pd.DataFrame(embedding.fit_transform(distmatrix))
    ax = plt.plot(df[0],df[1], 'bo')