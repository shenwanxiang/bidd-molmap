#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:54:12 2019

@author: wanxiang.shen@u.nus.edu

@usecase: statistic features' distribution
"""



import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


from molmap.utils.multiproc import MultiProcessUnorderedBarRun


class Summary:

    def __init__(self, n_jobs=1):
        
        '''
        n_jobs: number of paralleles
        '''
        self.n_jobs = n_jobs
        
        
    def _statistics_one(self, data, i):
        onefeat = data[:,i]
        
        
        onefeat = onefeat[~np.isnan(onefeat)]
        onefeat = onefeat[~np.isinf(onefeat)]
        
        s = pd.Series(onefeat)

        
        if len(s) != 0:
            maxv = s.max()
            minv = s.min()
            # using 0.8*(1-0.8) as a threshold to exclude feature
            var = s.var() 
            std = s.std()
            mean = s.mean()
            med = s.median()
            #skewness gt 0.75 will be log transformed
            skewness = s.skew()
            mode = s.mode().iloc[0]
        else:
            maxv = np.nan
            minv = np.nan
            var = np.nan
            std = np.nan
            mean = np.nan
            med = np.nan
            skewness = np.nan
            mode = np.nan
        
        del onefeat
        
        return {'index':i, 'max':maxv, 'mean':mean, 'min':minv, 'median':med, 
                'mode':mode, 'skewness':skewness, 'std':std, 'var': var}
    
    
    def fit(self, data, backend = 'threading', **kwargs):
        '''
        Parameters
        ----------
        data: np.memmap or np.array
        '''

        P = Parallel(n_jobs=self.n_jobs, backend = backend, **kwargs)
        res = P(delayed(self._statistics_one)(data,i) for i in tqdm(range(data.shape[1])))
        return pd.DataFrame(res)
    

def _func(i):
    S = Summary()
    return S._statistics_one(DATA, i)
    
    
def Summary2(data, n_jobs):
    global DATA, _func
    DATA = data
    res = MultiProcessUnorderedBarRun(_func, list(range(data.shape[1])), n_jobs)
    df = pd.DataFrame(res)
    dfres = df.sort_values('index').set_index('index')
    return dfres
