#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 20:29:36 2019

@author: wanxiang.shen@u.nus.edu

main molmap code

"""

from molmap.feature.fingerprint import Extraction as fext
from molmap.feature.descriptor import Extraction as dext
from molmap.utils.logtools import print_info, print_warn, print_error
from molmap.utils.matrixopt import smartpadding, conv2
from molmap.utils.matrixopt import Scatter2Grid, Scatter2Array 

from molmap.config import load_config
from molmap.utils import vismap


from sklearn.manifold import TSNE, MDS
from joblib import Parallel, delayed, load, dump
from umap import UMAP
from tqdm import tqdm
import pandas as pd
import numpy as np
import os



class Base:
    
    def __init__(self):
        pass
        
    def _save(self, filename):
        return dump(self, filename)
        
    def _load(self, filename):
        return load(filename)
 

    def MinMaxScaleClip(self, x, xmin, xmax):
        scaled = (x - xmin) / ((xmax - xmin) + 1e-8)
        return scaled.clip(0, 1)

    def StandardScaler(self, x, xmean, xstd):
        return (x-xmean) / (xstd + 1e-8) 
    
    
    

class MolMap(Base):
    
    def __init__(self, 
                 ftype = 'descriptor',
                 flist = None, 
                 metric = 'cosine', 
                 fmap_type = 'grid', 
                 fmap_shape = None, 
                 split_channels = False,
                 var_thr = 1e-4, ):
        """
        paramters
        -----------------
        ftype: {'fingerprint', 'descriptor'}, feature type
        flist: feature list, if you want use some of the features instead of all features
        metric: {'cosine', 'correlation'}
        fmap_shape: size of molmap, only works when fmap_type is 'scatter'
        fmap_type:{'scatter', 'grid'}, default: 'gird', if 'scatter', will return a scatter mol map without an assignment to a grid
        split_channels: bool, only works if fmap_type is 'scatter', if True, outputs will split into diff. channels using the types of feature
        var_thr: float, defalt is 1e-4, meaning that feature will be included only if the conresponding variance larger than this value. Since some of the feature has pretty low variances, we can remove them by increasing this threshold
        """
        
        super().__init__()
        assert ftype in ['descriptor', 'fingerprint'], 'no such feature type supported!'        
        assert fmap_type in ['scatter', 'grid'], 'no such feature map type supported!'
       
        self.ftype = ftype
        self.metric = metric
        self.method = None
        self.isfit = False

        
        #default we will load the  precomputed matrix
        dist_matrix = load_config(ftype, metric)
        
        scale_info = load_config(ftype, 'scale')      
        scale_info = scale_info[scale_info['var'] > var_thr]
        
        idx = scale_info.index.tolist()
        dist_matrix = dist_matrix.loc[idx][idx]

        
        if flist:
            self.dist_matrix = dist_matrix.loc[flist][flist]
        else:
            self.dist_matrix = dist_matrix
        
        self.flist = list(self.dist_matrix.columns)
        self.scale_info = scale_info.loc[self.flist]
        
        #init the feature extract object
        if ftype == 'fingerprint':
            self.extract = fext()
        else:
            self.extract = dext() 

        self.fmap_type = fmap_type
        
        if fmap_type == 'grid':
            S = Scatter2Grid()
        else:
            if fmap_shape == None:
                N = len(self.flist)
                l = np.int(np.sqrt(N))*2
                fmap_shape = (l, l)                
            S = Scatter2Array(fmap_shape)
        
        self._S = S
        self.split_channels = split_channels        

    def _fit_embedding(self, 
                        method = 'tsne',  
                        n_components = 2,
                        random_state = 32,  
                        verbose = 2,
                        n_neighbors = 15,
                        min_dist = 0.9, 
                        **kwargs):
        
        """
        parameters
        -----------------
        method: {'tsne', 'umap', 'mds'}, algorithm to embedd high-D to 2D
        kwargs: the extra parameters for the conresponding algorithm
        """
        dist_matrix = self.dist_matrix
        if 'metric' in kwargs.keys():
            metric = kwargs.get('metric')
            kwargs.pop('metric')
            
        else:
            metric = 'precomputed'

        if method == 'tsne':
            embedded = TSNE(n_components=n_components, 
                            random_state=random_state,
                            metric = metric,
                            verbose = verbose,
                            **kwargs)
        elif method == 'umap':
            embedded = UMAP(n_components = n_components, 
                            verbose = verbose,
                            n_neighbors = n_neighbors,
                            min_dist = min_dist,
                            random_state=random_state, 
                            metric = metric, **kwargs)
            
        elif method =='mds':
            if 'metric' in kwargs.keys():
                kwargs.pop('metric')
            if 'dissimilarity' in kwargs.keys():
                dissimilarity = kwargs.get('dissimilarity')
                kwargs.pop('dissimilarity')
            else:
                dissimilarity = 'precomputed'
                
            embedded = MDS(metric = True, 
                           n_components= n_components,
                           verbose = verbose,
                           dissimilarity = dissimilarity, 
                           random_state = random_state, **kwargs)
        
        embedded = embedded.fit(dist_matrix)    

        df = pd.DataFrame(embedded.embedding_, index = self.flist,columns=['x', 'y'])
        typemap = self.extract.bitsinfo.set_index('IDs')
        df = df.join(typemap)
        df['Channels'] = df['Subtypes']
        self.df_embedding = df
        self.embedded = embedded
        

    def fit(self, method = 'umap', min_dist = 0.1, n_neighbors = 50,
            verbose = 2, random_state = 32, **kwargs): 
        """
        parameters
        -----------------
        method: {'tsne', 'umap', 'mds'}, algorithm to embedd high-D to 2D
        kwargs: the extra parameters for the conresponding method
        """
        if 'n_components' in kwargs.keys():
            kwargs.pop('n_components')
            
        ## embedding  into a 2d 
        assert method in ['tsne', 'umap', 'mds'], 'no support such method!'
        
        self.method = method
        
        ## 2d embedding first
        self._fit_embedding(method = method,
                            n_neighbors = n_neighbors,
                            random_state = random_state,
                            min_dist = min_dist, 
                            verbose = verbose,
                            n_components = 2, **kwargs)

        
        if self.fmap_type == 'scatter':
            ## naive scatter algorithm
            print_info('Applying naive scatter feature map...')
            self._S.fit(self.df_embedding, self.split_channels, channel_col = 'Channels')
            print_info('Finished')
            
        else:
            ## linear assignment algorithm 
            print_info('Applying grid feature map(assignment), this may take several minutes(1~30 min)')
            self._S.fit(self.df_embedding)
            print_info('Finished')
        
        ## fit flag
        self.isfit = True
        self.fmap_shape = self._S.fmap_shape
    

    
    def transform(self, smiles, 
                  fmap_shape = None, 
                  scale = True, 
                  scale_method = 'minmax',
                  smoothing = False, 
                  kernel_size = 31, 
                  sigma = 3, 
                  mode = 'same'):
    
    
        """
        parameters
        --------------------
        smiles: compund smile string
        
        scale: bool, if True, we will apply MinMax scaling by the precomputed values
        scale_method: {'minmax', 'standard'}

        smoothing: bool, if True, it will apply a gaussian smoothing
        kernel_size: size of the gaussian smoothing kernel, default is a size of (31,31)
        sigma: sigma of gaussian smoothing kernel
        mode: {'valid', 'same'}
        """
        
        if not self.isfit:
            print_error('please fit first!')
            return

        arr = self.extract.transform(smiles)
        df = pd.DataFrame(arr).T
        df.columns = self.extract.bitsinfo.IDs
        
        if (scale) & (self.ftype == 'descriptor'):
            
            if scale_method == 'standard':
                df = self.StandardScaler(df,  
                                    self.scale_info['mean'],
                                    self.scale_info['std'])
            else:
                df = self.MinMaxScaleClip(df, 
                                     self.scale_info['min'], 
                                     self.scale_info['max'])
        
        df = df[self.flist]
        vector_1d = df.values[0] #shape = (N, )
        fmap = self._S.transform(vector_1d)
        
        if smoothing & (~self.split_channels):
            fmap = conv2(fmap, kernel_size, sigma, mode)                
        return np.nan_to_num(fmap)   
        
        
        
        
    def batch_transform(self, smiles_list, 
                        n_jobs=4, 
                        fmap_shape = None, 
                        scale = True, 
                        scale_method = 'minmax',
                        smoothing = False,
                        kernel_size = 31,  
                        sigma = 3, 
                        mode = 'same'):
    
        """
        parameters
        --------------------
        smiles_list: list of smiles strings
        n_jobs: number of parallel
        fmap_shape: target shape of mol map, only work if naive_map is False
        scale: bool, if True, we will apply MinMax scaling by the precomputed values
        scale_method: {'minmax', 'standard'}

        smoothing: bool, if True, it will apply a gaussian smoothing
        kernel_size: size of the gaussian smoothing kernel, default is a size of (31,31)
        sigma: sigma of gaussian smoothing kernel
        mode: {'valid', 'same'}
        
        """
        
                    
        P = Parallel(n_jobs=n_jobs, )
        res = P(delayed(self.transform)(smiles, 
                                        fmap_shape , 
                                        scale,
                                        scale_method, 
                                        conv = False) for smiles in tqdm(smiles_list, ascii=True)) 
        
        ## not thread safe opt
        if smoothing & (~self.split_channels):
            res2 = []
            for fmap in tqdm(res,ascii=True):
                fmap = conv2(fmap, kernel_size, sigma, mode)   
                res2.append(fmap)
        else:
            res2 = res
        X = np.stack(res2) 
        
        return X
    

    def plot_scatter(self, htmlpath='./', htmlname=None):
        
        df_scatter, H_scatter = vismap.plot_scatter(self,  
                                htmlpath=htmlpath, 
                                htmlname=htmlname)
        
        self.df_scatter = df_scatter
        self.H_scatter = H_scatter
        
        
    def plot_grid(self, htmlpath='./', htmlname=None):
        
        if self.fmap_type != 'grid':
            return
        
        df_grid, H_gird = vismap.plot_grid(self,  
                                htmlpath=htmlpath, 
                                htmlname=htmlname)
        
        self.df_grid = df_grid
        self.H_gird = H_gird        
        
        
    def load(self, filename):
        return self._load(filename)
    
    
    def save(self, filename):
        return self._save(filename)
    