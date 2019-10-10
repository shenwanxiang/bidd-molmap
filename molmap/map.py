#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 20:29:36 2019

@author: wanxiang.shen@u.nus.edu

core molmap code

"""

from molmap.feature.fingerprint import Extraction as fext
from molmap.feature.descriptor import Extraction as dext
from molmap.utils.logtools import print_info, print_warn, print_error
from molmap.utils.matrixopt import smartpadding, points2array, conv2
from molmap.config import load_config


from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE, MDS
from joblib import Parallel, delayed, load, dump
from lapjv import lapjv
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
    
    def __init__(self, flist = [], ftype = 'descriptor', metric = 'cosine', naive_map = False):
        """
        paramters
        -----------------
        dist_matrix: precomputed distance matrix, if None, we'll load the precomputed matrix
        ftype: {'fingerprint', 'descriptor'}, feature type
        flist: feature list, if you want use some of the features instead of all features
        metric: {'cosine', 'correlation'}
        naive_map: bool, if True, will return a naive mol map without an assignment to a grid

        """
        super().__init__()
        self.ftype = ftype
        self.metric = metric
        self.method = None
        self.isfit = False
        self.naive_map = naive_map
        #default we will load the  precomputed matrix
        
        dist_matrix = load_config(ftype, metric)
        
        if flist:
            self.dist_matrix = dist_matrix.loc[flist][flist]
        else:
            self.dist_matrix = dist_matrix
        
        self.flist = list(self.dist_matrix.columns)
        
        #init the feature extract object
        if ftype == 'fingerprint':
            self.extract = fext()
            sdf = load_config(ftype, 'scale')
            self.scale_info = sdf.loc[self.flist]

        else:
            self.extract = dext()
            sdf = load_config(ftype, 'scale')            
            self.scale_info = sdf.loc[self.flist]        
            
        
        

    def _fit_embedding(self, 
                        dist_matrix,
                        method = 'tsne',  
                        n_components = 2,
                        random_state = 42,  
                        verbose = 2,
                        n_neighbors = 15,
                        **kwargs):
        
        """
        parameters
        -----------------
        method: {'tsne', 'umap', 'mds'}, algorithm to embedd high-D to 2D
        kwargs: the extra parameters for the conresponding algorithm
        """
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

        return embedded
        
    
    
    def _fit_assignment(self, embedding_2d):
        """
        X_2d is a 2d embedding matrix by T-SNE or UMAP
        """
        N = len(embedding_2d)
        assert embedding_2d.shape[1] == 2

        size1 = int(np.ceil(np.sqrt(N)))
        size2 = int(np.ceil(N/size1))
        
        grid_size = (size1, size2)
        self.N = N
        self.grid_size = grid_size
        
        grid = np.dstack(np.meshgrid(np.linspace(0, 1, size2), np.linspace(0, 1, size1))).reshape(-1, 2)
        grid_map = grid[:N]
        cost_matrix = cdist(grid_map, embedding_2d, "sqeuclidean").astype(np.float)
        cost_matrix = cost_matrix * (100000 / cost_matrix.max())
        row_asses, col_asses, _ = lapjv(cost_matrix)
        self.row_asses = row_asses
        self.col_asses = col_asses
        return (row_asses, col_asses, grid_size, N)
    

    
    def fit(self, method = 'mds', random_state = 42,  verbose = 2, **kwargs): 
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
        
        self.embedded = self._fit_embedding(self.dist_matrix, 
                                            method = method,
                                            random_state = random_state,
                                            verbose = verbose,
                                            n_components = 2, **kwargs)
    
        
        if not self.naive_map:
            ## linear assignment algorithm 
            _ = self._fit_assignment(self.embedded.embedding_)
        
        ## fit flag
        self.isfit = True
        
        return self
    
    def _transform_naive(self, embedding_2d, target_size = None):
        
        N = len(embedding_2d)
        if target_size == None:
            target_size = (N,N)
        x = embedding_2d[:,0]
        y = embedding_2d[:,1]
        
        _, indices = points2array(x, y, target_size = target_size)
        self.naive_indices = indices
        return target_size, indices
        
        
        
    def transform(self, smiles, 
                  fmap_shape = None, 
                  scale = True, 
                  scale_method = 'minmax',
                  conv = False, 
                  kernel_size = 31, 
                  sigma = 3):
    
    
        """
        parameters
        --------------------
        smiles: compund smile string
        fmap_shape: target shape of mol map, only work if naive_map is False
        scale: bool, if True, we will apply MinMax scaling by the precomputed values
        scale_method: {'minmax', 'standard'}
        conv_naive: bool, if True, we will apply a convolution by using a gaussian kernel
        kernel_size: size of the gaussian kernel, default is a size of (31,31)
        sigma: sigma of gaussian kernel
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
        
        if self.naive_map:
            ### naive mol map ###
            size, indices = self._transform_naive(self.embedded.embedding_, 
                                                  fmap_shape)  
            
            M, N = size
            fmap = np.zeros(shape = (M*N, ))
            fmap[indices] = vector_1d
            fmap = fmap.reshape(M,N)
            self.fmap_shape = size
        else:
            ### linear assignment map ###
            x2 = vector_1d[self.row_asses]
            mend = self.grid_size[0]*self.grid_size[1] - self.N
            if mend > 0:
                x2 = np.append(x2, np.zeros(shape=(mend,)))
            fmap = x2.reshape(self.grid_size)  
            if fmap_shape != None:
                fmap = smartpadding(fmap, fmap_shape)
                self.fmap_shape = fmap_shape
            else:
                self.fmap_shape =  self.grid_size

        if conv:
            fmap = conv2(fmap, kernel_size, sigma)                
        return np.nan_to_num(fmap)   
        
        
        
        
    def batch_transform(self, smiles_list, 
                        n_jobs=4, 
                        fmap_shape = None, 
                        scale = True, 
                        scale_method = 'minmax',
                        conv = False,
                        kernel_size = 31,  
                        sigma = 3):
    
        """
        parameters
        --------------------
        smiles_list: list of smiles strings
        n_jobs: number of parallel
        fmap_shape: target shape of mol map, only work if naive_map is False
        scale: bool, if True, we will apply MinMax scaling by the precomputed values
        scale_method: {'minmax', 'standard'}
        conv_naive: bool, if True, we will apply a convolution by using a gaussian kernel
        kernel_size: size of the gaussian kernel, default is a size of (31,31)
        sigma: sigma of gaussian kernel
        """
        
                    
        P = Parallel(n_jobs=n_jobs, )
        res = P(delayed(self.transform)(smiles, 
                                        fmap_shape , 
                                        scale,
                                        scale_method, 
                                        conv = False) for smiles in tqdm(smiles_list, ascii=True)) 
        
        ## not thread safe opt
        if conv:
            res2 = []
            for fmap in tqdm(res,ascii=True):
                fmap = conv2(fmap, kernel_size, sigma)   
                res2.append(fmap)
        else:
            res2 = res
        X = np.stack(res2) 
        
        return X
    
    
    def load(self, filename):
        return self._load(filename)
    
    
    def save(self, filename):
        return self._save(filename)