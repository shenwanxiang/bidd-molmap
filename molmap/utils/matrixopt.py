#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 20:29:36 2019

@author: wanxiang.shen@u.nus.edu

matrix operation

"""

import numpy as np
from lapjv import lapjv
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist


class Scatter2Grid:
    
    def __init__(self):  
        """assign x,y coords to gird numpy array"""
        self.fmap_shape = None
        self.indices = None
        self.indices_list = None

        
    def fit(self, df, split_channels = True, channel_col = 'Channels'):
        """
        parameters
        ------------------
        df: dataframe with x, y columns
        split_channels: bool, if True, will apply split by group
        channel_col: column in df.columns, split to groups by this col        
        
        """
        df['idx'] = range(len(df))
        
        embedding_2d = df[['x','y']].values
        N = len(df)

        size1 = int(np.ceil(np.sqrt(N)))
        size2 = int(np.ceil(N/size1))
        grid_size = (size1, size2)
        
        grid = np.dstack(np.meshgrid(np.linspace(0, 1, size2), 
                                     np.linspace(0, 1, size1))).reshape(-1, 2)
        grid_map = grid[:N]
        cost_matrix = cdist(grid_map, embedding_2d, "sqeuclidean").astype(np.float)
        cost_matrix = cost_matrix * (100000 / cost_matrix.max())
        row_asses, col_asses, _ = lapjv(cost_matrix)

        self.row_asses = row_asses
        self.col_asses = col_asses
        self.fmap_shape = grid_size
        self.indices = col_asses
        
        
        
        self.channel_col = channel_col
        self.split_channels = split_channels
        df['indices'] = self.indices
        self.df = df
        
        if self.split_channels:
            def _apply_split(x):
                return x[['idx', 'indices']].to_dict('list')
            sidx = df.groupby(channel_col).apply(_apply_split)      
            channels = sidx.index.tolist()
            indices_list = sidx.tolist()            
            self.channels = channels
            self.indices_list = indices_list

            
    def transform(self, vector_1d):
        """vector_1d: extracted features
        """             
        ### linear assignment map ###
        M, N = self.fmap_shape

        if self.split_channels:
            arr_res = []
            for idict in self.indices_list:

                indices = idict['indices']
                idx = idict['idx']

                arr = np.zeros(self.fmap_shape)
                arr_1d = arr.reshape(M*N, )
                arr_1d[indices] = vector_1d[idx]
                arr = arr_1d.reshape(M, N)  
                arr_res.append(arr) 
            arr_res = np.stack(arr_res, axis=-1)
        else:
            arr_res = np.zeros(self.fmap_shape)
            arr_1d = arr_res.reshape(M*N, )
            arr_1d[self.indices] = vector_1d
            arr_res = arr_1d.reshape(M, N, 1)          
        return arr_res
    

    
class Scatter2Array:
    
    def __init__(self, fmap_shape = (128,128)):  
        """convert x,y coords to numpy array"""
        self.fmap_shape = fmap_shape
        self.indices = None
        self.indices_list = None
        
    def _fit(self, df):
        """df: dataframe with x, y columns"""
        M, N = self.fmap_shape
        self.X = np.linspace(df.x.min(), df.x.max(), M)
        self.Y = np.linspace(df.y.min(), df.y.max(), N)

    
    def _transform(self, dfnew):
        """dfnew: dataframe with x, y columns
           in case we need to split channels
        """             
        x = dfnew.x.values
        y = dfnew.y.values
        M, N = self.fmap_shape
        indices = []
        for i in range(len(dfnew)):
            #perform a l1 distance
            idx = np.argmin(abs(self.X-x[i]))
            idy = np.argmin(abs(self.Y-y[i]))     
            indice = N*idy + idx
            indices.append(indice)
        return indices
    
    
    def fit(self, df, split_channels = True, channel_col = 'Channels'):
        """
        parameters
        ---------------
        df: embedding_df, dataframe
        split_channels: bool, if True, will apply split by group
        channel_col: column in df.columns, split to groups by this col
        """
        df['idx'] = range(len(df))
        self.df = df
        self.channel_col = channel_col
        self.split_channels = split_channels
        _ = self._fit(df)
        
        if self.split_channels:
            g = df.groupby(channel_col)
            sidx = g.apply(self._transform)            
            self.channels = sidx.index.tolist()
            self.indices_list = sidx.tolist()
        else:    
            self.indices = self._transform(df)
            
            
    def transform(self, vector_1d):
        """vector_1d: feature values 1d array"""
        
        M, N = self.fmap_shape
        arr = np.zeros(self.fmap_shape)
        arr_1d = arr.reshape(M*N, )
            
        if self.split_channels:
            df = self.df
            arr_res = []
            for indices, channel in zip(self.indices_list, self.channels):
                arr = np.zeros(self.fmap_shape)
                df1 = df[df[self.channel_col] == channel]
                idx = df1.idx.tolist()
                arr_1d_copy = arr_1d.copy()
                arr_1d_copy[indices] = vector_1d[idx]
                arr_1d_copy = arr_1d_copy.reshape(M, N) 
                arr_res.append(arr_1d_copy)
            arr_res = np.stack(arr_res, axis=-1)
        else:
            arr_1d_copy = arr_1d.copy()
            arr_1d_copy[self.indices] = vector_1d
            arr_res = arr_1d_copy.reshape(M, N, 1) 
        return arr_res


def smartpadding(array, target_size, mode='constant', constant_values=0):
    """
    array: 2d array to be padded
    target_size: tuple of target array's shape
    """
    X, Y = array.shape
    M, N = target_size
    top = int(np.ceil((M-X)/2))
    bottom = int(M - X - top)
    right = int(np.ceil((N-Y)/2))
    left = int(N - Y - right)
    array_pad = np.pad(array, pad_width=[(top, bottom),
                                         (left, right)], 
                       mode=mode, 
                       constant_values=constant_values)
    
    return array_pad


def fspecial_gauss(size = 31, sigma = 2):

    """Function to mimic the 'fspecial' gaussian MATLAB function
      size should be odd value
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def conv2(array, kernel_size = 31, sigma = 2,  mode='same', fillvalue = 0):
    kernel = fspecial_gauss(kernel_size, sigma)
    return np.rot90(convolve2d(np.rot90(array, 2), np.rot90(kernel, 2), 
                               mode=mode, 
                               fillvalue = fillvalue), 2)
