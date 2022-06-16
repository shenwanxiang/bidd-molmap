#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 16:25:27 2021

@author: wanxiang.shen@u.nus.edu

main sequence featurizer code

"""
from molmap.utils.logtools import print_info, print_warn, print_error
from molmap.feature.sequence.aas.global_feature import Extraction as AASExtract
from molmap.feature.sequence.nas.global_feature import Extraction as NASExtract

from molmap.feature.sequence.aas.local_feature.aai import load_index, load_all
#from molmap.feature.sequence.nas import Extraction as NASext #change

from molmap import AggMolMap, show
from molmap._base import BaseMap

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE, MDS
from sklearn.utils import shuffle
from joblib import Parallel, delayed, load, dump
from umap import UMAP
from tqdm import tqdm
import pandas as pd
import numpy as np
import os



class GlobAASeqMolMap(BaseMap):
    
    def __init__(self, 
                 seq_list,
                 feature_para_dict ={'AAC12':{}, 'Autocorr':{}, 'CTD':{}, 
                                     'QSO':{"maxlag":30, "weight":0.1},
                                     'PAAC':{'lamda':30, "weight":0.05}},
                 metric = 'correlation',
                 info_distance = None,
                ):
        """
        Seqence MolMap Paramters
        -----------------
        seq_list: sequence list
        feature_para_dict: dict parameters for the corresponding feature type, say: {'PAAC':{'lamda':10}}
        metric: {'cosine', 'correlation', 'euclidean', 'jaccard', 'hamming', 'dice'}, default: 'correlation', measurement of feature distance
        info_distance: a vector-form distance vector of the feature points, shape should be: (n*(n-1)/2), where n is the number of the features
        """
        self.seq_list = seq_list
        self.seq_num = len(seq_list)
        self.feature_para_dict = feature_para_dict
        
        # init the feature extract object
        self.extract = AASExtract(feature_dict = feature_para_dict)   

        bitsinfodict = self.extract.bitsinfo.groupby('Subtypes').size().to_dict()
        feat_num = len(self.extract.bitsinfo)
        
        print_info('Total Sequences to calculate: %s, feature parameters: %s' % (self.seq_num, self.feature_para_dict))
        print_info('Total features to calculate: %s, details: %s' % (feat_num, bitsinfodict))

        ## extract Features
        X = self.extract.batch_transform(self.seq_list)
        dfx = pd.DataFrame(X, columns = self.extract.bitsinfo.IDs)

        ## init AggMap
        BaseMap.__init__(self, 
                           dfx, 
                           metric = metric,
                           info_distance = info_distance)
        self.dfx = dfx
        
    def fit(self, 
            cluster_channels = 0,
            var_thr = -1, 
            split_channels = True, 
            fmap_type = 'grid',  
            fmap_shape = None, 
            emb_method = 'umap', 
            min_dist = 0.1, 
            n_neighbors = 15,
            verbose = 2, 
            random_state = 32,
            group_color_dict  = {},
            lnk_method = 'complete',
            **kwargs): 
        """
        parameters
        -----------------
        cluster_channels: int, number of the channels(clusters) if False, use the defualt class instead of clustering
        var_thr: float, defalt is -1, meaning that feature will be included only if the conresponding variance larger than this value. Since some of the feature has pretty low variances, we can remove them by increasing this threshold
        split_channels: bool, if True, outputs will split into various channels using the types of feature
        fmap_type:{'scatter', 'grid'}, default: 'gird', if 'scatter', will return a scatter mol map without an assignment to a grid
        fmap_shape: None or tuple, size of molmap, only works when fmap_type is 'scatter', if None, the size of feature map will be calculated automatically
        emb_method: {'tsne', 'umap', 'mds'}, algorithm to embedd high-D to 2D
        group_color_dict: dict of the group colors, keys are the group names, values are the colors
        lnk_method: {'complete', 'average', 'single', 'weighted', 'centroid'}, linkage method
        kwargs: the extra parameters for the conresponding embedding method
        """
        if cluster_channels:
            feature_group_list = []
            group_color_dict = group_color_dict
        else:
            feature_group_list = self.extract.bitsinfo.Subtypes.tolist()
            group_color_dict = self.extract.colormaps
            
        self._fit(
            feature_group_list = feature_group_list,
            cluster_channels = cluster_channels,
            var_thr = var_thr, 
            split_channels = split_channels, 
            fmap_type = fmap_type,  
            fmap_shape = fmap_shape, 
            emb_method = emb_method, 
            min_dist = min_dist, 
            n_neighbors = n_neighbors,
            verbose = verbose, 
            random_state = random_state,
            group_color_dict  = group_color_dict,
            lnk_method = lnk_method,
            **kwargs)
        
        return self
  
    def transform(self, 
                  seq, 
                  scale = True, 
                  scale_method = 'minmax',):
    
        """
        parameters
        --------------------
        seq: string of AA sequence
        scale: bool, if True, we will apply MinMax scaling by the precomputed values
        scale_method: {'minmax', 'standard'}
        """
        arr_1d = self.extract.transform(seq)
        fmap = self._transform(arr_1d, 
                            scale = scale, 
                            scale_method = scale_method)
        return  fmap

    
    def batch_transform(self, 
                        seq_list, 
                        scale = True, 
                        scale_method = 'minmax',
                        n_jobs=4):
    
        """
        parameters
        --------------------
        seq_list: list of amino acid sequence strings
        scale: bool, if True, we will apply MinMax scaling by the precomputed values
        scale_method: {'minmax', 'standard'}
        n_jobs: number of parallel
        """
        
        array_2d = self.extract.batch_transform(seq_list)
        fmap = self._batch_transform(array_2d, 
                                    scale = scale, 
                                    scale_method = scale_method,
                                    n_jobs = n_jobs)
        
        return fmap


class LocalAASeqMolMap:
    '''
    get local properties
    '''
    def __init__(self):
        self.isfit = False
        self.aa_list = self.aaindex.data.columns.tolist()
        self.aa_list_pair = []
        for i in self.aa_list:
            for j in self.aa_list:
                self.aa_list_pair.append('%s,%s' % (i,j))        
                
        df = self.allindex.data
        #induces
        self.df_index1 = df[df.group == 'indices']
        self.index1 = self.df_index1.accession_number.unique().tolist()
        index1_subgroup = self.allindex.meta.set_index('accession_number').loc[self.index1]['subgroup'].to_dict()
        self.df_index1['subgroup'] = self.df_index1.accession_number.map(index1_subgroup)

        #mutaion matrices
        self.df_index2 = df[df.group == 'mutation matrices']
        self.index2 = self.df_index2.accession_number.unique().tolist()
        
        #contact potentials
        self.df_index3 = df[df.group == 'pair-wise contact potentials']
        self.index3 = self.df_index3.accession_number.unique().tolist()
        N = len(self.index1) + len(self.index2) + len(self.index3)
        print_info('INDEX-1: indices: %s' % len(self.index1))
        print_info('INDEX-2: mutation matrices: %s' % len(self.index2))
        print_info('INDEX-3: pair-wise contact potentials: %s' % len(self.index3))
        print_info('Total: %s' % N)
        
    @property
    def aaindex(self):    
        return load_index()

    @property
    def allindex(self):
        return load_all()
    
    def fit(self, aas):
        '''
        aas: animo acid sequence string
        '''
        res = []
        for i in aas:
            for j in aas:
                res.append('%s,%s' % (i,j))
        N = len(aas)
        self.N = N
        self.df2m = pd.DataFrame(index=res)
        self.df1m = pd.DataFrame(index = list(aas))
        self.isfit = True
        self.aas = aas
  
    def expand_aaindex(self, scale = True, feature_range = (0,1)):
        assert self.isfit, 'please fit first'
        for k, v in self.aaindex.data.T.to_dict().items():
            self.df1m[k] = self.df1m.index.map(v)   
        d1 = self.df1m.values
        if scale:
            scaler = MinMaxScaler(feature_range = feature_range)
            d1 = scaler.fit_transform(self.df1m)
        return d1 

    def get_matrices_aas_orders(self):
        assert self.isfit, 'please fit first'
        d = np.arange(self.N).reshape(self.N, 1) 
        d2 = np.log10(pairwise_distances(d, metric='l1') + 1)
        d2 = (d2 - d2.min())/((d2.max()-d2.min()))
        return d2
  
    def get_matrices_index_dist(self, index_key = 'ANDN920101'):
        '''
        index1
        '''
        assert self.isfit, 'please fit first'
        assert index_key in self.index1, 'index1 key only: %s' % self.index1
        data = self.aaindex.data.T[[index_key]]
        dist = pairwise_distances(data).reshape(-1, )
        ts = pd.Series(dist, index = self.aa_list_pair)
        ts = ts/ts.max()
        d = self.df2m.index.map(ts).values.reshape(self.N, self.N)
        return d

    def get_matrices_mutation(self, index_key = 'GONG920101'):
        '''
        index2
        '''
        assert self.isfit, 'please fit first'
        assert index_key in self.index2, 'index2 key only: %s' % self.index2
        
        ts = self.df_index2.set_index('accession_number').loc[index_key].set_index('animo_acid_key')['value']
        ts = (ts - ts.min())/((ts.max()-ts.min()))
        d = self.df2m.index.map(ts).values.reshape(self.N, self.N)
        return d

    def get_matrices_contact_potentials(self, index_key = 'BONM030106'):
        '''
        index3
        '''
        assert self.isfit, 'please fit first'
        assert index_key in self.index3, 'index3 key only: %s' % self.index3
        ts = self.df_index3.set_index('accession_number').loc[index_key].set_index('animo_acid_key')['value']
        ts = (ts - ts.min())/((ts.max()-ts.min()))
        d = self.df2m.index.map(ts).values.reshape(self.N, self.N)
        return d
    

            
class GlobNASeqMolMap(AggMolMap):
    pass