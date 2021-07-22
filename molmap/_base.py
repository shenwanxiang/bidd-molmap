#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jul  1 13:49:20 2021

@author: AdminCOOP
"""


from molmap.utils.logtools import print_info, print_warn, print_error
from molmap.utils.matrixopt import Scatter2Grid, Scatter2Array, smartpadding 
from molmap.utils import summary, calculator
from molmap.utils import vismap2 as vismap


from sklearn.manifold import TSNE, MDS
from joblib import Parallel, delayed, load, dump
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
import matplotlib.pylab as plt
import seaborn as sns
from umap import UMAP
from tqdm import tqdm
from copy import copy
import pandas as pd
import numpy as np


class Base:
    
    def __init__(self):
        pass
        
    def _save(self, filename):
        return dump(self, filename)
        
    def _load(self, filename):
        return load(filename)

    def MinMaxScaleClip(self, x, xmin, xmax):
        scaled = (x - xmin) / ((xmax - xmin) + 1e-8)
        return scaled

    def StandardScaler(self, x, xmean, xstd):
        return (x-xmean) / (xstd + 1e-8) 
    


class BaseMap(Base):
    
    def __init__(self, 
                 dfx,
                 metric = 'correlation',
                 info_distance = None,
                ):
        
        """
        paramters
        -----------------
        dfx: pandas DataFrame
        metric: {'cosine', 'correlation', 'euclidean', 'jaccard', 'hamming', 'dice'}, default: 'correlation', measurement of feature distance
        info_distance: a vector-form distance vector of the feature points, shape should be: (n*(n-1)/2), where n is the number of the features
        
        """
        
        assert type(dfx) == pd.core.frame.DataFrame, 'input dfx mush be pandas DataFrame!'
        super().__init__()

        self.metric = metric
        self.isfit = False
        self.alist = dfx.columns.tolist()
        self.ftype = 'feature points'
        self.cluster_flag = False
        m,n = dfx.shape
        info_distance_length = int(n*(n-1)/2)
        
        ## calculating distance
        if np.array(info_distance).any():
            assert len(info_distance) == info_distance_length, 'shape of info_distance must be (%s,)' % info_distance_length
            print_info('skip to calculate the distance')
            self.info_distance = np.array(info_distance)

        else:
            print_info('Calculating distance ...')
            D = calculator.pairwise_distance(dfx.values, n_cpus=16, method=metric)
            D = np.nan_to_num(D,copy=False)
            D_ = squareform(D)
            self.info_distance = D_.clip(0, np.inf)
            
        ## statistic info
        S = summary.Summary(n_jobs = 10)
        res= []
        for i in tqdm(range(dfx.shape[1]), ascii=True):
            r = S._statistics_one(dfx.values, i)
            res.append(r)
        dfs = pd.DataFrame(res, index = self.alist)
        self.info_scale = dfs
        
        
        
    def _fit_embedding(self, 
                        dist_matrix,
                        method = 'umap',  
                        n_components = 2,
                        random_state = 32,  
                        verbose = 2,
                        n_neighbors = 15,
                        min_dist = 0.1,
                        **kwargs):
        
        """
        parameters
        -----------------
        dist_matrix: distance matrix to fit
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
                            n_neighbors = n_neighbors,
                            min_dist = min_dist,
                            verbose = verbose,
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
    
    
   
            

    def _fit(self, 
            feature_group_list = [],
            cluster_channels = 3,
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
        feature_group_list: list of the group name for each feature point
        cluster_channels: int, number of the channels(clusters) if feature_group_list is empty
        var_thr: float, defalt is -1, meaning that feature will be included only if the conresponding variance larger than this value. Since some of the feature has pretty low variances, we can remove them by increasing this threshold
        split_channels: bool, if True, outputs will split into various channels using the types of feature
        fmap_type:{'scatter', 'grid'}, default: 'gird', if 'scatter', will return a scatter mol map without an assignment to a grid
        fmap_shape: None or tuple, size of molmap, only works when fmap_type is 'scatter', if None, the size of feature map will be calculated automatically
        emb_method: {'tsne', 'umap', 'mds'}, algorithm to embedd high-D to 2D
        group_color_dict: dict of the group colors, keys are the group names, values are the colors
        lnk_method: {'complete', 'average', 'single', 'weighted', 'centroid'}, linkage method
        kwargs: the extra parameters for the conresponding embedding method
        """
            
        if 'n_components' in kwargs.keys():
            kwargs.pop('n_components')
            
            
        ## embedding  into a 2d 
        assert emb_method in ['tsne', 'umap', 'mds'], 'No Such Method Supported: %s' % emb_method
        assert fmap_type in ['scatter', 'grid'], 'No Such Feature Map Type Supported: %s'   % fmap_type     
        self.var_thr = var_thr
        self.split_channels = split_channels
        self.fmap_type = fmap_type
        self.fmap_shape = fmap_shape
        self.emb_method = emb_method
        self.lnk_method = lnk_method
        if fmap_shape != None:
            assert len(fmap_shape) == 2, "fmap_shape must be a tuple with two elements!"
        # flist and distance
        flist = self.info_scale[self.info_scale['var'] > self.var_thr].index.tolist()
        
        dfd = pd.DataFrame(squareform(self.info_distance),
                           index=self.alist,
                           columns=self.alist)
        dist_matrix = dfd.loc[flist][flist]
        self.flist = flist
        
        self.x_mean = self.info_scale['mean'].values
        self.x_std =  self.info_scale['std'].values
        
        self.x_min = self.info_scale['min'].values
        self.x_max = self.info_scale['max'].values
        
   
                
        #bitsinfo
        dfb = pd.DataFrame(self.alist, columns = ['IDs'])
        if feature_group_list != []:
            
            self.cluster_flag = False
            
            assert len(feature_group_list) == len(self.alist), "the length of the input group list is not equal to length of the feature list"
            self.cluster_channels = len(set(feature_group_list))
            self.feature_group_list = feature_group_list
            
            dfb['Subtypes'] = feature_group_list
            
            if set(feature_group_list).issubset(set(group_color_dict.keys())):
                self.group_color_dict = group_color_dict
                dfb['colors'] = dfb['Subtypes'].map(group_color_dict)
            else:
                unique_types = dfb['Subtypes'].unique()
                color_list = sns.color_palette("hsv", len(unique_types)).as_hex()
                group_color_dict = dict(zip(unique_types, color_list))
                dfb['colors'] = dfb['Subtypes'].map(group_color_dict)
                self.group_color_dict = group_color_dict
        else:
            
            self.cluster_channels = cluster_channels
            print_info('applying hierarchical clustering to obtain group information ...')
            self.cluster_flag = True
            
            Z = linkage(squareform(dfd.values),  lnk_method)
            labels = fcluster(Z, cluster_channels, criterion='maxclust')
            
            feature_group_list = ['cluster_%s' % str(i).zfill(2) for i in labels]
            dfb['Subtypes'] = feature_group_list
            dfb = dfb.sort_values('Subtypes')
            unique_types = dfb['Subtypes'].unique()
            
            if not set(unique_types).issubset(set(group_color_dict.keys())):
                color_list = sns.color_palette("hsv", len(unique_types)).as_hex()
                group_color_dict = dict(zip(unique_types, color_list))
            
            dfb['colors'] = dfb['Subtypes'].map(group_color_dict)
            self.group_color_dict = group_color_dict           
            self.Z = Z
            self.feature_group_list = feature_group_list
            

        self.bitsinfo = dfb
        colormaps = dfb.set_index('Subtypes')['colors'].to_dict()
        colormaps.update({'NaN': '#000000'})
        self.colormaps = colormaps
        
        if fmap_type == 'grid':
            S = Scatter2Grid()
        else:
            if fmap_shape == None:
                N = len(self.flist)
                l = np.int(np.sqrt(N))*2
                fmap_shape = (l, l)                
            S = Scatter2Array(fmap_shape)
        
        self._S = S

        ## 2d embedding first
        embedded = self._fit_embedding(dist_matrix,
                                       method = emb_method,
                                       n_neighbors = n_neighbors,
                                       random_state = random_state,
                                       min_dist = min_dist, 
                                       verbose = verbose,
                                       n_components = 2, **kwargs)
        
        self.embedded = embedded 
        
        df = pd.DataFrame(embedded.embedding_, index = self.flist,columns=['x', 'y'])
        typemap = self.bitsinfo.set_index('IDs')
        df = df.join(typemap)
        df['Channels'] = df['Subtypes']
        self.df_embedding = df
        self.channel_order = list(colormaps.keys())
        self.channel_order.remove('NaN')
        for i in list(set(self.channel_order) - set(self.df_embedding.Channels.unique())):
            self.channel_order.remove(i)
        
        if self.fmap_type == 'scatter':
            ## naive scatter algorithm
            print_info('Applying naive scatter feature map...')
            self._S.fit(self.df_embedding, self.split_channels, 
                        channel_col = 'Channels', channel_order=self.channel_order)
            print_info('Finished')
            
        else:
            ## linear assignment algorithm 
            print_info('Applying grid feature map(assignment), this may take several minutes(1~30 min)')
            self._S.fit(self.df_embedding, self.split_channels, 
                        channel_col = 'Channels', channel_order=self.channel_order)
            print_info('Finished')
        
        ## fit flag
        self.isfit = True
        if self.fmap_shape == None:
            self.fmap_shape = self._S.fmap_shape        
        
        else:
            m, n = self.fmap_shape
            p, q = self._S.fmap_shape
            assert (m >= p) & (n >=q), "fmap_shape's width must >= %s, height >= %s " % (p, q)
            
        return self
        

    def _transform(self, 
                  arr_1d, 
                  scale = True, 
                  scale_method = 'minmax',):
    
    
        """
        parameters
        --------------------
        arr_1d: 1d numpy array feature points
        scale: bool, if True, we will apply MinMax scaling by the precomputed values
        scale_method: {'minmax', 'standard'}
        """
        
        if not self.isfit:
            print_error('please fit first!')
            return

        if scale:
            if scale_method == 'standard':
                arr_1d = self.StandardScaler(arr_1d, self.x_mean, self.x_std)
            else:
                arr_1d = self.MinMaxScaleClip(arr_1d, self.x_min, self.x_max)
        
        df = pd.DataFrame(arr_1d).T
        df.columns = self.alist
        
        df = df[self.flist]
        vector_1d = df.values[0] #shape = (N, )
        fmap = self._S.transform(vector_1d)  
        p, q, c = fmap.shape
        
        if self.fmap_shape != None:        
            m, n = self.fmap_shape
            if (m > p) | (n > q):
                fps = []
                for i in range(c):
                    fp = smartpadding(fmap[:,:,i], self.fmap_shape)
                    fps.append(fp)
                fmap = np.stack(fps, axis=-1)
        return np.nan_to_num(fmap)   
    
    

    
    def _batch_transform(self, 
                        array_2d, 
                        scale = True, 
                        scale_method = 'minmax',
                        n_jobs=4):
    
        """
        parameters
        --------------------
        array_2d: 2D numpy array feature points, M(samples) x N(feature ponits)
        scale: bool, if True, we will apply MinMax scaling by the precomputed values
        scale_method: {'minmax', 'standard'}
        n_jobs: number of parallel
        """
        
        if not self.isfit:
            print_error('please fit first!')
            return
        
        assert type(array_2d) == np.ndarray, 'input must be numpy ndarray!' 
        assert array_2d.ndim == 2, 'input must be 2-D  numpy array!' 
        
        P = Parallel(n_jobs=n_jobs)
        res = P(delayed(self._transform)(arr_1d, 
                                        scale,
                                        scale_method) for arr_1d in tqdm(array_2d, ascii=True)) 
        X = np.stack(res) 
        
        return X
    
    
    def plot_scatter(self, htmlpath='./', htmlname=None, radius = 2, enabled_data_labels = False):
        """radius: the size of the scatter, must be int"""
        df_scatter, H_scatter = vismap.plot_scatter(self,  
                                                    htmlpath=htmlpath, 
                                                    htmlname=htmlname,
                                                    radius = radius,
                                                    enabled_data_labels = enabled_data_labels)
        
        self.df_scatter = df_scatter
        return H_scatter   
        
        
    def plot_grid(self, htmlpath='./', htmlname=None, enabled_data_labels = False):
        
        if self.fmap_type != 'grid':
            return
        
        df_grid, H_grid = vismap.plot_grid(self,
                                           htmlpath=htmlpath, 
                                           htmlname=htmlname,
                                           enabled_data_labels = enabled_data_labels)
        
        self.df_grid = df_grid
        return H_grid       
        
        
        
    def plot_tree(self, figsize=(16,8), add_leaf_labels = True, leaf_font_size = 18, leaf_rotation = 90):

        fig = plt.figure(figsize=figsize)
        
        if self.cluster_flag:
            
            Z = self.Z
            

            D_leaf_colors = self.bitsinfo['colors'].to_dict() 
            link_cols = {}
            for i, i12 in enumerate(Z[:,:2].astype(int)):
                c1, c2 = (link_cols[x] if x > len(Z) else D_leaf_colors[x] for x in i12)
                link_cols[i+1+len(Z)] = c1
            
            if add_leaf_labels:
                labels = self.alist
            else:
                labels = None
            P = dendrogram(Z, labels = labels, 
                          leaf_rotation = leaf_rotation, 
                          leaf_font_size = leaf_font_size, 
                          link_color_func=lambda x: link_cols[x])
        
        return fig
        
        
    def copy(self):
        return copy(self)
        
        
    def load(self, filename):
        return self._load(filename)
    
    
    def save(self, filename):
        return self._save(filename)