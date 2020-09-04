# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:01:00 2020

@author: SHEN WANXIANG
"""


import warnings
warnings.filterwarnings("ignore")


import matplotlib.pyplot as plt
import seaborn as sns





def imshow(x_arr,  ax, mode = 'dark',  
           color_list = ['#ff0c00','#25ff00', '#1300ff','#d000ff','#e2ff00',
                         '#00fff6', '#ff8800', '#fccde5','#178b66', '#8a0075'], 
           x_max = 255, vmin = -1, vmax = 1,):
    
    
    assert x_arr.ndim == 3, 'input must be 3d array!'
    w, h, c = x_arr.shape
    assert len(color_list) >= c, 'length of the color list should equal or larger than channel numbers'
    
    x = x_arr.copy()
    x[x == 0] = 'nan'

    xxx = x_arr.sum(axis=-1)
    xxx[xxx != 0] = 'nan'

    if mode == 'dark':
        cmaps = [sns.dark_palette(color, n_colors =  50, reverse=False) for color in color_list]

    else:
        cmaps = [sns.light_palette(color, n_colors =  50, reverse=False) for color in color_list]
        
    for i in range(c):
        data = x[:,:,i]/x_max
        sns.heatmap(data, cmap = cmaps[i],  vmin = vmin, vmax = vmax,  
                    yticklabels=False, xticklabels=False, cbar=False, ax=ax, ) # linewidths=0.005, linecolor = '0.9'

    if mode == 'dark':
        sns.heatmap(xxx, vmin=-10000, vmax=1, cmap = 'Greys', yticklabels=False, xticklabels=False, cbar=False, ax=ax)
    else:
        sns.heatmap(xxx, vmin=0, vmax=1, cmap = 'Greys', yticklabels=False, xticklabels=False, cbar=False, ax=ax)
        ax.axhline(y=0, color='grey',lw=2, ls =  '--')
        ax.axhline(y=data.shape[0], color='grey',lw=2, ls =  '--')
        ax.autoscale()
        ax.axvline(x=data.shape[1], color='grey',lw=2, ls =  '--')  
        ax.axvline(x=0, color='grey',lw=2, ls =  '--')


def imshow_wrap(x,  mode = 'dark', color_list = ['#ff0c00','#25ff00', '#1300ff','#d000ff','#e2ff00', 
                                                 '#00fff6', '#ff8800', '#fccde5','#178b66', '#8a0075'], 
                x_max = 255, vmin = -1, vmax = 1,):
    
    fig, ax = plt.subplots(figsize=(4,4))
    imshow(x.astype(float), mode = mode, color_list = color_list, ax=ax, x_max = x_max, vmin = vmin, vmax=vmax)