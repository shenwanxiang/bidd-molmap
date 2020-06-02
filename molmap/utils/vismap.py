
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from scipy.spatial.distance import squareform
import seaborn as sns
from highcharts import Highchart
import pandas as pd
import numpy as np
import os

from molmap.utils.logtools import print_info


def plot_scatter(molmap, htmlpath = './', htmlname = None, radius = 3):
    '''
    molmap: the object of molmap
    htmlpath: the figure path, not include the prefix of 'html'
    htmlname: the name 
    radius: int, defaut:3, the radius of scatter dot
    '''
    
    title = '2D emmbedding of %s based on %s method' % (molmap.ftype, molmap.method)
    subtitle = 'number of %s: %s, metric method: %s' % (molmap.ftype, len(molmap.flist), molmap.metric)
    name = '%s_%s_%s_%s_%s' % (molmap.ftype,len(molmap.flist), molmap.metric, molmap.method, 'scatter')
    
    if not os.path.exists(htmlpath):
        os.makedirs(htmlpath)
    
    if htmlname:
        name = htmlname + '_' + name 
        
    filename = os.path.join(htmlpath, name)
    print_info('generate file: %s' % filename)
        
    
    xy = molmap.embedded.embedding_
    colormaps = molmap.extract.colormaps
    
    df = pd.DataFrame(xy, columns = ['x', 'y'])
    bitsinfo = molmap.extract.bitsinfo.set_index('IDs')
    df = df.join(bitsinfo.loc[molmap.flist].reset_index())
    df['colors'] = df['Subtypes'].map(colormaps)

    H = Highchart(width=1000, height=850)
    H.set_options('chart', {'type': 'scatter', 'zoomType': 'xy'})    
    H.set_options('title', {'text': title})
    H.set_options('subtitle', {'text': subtitle})
    H.set_options('xAxis', {'title': {'enabled': True,'text': 'X', 'style':{'fontSize':20}},
                           'labels':{'style':{'fontSize':20}}, 
                           'gridLineWidth': 1,
                           'startOnTick': True,
                           'endOnTick': True,
                           'showLastLabel': True})
    
    H.set_options('yAxis', {'title': {'text': 'Y', 'style':{'fontSize':20}},
                            'labels':{'style':{'fontSize':20}}, 
                            'gridLineWidth': 1,})
    
#     H.set_options('legend', {'layout': 'horizontal','verticalAlign': 'top','align':'right','floating': False,
#                              'backgroundColor': "(Highcharts.theme && Highcharts.theme.legendBackgroundColor) || '#FFFFFF'",
#                              'borderWidth': 1})
    
    
    H.set_options('legend', {'align': 'right', 'layout': 'vertical',
                             'margin': 1, 'verticalAlign': 'top', 'y':40,
                              'symbolHeight': 12, 'floating': False,})

    
    H.set_options('plotOptions', {'scatter': {'marker': {'radius': radius,
                                                         'states': {'hover': {'enabled': True,
                                                                              'lineColor': 'rgb(100,100,100)'}}},
                                              'states': {'hover': {'marker': {'enabled': False} }},
                                              'tooltip': {'headerFormat': '<b>{series.name}</b><br>',
                                                          'pointFormat': '{point.IDs}'}},
                                  'series': {'turboThreshold': 5000}})
    
    for subtype, color in colormaps.items():
        dfi = df[df['Subtypes'] == subtype]
        if len(dfi) == 0:
            continue
            
        data = dfi.to_dict('records')
        H.add_data_set(data, 'scatter', subtype, color=color)
    H.save_file(filename)
    print_info('save html file to %s' % filename)
    return df, H



def plot_grid(molmap, htmlpath = './', htmlname = None):
    '''
    molmap: the object of molmap
    htmlpath: the figure path
    '''    

    if not os.path.exists(htmlpath):
        os.makedirs(htmlpath)    
    
    title = 'Assignment of %s by %s emmbedding result' % (molmap.ftype, molmap.method)
    subtitle = 'number of %s: %s, metric method: %s' % (molmap.ftype, len(molmap.flist), molmap.metric)    

    name = '%s_%s_%s_%s_%s' % (molmap.ftype,len(molmap.flist), molmap.metric, molmap.method, 'molmap')
    
    if htmlname:
        name = name = htmlname + '_' + name   
    
    filename = os.path.join(htmlpath, name)
    print_info('generate file: %s' % filename)
    
    
    
    m,n = molmap.fmap_shape
    colormaps = molmap.extract.colormaps
    position = np.zeros(molmap.fmap_shape, dtype='O').reshape(m*n,)
    position[molmap._S.col_asses] = molmap.flist
    position = position.reshape(m, n)
    

    
    x = []
    for i in range(n):
        x.extend([i]*m)
        
    y = list(range(m))*n
        
        
    v = position.reshape(m*n, order = 'f')

    df = pd.DataFrame(list(zip(x,y, v)), columns = ['x', 'y', 'v'])
    bitsinfo = molmap.extract.bitsinfo
    subtypedict = bitsinfo.set_index('IDs')['Subtypes'].to_dict()
    subtypedict.update({0:'NaN'})
    df['Subtypes'] = df.v.map(subtypedict)
    df['colors'] = df['Subtypes'].map(colormaps) 

    
    H = Highchart(width=1000, height=850)
    H.set_options('chart', {'type': 'heatmap', 'zoomType': 'xy'})
    H.set_options('title', {'text': title})
    H.set_options('subtitle', {'text': subtitle})

#     H.set_options('xAxis', {'title': '', 
#                             'min': 0, 'max': molmap.fmap_shape[1]-1,
#                             'allowDecimals':False,
#                             'labels':{'style':{'fontSize':20}}})
    
#     H.set_options('yAxis', {'title': '', 'tickPosition': 'inside', 
#                             'min': 0, 'max': molmap.fmap_shape[0]-1,
#                             'reversed': True,
#                             'allowDecimals':False,
#                             'labels':{'style':{'fontSize':20}}})

    H.set_options('xAxis', {'title': None,                         
                            'min': 0, 'max': molmap.fmap_shape[1],
                            'startOnTick': False,
                            'endOnTick': False,    
                            'allowDecimals':False,
                            'labels':{'style':{'fontSize':20}}})

    
    H.set_options('yAxis', {'title': {'text': ' ', 'style':{'fontSize':20}}, 
                            'startOnTick': False,
                            'endOnTick': False,
                            'gridLineWidth': 0,
                            'reversed': True,
                            'min': 0, 'max': molmap.fmap_shape[0],
                            'allowDecimals':False,
                            'labels':{'style':{'fontSize':20}}})
    


    H.set_options('legend', {'align': 'right', 'layout': 'vertical',
                             'margin': 1, 'verticalAlign': 'top', 
                             'y': 60, 'symbolHeight': 12, 'floating': False,})

    
    H.set_options('tooltip', {'headerFormat': '<b>{series.name}</b><br>',
                              'pointFormat': '{point.v}'})

    
    H.set_options('plotOptions', {'series': {'turboThreshold': 5000}})
    
    for subtype, color in colormaps.items():
        dfi = df[df['Subtypes'] == subtype]
        if len(dfi) == 0:
            continue
        H.add_data_set(dfi.to_dict('records'), 'heatmap', 
                       name = subtype,
                       color = color,#dataLabels = {'enabled': True, 'color': '#000000'}
                      )
    H.save_file(filename)
    print_info('save html file to %s' % filename)
    
    return df, H






def _getNewick(node, newick, parentdist, leaf_names):
    if node.is_leaf():
        return "%s:%.2f%s" % (leaf_names[node.id], parentdist - node.dist, newick)
    else:
        if len(newick) > 0:
            newick = "):%.2f%s" % (parentdist - node.dist, newick)
        else:
            newick = ");"
        newick = _getNewick(node.get_left(), newick, node.dist, leaf_names)
        newick = _getNewick(node.get_right(), ",%s" % (newick), node.dist, leaf_names)
        newick = "(%s" % (newick)
        return newick
    
def _mp2newick(molmap, treefile = 'mytree'):

    dist_matrix = molmap.dist_matrix
    leaf_names = molmap.flist
    df = molmap.df_embedding[['colors','Subtypes']]
    
    dists = squareform(dist_matrix)
    linkage_matrix = linkage(dists, 'complete')
    tree = to_tree(linkage_matrix, rd=False)
    newick = getNewick(tree, "", tree.dist, leaf_names = leaf_names)
    
    with open(treefile + '.nwk', 'w') as f:
        f.write(newick)
    df.to_excel(treefile + '.xlsx')
    
        
def plot_tree(molmap, htmlpath = './', htmlname = None):
    pass