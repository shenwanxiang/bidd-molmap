# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 11:58:46 2021

@author: wanxiang.shen@u.nus.edu
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from biopandas.pdb import PandasPdb
from Bio import PDB
import io, PIL
from sklearn.metrics import pairwise_distances

from molmap.agg import AggMolMap



#The hydrophobicity values are from JACS, 1962, 84: 4240-4246. (C. Tanford).
_Hydrophobicity={"A":0.62,"R":-2.53,"N":-0.78,"D":-0.90,"C":0.29,"Q":-0.85,
                 "E":-0.74,"G":0.48,"H":-0.40,"I":1.38,"L":1.06,"K":-1.50,
                 "M":0.64,"F":1.19,"P":0.12,"S":-0.18,"T":-0.05,"W":0.81,"Y":0.26,"V":1.08}

#The hydrophilicity values are from PNAS, 1981, 78:3824-3828 (T.P.Hopp & K.R.Woods).
_hydrophilicity={"A":-0.5,"R":3.0,"N":0.2,"D":3.0,"C":-1.0,"Q":0.2,"E":3.0,
                 "G":0.0,"H":-0.5,"I":-1.8,"L":-1.8,"K":3.0,"M":-1.3,
                 "F":-2.5,"P":0.0,"S":0.3,"T":-0.4,"W":-3.4,"Y":-2.3,"V":-1.5}

#The side-chain mass: CRC Handbook of Chemistry and Physics, 66th ed., CRC Press, Boca Raton, Florida (1985).

_residuemass={"A":15.0,"R":101.0,"N":58.0,"D":59.0,"C":47.0,"Q":72.0,
              "E":73.0,"G":1.000,"H":82.0,"I":57.0,"L":57.0,"K":73.0,
              "M":75.0,"F":91.0,"P":42.0,"S":31.0,"T":45.0,"W":130.0,"Y":107.0,"V":43.0}


#R.M.C. Dawson, D.C. Elliott, W.H. Elliott, K.M. Jones, Data for Biochemical Research 3rd ed., Clarendon Press Oxford (1986).
_pK1={"A":2.35,"C":1.71,"D":1.88,"E":2.19,"F":2.58,"G":2.34,"H":1.78,
      "I":2.32,"K":2.20,"L":2.36,"M":2.28,"N":2.18,"P":1.99,"Q":2.17,
      "R":2.18,"S":2.21,"T":2.15,"V":2.29,"W":2.38,"Y":2.20}

_pK2={"A":9.87,"C":10.78,"D":9.60,"E":9.67,"F":9.24,"G":9.60,"H":8.97,
      "I":9.76,"K":8.90,"L":9.60,"M":9.21,"N":9.09,"P":10.6,"Q":9.13,
      "R":9.09,"S":9.15,"T":9.12,"V":9.74,"W":9.39,"Y":9.11}

_pI={"A":6.11,"C":5.02,"D":2.98,"E":3.08,"F":5.91,"G":6.06,"H":7.64,
     "I":6.04,"K":9.47,"L":6.04,"M":5.74,"N":10.76,"P":6.30,
     "Q":5.65,"R":10.76,"S":5.68,"T":5.60,"V":6.02,"W":5.88,"Y":5.63}

_AvFlexibility={"A":0.357,"R":0.529,"N":0.463,"D":0.511,"C":0.346,"Q":0.493,
                "E":0.497,"G":0.544,"H":0.323,"I":0.462,"L":0.365,"K":0.466,"M":0.295,
                "F":0.314,"P":0.509,"S":0.507,"T":0.444,"W":0.305,"Y":0.420,"V":0.386}

_Polarizability={"A":0.046,"R":0.291,"N":0.134,"D":0.105,"C":0.128,"Q":0.180,
                 "E":0.151,"G":0.000,"H":0.230,"I":0.186,"L":0.186,"K":0.219,
                 "M":0.221,"F":0.290,"P":0.131,"S":0.062,"T":0.108,"W":0.409,"Y":0.298,"V":0.140}

_FreeEnergy={"A":-0.368,"R":-1.03,"N":0.0,"D":2.06,"C":4.53,"Q":0.731,
             "E":1.77,"G":-0.525,"H":0.0,"I":0.791,"L":1.07,"K":0.0,"M":0.656,
             "F":1.06,"P":-2.24,"S":-0.524,"T":0.0,"W":1.60,"Y":4.91,"V":0.401}

_ResidueASA={"A":115.0,"R":225.0,"N":160.0,"D":150.0,"C":135.0,"Q":180.0,
             "E":190.0,"G":75.0,"H":195.0,"I":175.0,"L":170.0,"K":200.0,"M":185.0,
             "F":210.0,"P":145.0,"S":115.0,"T":140.0,"W":255.0,"Y":230.0,"V":155.0}

_ResidueVol={"A":52.6,"R":109.1,"N":75.7,"D":68.4,"C":68.3,"Q":89.7,"E":84.7,
             "G":36.3,"H":91.9,"I":102.0,"L":102.0,"K":105.1,"M":97.7,"F":113.9,
             "P":73.6,"S":54.9,"T":71.2,"W":135.4,"Y":116.2,"V":85.1}

_Steric={"A":0.52,"R":0.68,"N":0.76,"D":0.76,"C":0.62,"Q":0.68,"E":0.68,
         "G":0.00,"H":0.70,"I":1.02,"L":0.98,"K":0.68,"M":0.78,"F":0.70,
         "P":0.36,"S":0.53,"T":0.50,"W":0.70,"Y":0.70,"V":0.76}

_Mutability={"A":100.0,"R":65.0,"N":134.0,"D":106.0,"C":20.0,"Q":93.0,"E":102.0,
             "G":49.0,"H":66.0,"I":96.0,"L":40.0,"K":-56.0,"M":94.0,"F":41.0,"P":56.0,
             "S":120.0,"T":97.0,"W":18.0,"Y":41.0,"V":74.0}

def standard_scale(aap):
    scaler = StandardScaler()
    s = pd.Series(aap)
    res = scaler.fit_transform(s.values.reshape(-1,1)).reshape(-1,)
    return pd.Series(res, index=s.index).to_dict()

IntrinsicAAPs = {'Hydrophobicity':_Hydrophobicity,
                'Hydrophilicity':_hydrophilicity,
                'ASA':_ResidueASA,
                'Flexibility': _AvFlexibility,
                'FreeEnergy': _FreeEnergy, 
                'Steric': _Steric,                  
                'pKa':_pK1,
                'pKb':_pK2,
                'pI':_pI}


class IntrinsicAAP:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_pdb_xyzb_ca(df_aa):
    
    '''
    https://en.wikipedia.org/wiki/Protein_contact_map:
    df_aa is the dataframe that is groupbyed from `residue_name` and  `residue_number`
    '''
    #df_aa = df[(df['residue_name'] == 'PRO') & (df['residue_number'] ==2)]

    ts = df_aa[df_aa.atom_name == 'CA'] 
    if len(ts == 1):
        ts = ts.iloc[-1] ## if multi-CA, select the last one
        x,y,z,b = ts.x_coord, ts.y_coord, ts.z_coord, ts.b_factor
    else:
        x,y,z,b = np.nan, np.nan, np.nan, np.nan
        aa = df_aa[['residue_name', 'residue_number']].iloc[0].to_dict()
        print('CA atom not exists in the residue: %s' % aa)  
    return x,y,z,b


def get_pdb_xyzb_cb(df_aa):
    '''
    https://en.wikipedia.org/wiki/Protein_contact_map:
    
    df_aa is the dataframe that is groupbyed from `residue_name` and  `residue_number`
    '''
    #df_aa = df[(df['residue_name'] == 'PRO') & (df['residue_number'] ==2)]
    
    ts = df_aa[df_aa.atom_name == 'CB'] ## if multi-CA, select the last one
    
    if len(ts) ==1:
        ts = ts.iloc[-1]
        x,y,z,b = ts.x_coord, ts.y_coord, ts.z_coord, ts.b_factor
    else:
        ##distance between Cβ-Cβ atoms with threshold 6-12 Å (Cα is used for Glycine):
        ts = df_aa[df_aa.atom_name == 'CA'] #Case Glycine
        if len(ts) ==1:
            ts = ts.iloc[-1]
            x,y,z,b = ts.x_coord, ts.y_coord, ts.z_coord, ts.b_factor
        else:
            x,y,z,b = np.nan, np.nan, np.nan, np.nan
            aa = df_aa[['residue_name', 'residue_number']].iloc[0].to_dict()
            print('CA atom not exists in the residue: %s' % aa)            
    return x,y,z,b



def get_pdb_xyzb_mean(df_aa):
    
    '''
    get the mean coord.
    https://en.wikipedia.org/wiki/Protein_contact_map:
    df_aa is the dataframe that is groupbyed from `residue_name` and  `residue_number`
    '''
    #df_aa = df[(df['residue_name'] == 'PRO') & (df['residue_number'] ==2)]

    x,y,z,b = df_aa[['x_coord', 'y_coord', 'z_coord', 'b_factor']].mean().tolist()
    
    return x,y,z,b


class PDB2Fmap:
    
    def __init__(self, embd_grain = 'CA', fmap_shape = None):
        
        '''
        embd_grain: {'CA', 'CB', 'mean', 'all'}
        '''
        self.embd_grain = embd_grain 
        self.fmap_shape = fmap_shape

        
        
    def fit(self, pdb_file, embd_chain = None):
        '''
        pdb_file: pdf file path
        embd_chain: pdb chain to do embedding
        '''
        self.pdb_file = pdb_file
        self.pdb = PandasPdb().read_pdb(self.pdb_file)
        self.embd_chain = embd_chain
        
        if embd_chain != None:
            self.dfpdb = self.pdb.df['ATOM'][self.pdb.df['ATOM'].chain_id == embd_chain]
        else:
            self.dfpdb = self.pdb.df['ATOM']
        if self.embd_grain == 'mean':
            df_embd = self.dfpdb.groupby(['residue_number', 'residue_name']).apply(get_pdb_xyzb_mean).apply(pd.Series)
            df_embd.columns = ['x_coord', 'y_coord', 'z_coord', 'b_factor']
            df_embd = df_embd.reset_index()

        if self.embd_grain == 'CB':
            df_embd = self.dfpdb.groupby(['residue_number', 'residue_name']).apply(get_pdb_xyzb_cb).apply(pd.Series) 
            df_embd.columns = ['x_coord', 'y_coord', 'z_coord', 'b_factor']
            df_embd = df_embd.reset_index()

        if self.embd_grain == 'CA':
            df_embd = self.dfpdb.groupby(['residue_number', 'residue_name']).apply(get_pdb_xyzb_ca).apply(pd.Series)  
            df_embd.columns = ['x_coord', 'y_coord', 'z_coord', 'b_factor']
            df_embd = df_embd.reset_index()
  
        if self.embd_grain == 'all':
            df_embd = self.dfpdb[['residue_name', 'residue_number', 'x_coord', 'y_coord','z_coord', 'b_factor']]
        
        df_embd['residue_name_1aa'] = df_embd['residue_name'].map(PDB.protein_letters_3to1)    
        df_embd.index = df_embd.index.astype(str) + '-' + df_embd['residue_name_1aa']
        dfx = df_embd[['x_coord','y_coord','z_coord']].T
        self.dfx = dfx
        self.df_embd = df_embd
        self.mp = AggMolMap(dfx, metric='euclidean')
        self.mp.fit(fmap_shape = self.fmap_shape, cluster_channels=1)
        self.fmap_shape = self.mp.fmap_shape
        
    def transform_xyz(self, scale = True, feature_range=(0, 1)):
        '''
        x, y, z coordinates
        '''
        if scale:
            scaler = MinMaxScaler(feature_range = feature_range)
            x = scaler.fit_transform(self.dfx.T).T
        else:
            x = self.dfx.values
        X = self.mp.batch_transform(x, scale=False)
        return X

    def transofrm_bf(self, scale = True, feature_range=(0, 1)):
        '''
        b-factor
        '''
        if scale:
            scaler = MinMaxScaler(feature_range = feature_range)
            x = scaler.fit_transform(self.df_embd[['b_factor']]).T
        else:
            x = self.df_embd[['b_factor']].values.T
        X = self.mp.transform(x[0], scale=False)
        return X


    def transofrm_pkt(self, pkt_file):
        '''
        pocket pdb file
        '''
        self.pkt_file = pkt_file

        ## pocket
        self.pkt = PandasPdb().read_pdb(self.pkt_file)
        if self.embd_chain != None:
            self.dfpkt = self.pkt.df['ATOM'][self.pkt.df['ATOM'].chain_id == self.embd_chain]
        else:
            self.dfpkt = self.pkt.df['ATOM']
            
        pkt_residue_number = self.dfpkt.residue_number.unique()

        self.df_embd['pocket'] = self.df_embd.residue_number.isin(pkt_residue_number)*1
        x = self.df_embd[['pocket']].values.T
        X = self.mp.transform(x[0], scale=False)
        return X

    def transform_custom(self, aap_df, scale = True, feature_range=(0, 1)):
        
        '''
        aap_df: dataframe of animo acid propetries, each column is one type of property, total 20 rows for all 20 types of animo acids
        aap_df example:
        ==============
        >>> from molmap.feature.sequence.aas.local_feature.aai import load_index
        >>> aap_df = load_index.data.T        
        '''

        df_custom = pd.DataFrame(index = self.df_embd.index)
        for k, v in aap_df.to_dict().items():
            df_custom[k] = self.df_embd.residue_name_1aa.map(v)
        self.df_custom = df_custom
        if scale:
            scaler = MinMaxScaler(feature_range = feature_range)
            x = scaler.fit_transform(self.df_custom).T
        else:
            x = self.df_custom.values.T
        X = self.mp.batch_transform(x, scale=False)       
        
        return X

    
    def transform_intrinsic(self, scale = True, feature_range=(0, 1)):
        
        df_intrinsic = pd.DataFrame(index = self.df_embd.index)
        for k, v in IntrinsicAAPs.items():
            df_intrinsic[k] = self.df_embd.residue_name_1aa.map(v)
        self.df_intrinsic = df_intrinsic
        if scale:
            scaler = MinMaxScaler(feature_range = feature_range)
            x = scaler.fit_transform(self.df_intrinsic).T
        else:
            x = self.df_intrinsic.values.T
        X = self.mp.batch_transform(x, scale=False)        

        return X



class PDB2Img:
    
    def __init__(self, pdb_file,  embd_grain = 'CA', embd_chain = None):
        
        '''
        embd_grain: {'CA', 'CB', 'mean', 'all'}
        pdb_file: pdf file path
        embd_chain: pdb chain to do embedding
        '''
        self.embd_grain = embd_grain 
        self.pdb_file = pdb_file
        self.pdb = PandasPdb().read_pdb(self.pdb_file)
        self.embd_chain = embd_chain
        
        if embd_chain != None:
            self.dfpdb = self.pdb.df['ATOM'][self.pdb.df['ATOM'].chain_id == embd_chain]
        else:
            self.dfpdb = self.pdb.df['ATOM']
        if self.embd_grain == 'mean':
            df_embd = self.dfpdb.groupby(['residue_number', 'residue_name']).apply(get_pdb_xyzb_mean).apply(pd.Series)
            df_embd.columns = ['x_coord', 'y_coord', 'z_coord', 'b_factor']
            df_embd = df_embd.reset_index()

        if self.embd_grain == 'CB':
            df_embd = self.dfpdb.groupby(['residue_number', 'residue_name']).apply(get_pdb_xyzb_cb).apply(pd.Series) 
            df_embd.columns = ['x_coord', 'y_coord', 'z_coord', 'b_factor']
            df_embd = df_embd.reset_index()

        if self.embd_grain == 'CA':
            df_embd = self.dfpdb.groupby(['residue_number', 'residue_name']).apply(get_pdb_xyzb_ca).apply(pd.Series)  
            df_embd.columns = ['x_coord', 'y_coord', 'z_coord', 'b_factor']
            df_embd = df_embd.reset_index()
  
        if self.embd_grain == 'all':
            df_embd = self.dfpdb[['residue_name', 'residue_number', 'x_coord', 'y_coord','z_coord', 'b_factor']]
        
        df_embd['residue_name_1aa'] = df_embd['residue_name'].map(PDB.protein_letters_3to1)    
        df_embd.index = df_embd.index.astype(str) + '-' + df_embd['residue_name_1aa']
        dfx = df_embd[['x_coord','y_coord','z_coord']]
        self.dfx = dfx
        self.df_embd = df_embd
        self.dfx_dist = pairwise_distances(self.dfx)

    def transform(self, fmap_shape = None, cmap='jet_r', vmin = 0, vmax=80, dpi=100):
        '''
        fig size: dpi*3
        '''
        fig, ax = plt.subplots()
        ax.imshow(self.dfx_dist, cmap=cmap, vmin = vmin, vmax=vmax)
        ax.axis('off')
        with io.BytesIO() as buff:
            fig.savefig(buff, bbox_inches='tight', pad_inches=0, dpi=dpi)
            buff.seek(0)
            im = PIL.Image.open(buff)
            im = im.convert('RGB')    
            if fmap_shape != None:
                im = im.resize(fmap_shape)
            x = np.array(im) / 255
            
        return fig, x
    
    
if __name__ == '__main__':
    pm = PDB2Fmap(embd_grain='all', fmap_shape=None)
    pm.fit(pdb_file='./1a1e/1a1e_protein.pdb', embd_chain='B')
    X = pm.transform_xyz(scale=True, feature_range=(0.1,1))
    X = pm.transofrm_bf(scale = True, feature_range=(0.2,1))
    X = pm.transofrm_pkt('./1a1e/1a1e_pocket.pdb')
    X = pm.transform_intrinsic()
    sns.heatmap(X[2].reshape(*pm.fmap_shape), cmap = 'jet')

    from molmap.feature.sequence.aas.local_feature.aai import load_index
    aaidx = load_index()
    dfindex = aaidx.data
    X = pm.transform_custom(dfindex.T)