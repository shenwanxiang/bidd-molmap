## two types of PAAC: 100 = 50 + 50
from molmap.feature.sequence.aas.global_feature._PseudoAAC import GetPseudoAAC as GetPAACtype1
from molmap.feature.sequence.aas.global_feature._PseudoAAC import GetAPseudoAAC as GetPAACtype2

### AACs: 820 = 20 + 400 + 400
from molmap.feature.sequence.aas.global_feature._AAComposition import CalculateAAComposition as GetAAC1
from molmap.feature.sequence.aas.global_feature._AAComposition import CalculateDipeptideComposition as GetAAC2
from molmap.feature.sequence.aas.global_feature._AAComposition import Calculate2AACon3AA as GetAAC3AsAAC2
from molmap.feature.sequence.aas.global_feature._AAComposition import GetSpectrumDict as GetAAC3


### AutoCorr: 720 = 240 + 240 + 240
from molmap.feature.sequence.aas.global_feature._Autocorrelation import CalculateNormalizedMoreauBrotoAutoTotal as GetAutoCorrMoreauBroto
from molmap.feature.sequence.aas.global_feature._Autocorrelation import CalculateMoranAutoTotal as GetAutoCorrMoran
from molmap.feature.sequence.aas.global_feature._Autocorrelation import CalculateGearyAutoTotal as GetAutoCorrGeary


### CTDs: 147 = 21+21+105
from molmap.feature.sequence.aas.global_feature._CTD import CalculateC as GetCTD_C
from molmap.feature.sequence.aas.global_feature._CTD import CalculateT as GetCTD_T
from molmap.feature.sequence.aas.global_feature._CTD import CalculateD as GetCTD_D


### QuasiSequenceOrders: 160 = 60+100
from molmap.feature.sequence.aas.global_feature._QuasiSequenceOrder import GetSequenceOrderCouplingNumberTotal as GetSO_SNC
from molmap.feature.sequence.aas.global_feature._QuasiSequenceOrder import GetQuasiSequenceOrder as GetSO_QSO

import pandas as pd
import numpy as np
from collections import OrderedDict
from joblib import Parallel, delayed
from tqdm import tqdm



def _GetAAC12(ps):
    '''
    Get 1-mer, 2-mer, 3-mer, 2-3mer
    '''
    AACs = {}
    AACs.update(GetAAC1(ps))
    AACs.update(GetAAC2(ps))
    AACs.update(GetAAC3AsAAC2(ps))
    return AACs

def _GetAAC3(ps):
    return GetAAC3(ps)

def _GetAutoCorr(ps):
    AC = {}
    AC.update(GetAutoCorrMoreauBroto(ps))
    AC.update(GetAutoCorrMoran(ps))
    AC.update(GetAutoCorrGeary(ps))
    return AC

def _GetCTD(ps):
    CTD={}
    CTD.update(GetCTD_C(ps))
    CTD.update(GetCTD_T(ps))
    CTD.update(GetCTD_D(ps))
    return CTD    

def _GetQSO(ps,  maxlag=30, weight=0.1):
    QSO = {}
    QSO.update(GetSO_QSO(ps, maxlag = maxlag, weight = weight))
    QSO.update(GetSO_SNC(ps, maxlag = maxlag))
    return QSO 

def _GetPAAC(ps, lamda=30, weight=0.05, **args):
    '''
    lambda value should be smaller than length of the sequence
    '''
    n = len(ps)
    assert lamda < n, 'lamda value should be smaller than length of the sequence: length:%s, lamda:%s' % (n, lamda)
    PAAC = {}
    PAAC.update(GetPAACtype1(ps, lamda=lamda, weight=weight, **args))
    PAAC.update(GetPAACtype2(ps, lamda=lamda, weight=weight, **args))
    return PAAC
 
mapfunc = {_GetAAC12: 'AAC12', 
           _GetAAC3: 'AAC3', 
           _GetAutoCorr:'Autocorr',     
           _GetCTD:'CTD',  
           _GetQSO:'QSO',
           _GetPAAC:'PAAC'}

mapkey = dict(map(reversed, mapfunc.items()))

colormaps = {'AAC12': '#00ff1b',
             'AAC3': '#00ff86', 
             'Autocorr': '#bfff00',
             'CTD': '#ffd500',
             'QSO': '#ff0082',
             'PAAC': '#0033ff',
             'NaN': '#000000'}
        
# import seaborn as sns
# sns.palplot(colormaps.values())


class Extraction:
    
    
    def __init__(self,  feature_dict = {}):
        """        
        parameters
        -----------------------
        feature_dict: dict parameters for the corresponding fingerprint type, say: {'PAAC':{'lamda':10}}
        """
        if feature_dict == {}:
            factory = mapkey
            self.flag = 'all'
            cm = colormaps
        else:
            keys = [key for key in set(feature_dict.keys()) & set(mapkey)]
            factory = {}
            cm = {}
            for k, v in mapkey.items():
                if k in keys:
                    factory[k] = mapkey[k]
                    cm[k] = colormaps[k]
            self.flag = 'auto'
        assert factory != {}, 'types of feature %s can be used' % list(mapkey.keys())
            
        self.factory = factory
        self.feature_dict = feature_dict
        
        self._PS = 'MLMPKKNRIAIHELLFKEGVMVAKKDVHMPKHPELADKNVPNLHVMKAMQSLKSMLMMLM'
        _ = self._transform_ps(self._PS)
        self.colormaps = cm        

        
    def _transform_ps(self, ps):
        """
        ps: protein sequence
        """
        _all = []

        for key,func in self.factory.items():
            kwargs = self.feature_dict.get(key)
            
            if type(kwargs) == dict:
                res = func(ps, **kwargs)
            else:
                res = func(ps)
            df = pd.Series(res).to_frame(name = 'Value')
            df['Subtypes'] = key 
            df.index.name = 'IDs'
            _all.append(df)
        
        bitsinfo = pd.concat(_all).reset_index()
        bitsinfo['colors'] = bitsinfo.Subtypes.map(colormaps)
        self.bitsinfo = bitsinfo[['IDs', 'Subtypes', 'colors']] 
        return bitsinfo['Value'].values
  
    def transform(self, ps):
        '''
        ps: protein sequence
        '''
        try:
            arr = self._transform_ps(ps)
        except:
            #arr = np.nan * np.ones(shape=(len(self.bitsinfo), ))
            arr = np.zeros(shape=(len(self.bitsinfo), ))
            print('error when calculating %s' % ps)
        return arr
    
    
    def batch_transform(self, ps_list, n_jobs = 4):
        
        '''
        ps: list of the protein sequence
        '''
        
        P = Parallel(n_jobs=n_jobs)
        res = P(delayed(self.transform)(ps) for ps in tqdm(ps_list, ascii=True))
        return np.stack(res)
        
        
    
if __name__=="__main__":
    
    E = Extraction(feature_dict = {'AAC12':{}, 
                                   #'AAC3':{},
                                   'Autocorr':{}, 
                                   'CTD':{}, 
                                   'QSO':{"maxlag":30, "weight":0.1},
                                   'PAAC':{'lamda':30, "weight":0.05}})
    E.transform(E._PS)