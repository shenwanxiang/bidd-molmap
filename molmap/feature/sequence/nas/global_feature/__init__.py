import pandas as pd
import numpy as np
from collections import OrderedDict
from joblib import Parallel, delayed
from tqdm import tqdm

from molmap.feature.sequence.nas.global_feature.nac import Kmer, RevcKmer, IDkmer
# kmer = nac.Kmer(k=5, normalize=True, upto=True) #4**5 + 4**4 + 4**3 + 4**2 + 4**1
# revkmer = nac.RevcKmer(k=5, normalize=True, upto=True)
# idkmer = nac.IDkmer(k=5, upto=True,)

from molmap.feature.sequence.nas.global_feature.ac import DAC, DCC, DACC, TAC, TCC, TACC
# lag = 5
# dac = ac.DAC(lag = lag)
# dcc = ac.DCC(lag = 1)
# #dacc = ac.DACC(lag = lag) #slow
# tac = ac.TAC(lag = lag)
# tcc = ac.TCC(lag = lag)
# tacc = ac.TACC(lag = lag)

from molmap.feature.sequence.nas.global_feature.psenac import PseDNC, PseKNC, PCPseDNC, PCPseTNC, SCPseDNC, SCPseTNC
# lamada = 3
# PseDNC = psenac.PseDNC(lamada=lamada, w=0.05)
# PseKNC = psenac.PseKNC(k=3, lamada=lamada, w=0.5)
# PCPseDNC = psenac.PCPseDNC(lamada=lamada, w=0.05)
# PCPseTNC = psenac.PCPseTNC(lamada=lamada, w=0.05)
# SCPseDNC = psenac.SCPseDNC(lamada=lamada, w=0.05)
# SCPseTNC = psenac.SCPseTNC(lamada=lamada, w=0.05)





def _GetNAC(ns, k = 5, normalize=True, upto=True):
    '''
    Get 1-mer, 2-mer, 3-mer, 2-3mer
    '''
    NACs = {}
    kmer = Kmer(k=k, normalize=normalize, upto=upto) #4**5 + 4**4 + 4**3 + 4**2 + 4**1
    revkmer = RevcKmer(k=k, normalize=normalize, upto=upto)
    
    kmer_res = dict(zip(kmer.feature_name_list, kmer.make_kmer_vec(ns)))
    rkmer_res = dict(zip(revkmer.feature_name_list, revkmer.make_kmer_vec(ns)))
    
    NACs.update(kmer_res)
    NACs.update(rkmer_res)
    
    return NACs



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
 

mapfunc = {   
           _GetNAC:'NACompostion',  
           _GetAC:'Autocorr',
           _GetPSENAC:'PseudoNACompostion'}

mapkey = dict(map(reversed, mapfunc.items()))

colormaps = {'NACompostion': '#00ff1b',
             'Autocorr': '#bfff00',
             'PseudoNACompostion': '#0033ff',
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
        
        self._PS = 'ATCTACAAGACCCCTCCTATCAAGGACTTCGGCGGCTTCAATTTCAGCCAGATTCTGCCC'
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