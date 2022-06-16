import pandas as pd
import numpy as np
from collections import OrderedDict
from joblib import Parallel, delayed
from tqdm import tqdm

from molmap.feature.sequence.nas.global_feature.nac import Kmer, RevcKmer, IDkmer
# kmer = nac.Kmer(k=5, normalize=True, upto=True) # 1364
# revkmer = nac.RevcKmer(k=5, normalize=True, upto=True) #692
# idkmer = nac.IDkmer(k=5, upto=True,)

from molmap.feature.sequence.nas.global_feature.ac import DAC, DCC, DACC, TAC, TCC, TACC
# lag = 5
# dac = ac.DAC(lag = lag) # 190 #
# dcc = ac.DCC(lag = 1) #1406 #
# #dacc = ac.DACC(lag = lag) #slow
# tac = ac.TAC(lag = lag) #60
# tcc = ac.TCC(lag = lag) # 660 #
# tacc = ac.TACC(lag = lag) #720

ac_fdict = {'dac':DAC, 'dcc':DCC, 'dacc':DACC, 
 'tac':TAC, 'tcc':TCC, 'tacc':TACC}


from molmap.feature.sequence.nas.global_feature.psenac import PseDNC, PseKNC, PCPseDNC, PCPseTNC, SCPseDNC, SCPseTNC
# lamada = 3
# PseDNC = psenac.PseDNC(lamada=lamada, w=0.05)
# PseKNC = psenac.PseKNC(k=3, lamada=lamada, w=0.5)
# PCPseDNC = psenac.PCPseDNC(lamada=lamada, w=0.05)
# PCPseTNC = psenac.PCPseTNC(lamada=lamada, w=0.05)
# SCPseDNC = psenac.SCPseDNC(lamada=lamada, w=0.05)
# SCPseTNC = psenac.SCPseTNC(lamada=lamada, w=0.05)

psenac_fdict = {'psednc':PseDNC, 'pseknc':PseKNC, 
                'pcpdnc':PCPseDNC, 'pcptnc':PCPseTNC,
                'scpdnc':SCPseDNC, 'scptnc':SCPseTNC}



def _GetListKey(mylist, name):
    n = len(mylist)
    keys = ['%s_%s' % (name, i) for i in range(1, n+1)]
    return dict(zip(keys, mylist))


def _GetNAC(ns, k = 5, normalize=True, upto=True):
    '''
    if k=5, 1364+692 = 2056
    '''
    NACs = {}
    kmer = Kmer(k=k, normalize=normalize, upto=upto) #4**5 + 4**4 + 4**3 + 4**2 + 4**1
    revkmer = RevcKmer(k=k, normalize=normalize, upto=upto)
    
    kmer_res = dict(zip(kmer.feature_name_list, kmer.make_vec([ns])[0]))
    rkmer_res = dict(zip(revkmer.feature_name_list, revkmer.make_vec([ns])[0]))
    
    NACs.update(kmer_res)
    NACs.update(rkmer_res)
    
    return NACs


def _GetAC(ns, all_property=True, 
           phyche_index=None, extra_phyche_index=None,
           subset = {'dac': 3, 'dcc':1, 'tac':3,'tcc':3}):
    '''
    subset: {function_key, lag}, {'dac': 3, 'dcc':1, 'tac':3,'tcc':3, 'dacc':3, 'tacc':3}
    '''
    ACs = {}
    for fname, lag in subset.items():
        calc = ac_fdict[fname](lag=lag)
        vector = calc.make_vec([ns], 
                         phyche_index=phyche_index, 
                         all_property = all_property, 
                         extra_phyche_index=extra_phyche_index)[0]
        
        res = _GetListKey(vector, name = fname)
        ACs.update(res)
        
    return ACs


def _GetPSENAC(ns, lamada=3, w=0.05):
    '''
    lambda value should be smaller than length of the sequence
    '''

    PSENACs = {}
    for fname, clas in psenac_fdict.items():
        calc = clas(lamada=lamada, w=w)

        if fname in ['pcpdnc', 'pcptnc', 'scpdnc', 'scptnc']:
            vector = calc.make_vec([ns], all_property = True)[0]
        else:
            vector = calc.make_vec([ns])[0]
        res = _GetListKey(vector, name = fname)
        PSENACs.update(res)
    return PSENACs


mapfunc = {   
           _GetNAC:'NAC',  
           _GetAC:'AC',
           _GetPSENAC:'PNAC'}

mapkey = dict(map(reversed, mapfunc.items()))

colormaps = {'NAC': '#00ff1b',
             'AC': '#bfff00',
             'PNAC': '#0033ff',
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

    E = Extraction(feature_dict = {'NAC':{'k':5, 'normalize':True, 'upto':True}, 
                                   'AC':{'all_property':True, 'subset':{'dac': 3, 'tac':3}}, 
                                   'PNAC':{'lamada':3, "w":0.05}})
    E.transform(E._PS)