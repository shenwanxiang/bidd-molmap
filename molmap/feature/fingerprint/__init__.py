from .atompairs import GetAtomPairFPs
from .avalonfp import GetAvalonFPs
from .rdkitfp import GetRDkitFPs
from .morganfp import GetMorganFPs
from .estatefp import GetEstateFPs
from .maccskeys import GetMACCSFPs
from .pharmErGfp import GetPharmacoErGFPs
from .pharmPointfp import GetPharmacoPFPs
from .pubchemfp import GetPubChemFPs
from .torsions import GetTorsionFPs

from molmap.config import load_config

from rdkit import Chem
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


mapfunc = {GetAtomPairFPs:'AtomPairFP', 
           GetAvalonFPs:'AvalonFP', 
           GetRDkitFPs: 'RDkitFP', 
           GetMorganFPs:'MorganFP', 
           GetEstateFPs:'EstateFP', 
           GetMACCSFPs:'MACCSFP', 
           GetPharmacoErGFPs:'PharmacoErGFP', 
           GetPharmacoPFPs: 'PharmacoPFP', 
           GetPubChemFPs:'PubChemFP',  
           GetTorsionFPs:'TorsionFP'}



mapkey = dict(map(reversed, mapfunc.items()))


colormaps = {'AtomPairFP': '#ff8800',
             'AvalonFP': '#d4dd80',
             'RDkitFP': '#eeff00',
             'MorganFP': '#00ff27',
             'EstateFP': '#00ffaf',
             'MACCSFP': '#00c7ff',
             'PharmacoErGFP': '#003fff',
             'PharmacoPFP': '#4f00ff',
             'PubChemFP': '#d600ff',
             'TorsionFP': '#ff00a0',
             'NaN': '#000000'}



class Extraction:
    
    def __init__(self,  feature_dict = {}):
        """        
        parameters
        -----------------------
        feature_dict: dict parameters for the corresponding fingerprint type, say: {'AtomPairFP':{'nBits':2048}}
        """
        if feature_dict == {}:
            factory = mapkey
            self.flag = 'all'
        else:
            factory = {key:mapkey[key] for key in set(feature_dict.keys()) & set(mapkey)}
            self.flag = 'auto'
        assert factory != {}, 'types of feature %s can be used' % list(mapkey.keys())
            
        self.factory = factory
        self.feature_dict = feature_dict
        _ = self._transform_mol(Chem.MolFromSmiles('C'))
        self.colormaps = colormaps        
        self.scaleinfo = load_config('fingerprint', 'scale')
        
    def _transform_mol(self, mol):
        """
        mol" rdkit mol object
        """
        _all = []
        _length = []
        for key,func in self.factory.items():
            kwargs = self.feature_dict.get(key)
            
            if type(kwargs) == dict:
                arr = func(mol, **kwargs)
            else:
                arr = func(mol)
            _length.append(len(arr))
            _all.append(arr)

        concantefp = np.concatenate(_all)
        
        keys = []
        for key, length in zip(self.factory.keys(),  _length):
            keys.extend([(key+str(i), key) for i in range(length)])
            
        bitsinfo = pd.DataFrame(keys, columns=['IDs', 'Subtypes'])
        bitsinfo['colors'] = bitsinfo.Subtypes.map(colormaps)
        self.bitsinfo = bitsinfo            
        return concantefp

    
    def transform(self, smiles):
        '''
        smiles: smile string
        '''
        try:
            mol = Chem.MolFromSmiles(smiles)
            arr = self._transform_mol(mol)
        except:
            #arr = np.nan * np.ones(shape=(len(self.bitsinfo), ))
            arr = np.zeros(shape=(len(self.bitsinfo), ))
            print('error when calculating %s' % smiles)
            
        return arr
    
    
    def batch_transform(self, smiles_list, n_jobs = 4):
        P = Parallel(n_jobs=n_jobs)
        res = P(delayed(self.transform)(smiles) for smiles in tqdm(smiles_list, ascii=True))
        return np.stack(res)
        
        
    