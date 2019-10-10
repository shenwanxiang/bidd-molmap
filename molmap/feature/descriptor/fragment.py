#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 18:04:35 2019

@author: charleshen

fragments number of 86 fragments: https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html

ref: https://github.com/rdkit/rdkit/blob/master/Data/FragmentDescriptors.csv
"""
from rdkit.Chem import Fragments
from collections import OrderedDict
from rdkit import Chem

def GetFragment(mol):
    """
    #################################################################
    Get the dictionary of Fragments descriptors for given moelcule mol
    
    Usage:
        
        result=GetFragments(mol)
        
        Input: mol is a molecule object.

        Output: result is a dict form containing all constitutional values.
    #################################################################
    """
    result=OrderedDict()
    for key, func in Fragments.fns:
        result[key]=func(mol)
    return result


_FragmentNames = list(GetFragment(Chem.MolFromSmiles('C')).keys())


if __name__ =='__main__':
    
    import pandas as pd
    from tqdm import tqdm
    from rdkit import Chem
    smis = ['C'*(i+1) for i in range(100)]
    x = []
    for index, smi in tqdm(enumerate(smis), ascii=True):
        m = Chem.MolFromSmiles(smi)
        x.append(GetFragment(m))
        
    pd.DataFrame(x)