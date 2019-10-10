#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:54:12 2019

@author: wanxiang.shen@u.nus.edu

Combining a set of chemical features with the 2D (topological) distances between them gives a 2D pharmacophore. When the distances are binned, unique integer ids can be assigned to each of these pharmacophores and they can be stored in a fingerprint. Details of the encoding are in: https://www.rdkit.org/docs/RDKit_Book.html#ph4-figure
"""

_type = 'Pharmacophore-based'


#from fdef import featFactory
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem import DataStructs
from rdkit.Chem import ChemicalFeatures

import numpy as np
import os

fdef = os.path.join(os.path.dirname(__file__), 'mnimalfatures.fdef')
featFactory = ChemicalFeatures.BuildFeatureFactory(fdef)


def GetPharmacoPFPs(mol, 
                    bins = [(i,i+1) for i in range(20)], 
                    minPointCount = 2, 
                    maxPointCount = 2,
                    return_bitInfo = False):
    '''
    Note: maxPointCont with 3 is slowly
    
    bins = [(i,i+1) for i in range(20)], 
    maxPonitCount=2 for large-scale computation
    
    '''
    MysigFactory = SigFactory(featFactory,
                              trianglePruneBins=False,
                              minPointCount=minPointCount,
                              maxPointCount=maxPointCount)
    MysigFactory.SetBins(bins)
    MysigFactory.Init()
    
    res = Generate.Gen2DFingerprint(mol,MysigFactory)
    arr = np.array(list(res)).astype(np.bool)
    if return_bitInfo:
        description = []
        for i in range(len(res)):
            description.append(MysigFactory.GetBitDescription(i))
        return arr, description

    return arr
    
    
if __name__ == '__main__':
    
    from rdkit import Chem
    mol = Chem.MolFromSmiles('CC#CC(=O)NC1=NC=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl')
    a = GetPharmacoPFPs(mol, bins = [(i,i+1) for i in range(20)], minPointCount = 2, maxPointCount =2)
    