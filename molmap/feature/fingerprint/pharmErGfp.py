#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:54:12 2019

@author: wanxiang.shen@u.nus.edu

@calculate ErG fps, more info: https://pubs.acs.org/doi/full/10.1021/ci050457y#
"""

_type = 'Pharmacophore-based'

import numpy as np
from rdkit.Chem import AllChem

def GetPharmacoErGFPs(mol, fuzzIncrement = 0.3, maxPath = 21, binary = True):
    '''
    https://pubs.acs.org/doi/full/10.1021/ci050457y#
    return maxPath*21 bits
    '''
    arr  = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath = 21)
    arr = arr.astype(np.float32)
    
    if binary:
        arr = arr.astype(np.bool)
    return arr
        