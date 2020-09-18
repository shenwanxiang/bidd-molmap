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


## get info from : https://github.com/rdkit/rdkit/blob/d41752d558bf7200ab67b98cdd9e37f1bdd378de/Code/GraphMol/ReducedGraphs/ReducedGraphs.cpp
Donor = ["[N;!H0;v3,v4&+1]", "[O,S;H1;+0]", "[n&H1&+0]"]


Acceptor = ["[O,S;H1;v2;!$(*-*=[O,N,P,S])]",  "[O;H0;v2]", "[O,S;v1;-]",
            "[N;v3;!$(N-*=[O,N,P,S])]", "[n&H0&+0]", "[o;+0;!$([o]:n);!$([o]:c:n)]"]


Positive = ["[#7;+]", "[N;H2&+0][$([C,a]);!$([C,a](=O))]", 
            "[N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]", 
            "[N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))]"]

Negative = ["[C,S](=[O,S,P])-[O;H1,-1]"]

Hydrophobic = ["[C;D3,D4](-[CH3])-[CH3]", "[S;D2](-C)-C"]

Aromatic = ["a"]


PROPERTY_KEY = ["Donor", "Acceptor",  "Positive", "Negative",  "Hydrophobic",  "Aromatic"]


def GetPharmacoErGFPs(mol, fuzzIncrement = 0.3, maxPath = 21, binary = True, return_bitInfo = False):
    '''
    https://pubs.acs.org/doi/full/10.1021/ci050457y#
    return maxPath*21 bits
    
    size(v) = (n(n + 1)/2) * (maxDist - minDist + 1)

    '''
    minPath = 1
    
    arr  = AllChem.GetErGFingerprint(mol, fuzzIncrement=fuzzIncrement, maxPath = maxPath, minPath = minPath)
    arr = arr.astype(np.float32)
    
    if binary:
        arr = arr.astype(np.bool)

    if return_bitInfo:
        bitInfo = []
        for i in range(len(PROPERTY_KEY)):
            for j in range(i, len(PROPERTY_KEY)):
                for path in range(minPath, maxPath+1):
                    triplet = (PROPERTY_KEY[i], PROPERTY_KEY[j], path)
                    bitInfo.append(triplet)
        return arr, bitInfo

    return arr