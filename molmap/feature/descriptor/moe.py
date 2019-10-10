#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 20:29:36 2019

@author: charleshen

@note: this code derived from PybioMed<https://github.com/gadsbyfly/PyBioMed>, with a major modified


This module mainly implements the calculation of MOE-type descriptors, which  include LabuteASA, TPSA, slogPVSA, MRVSA, PEOEVSA, EstateVSA and VSAEstate, respectively (60).

* 1 TPSA:  J. Med. Chem. 43:3714-7, (2000)
* 2 LabuteASA:  J. Mol. Graph. Mod. 18:464-77(2000)
* 3 PEOE_VSA1 - PEOE_VSA14:  MOE-type descriptors using partial charges and surface area contributions http://www.chemcomp.com/journal/vsadesc.htm
* 4 SMR_VSA1 - SMR_VSA10: MOE-type descriptors using MR contributions and surface area contributions http://www.chemcomp.com/journal/vsadesc.htm
* 5 SlogP_VSA1 - SlogP_VSA12: MOE-type descriptors using LogP contributions and surface area contributions http://www.chemcomp.com/journal/vsadesc.htm
* 6 EState_VSA1 - EState_VSA11: MOE-type descriptors using EState indices and surface area contributions
* 7 VSA_EState1 - VSA_EState10:  MOE-type descriptors using EState indices and surface area contributions 

"""

from mordred import Calculator, descriptors
import numpy as np

_calc = Calculator(descriptors.MoeType)
_MOENames = [str(i) for i in _calc.descriptors]


def GetMOE(mol):
    """
    #################################################################
    The calculation of MOE-type descriptors (ALL).
    
    Usage:
        
        result=GetMOE(mol)
        
        Input: mol is a molecule object
        
        Output: result is a dict form 
    #################################################################
    """
    r = _calc(mol)
    r = r.fill_missing(0)
    return r.asdict()


#########################################################################

if __name__=="__main__":
    
    
    smi5=['COCCCC','CCC(C)CC','CC(C)CCC','CC(C)C(C)C','CCOCCN','c1ccccc1N']
    smis = ['CCCC','CCCCC','CCCCCC','CC(N)C(=O)O','CC(N)C(=O)[O-].[Na+]']
    for index, smi in enumerate(smis):
        m = Chem.MolFromSmiles(smi)
        GetMOE(m)
        

