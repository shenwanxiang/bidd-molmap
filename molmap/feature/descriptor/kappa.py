#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 18:04:35 2019

@author: charleshen

@note: this code derived from PybioMed<https://github.com/gadsbyfly/PyBioMed>, with a major modified


"""


from rdkit import Chem

def CalculateKappaAlapha1(mol):
    """
    #################################################################
    Calculation of molecular shape index for one bonded fragment 
    
    with Alapha
    
    ---->kappam1
    
    Usage:
        
        result=CalculateKappaAlapha1(mol)
        
        Input: mol is a molecule object.
        
        Output: result is a numeric value.
    #################################################################
    """
    P1=mol.GetNumBonds()
    A=mol.GetNumAtoms()
    alpha=Chem.GraphDescriptors.HallKierAlpha(mol)
    denom=P1+alpha
    if denom:
        kappa=(A+alpha)*(A+alpha-1)**2/denom**2
    else:
        kappa=0.0
    return round(kappa,3)


def CalculateKappaAlapha2(mol):
    """
    #################################################################
    Calculation of molecular shape index for two bonded fragment 
    
    with Alapha
    
    ---->kappam2
    
    Usage:
        
        result=CalculateKappaAlapha2(mol)
        
        Input: mol is a molecule object.
        
        Output: result is a numeric value.
    #################################################################
    """
    P2=len(Chem.FindAllPathsOfLengthN(mol,2))
    A=mol.GetNumAtoms()
    alpha=Chem.GraphDescriptors.HallKierAlpha(mol)
    denom=P2+alpha
    if denom:
        kappa=(A+alpha-1)*(A+alpha-2)**2/denom**2
    else:
        kappa=0.0
    return round(kappa,3)


def CalculateKappaAlapha3(mol):
    """
    #################################################################
    Calculation of molecular shape index for three bonded fragment 
    
    with Alapha
    
    ---->kappam3
    
    Usage:
        
        result=CalculateKappaAlapha3(mol)
        
        Input: mol is a molecule object.
        
        Output: result is a numeric value.
    #################################################################
    """
    P3=len(Chem.FindAllPathsOfLengthN(mol,3))
    A=mol.GetNumAtoms()
    alpha=Chem.GraphDescriptors.HallKierAlpha(mol)
    denom=P3+alpha
    if denom:
        if A % 2 == 1:
            kappa=(A+alpha-1)*(A+alpha-3)**2/denom**2
        else:
            kappa=(A+alpha-3)*(A+alpha-2)**2/denom**2
    else:
        kappa=0.0
    return round(kappa,3)



def CalculateFlexibility(mol):
    """
    #################################################################
    Calculation of Kier molecular flexibility index
    
    ---->phi
    
    Usage:
        
        result=CalculateFlexibility(mol)
        
        Input: mol is a molecule object.
        
        Output: result is a numeric value.
    #################################################################
    """
    kappa1=CalculateKappaAlapha1(mol)
    kappa2=CalculateKappaAlapha2(mol)
    A=mol.GetNumAtoms()
    phi=kappa1*kappa2/(A+0.0)
    return phi




##kappa

_Kappa = {
            'HallKierAlpha': Chem.GraphDescriptors.HallKierAlpha,
            'Kappa1': Chem.GraphDescriptors.Kappa1,
            'Kappa2': Chem.GraphDescriptors.Kappa2,
            'Kappa3': Chem.GraphDescriptors.Kappa3,
            'KappaAlapha1': CalculateKappaAlapha1,
            'KappaAlapha2': CalculateKappaAlapha2,
            'KappaAlapha3': CalculateKappaAlapha3,
            'KierFlexibilit': CalculateFlexibility
            }


_KappaNames = list(_Kappa.keys())

from collections import OrderedDict
def GetKappa(mol):
    """
    #################################################################
    Calculation of all kappa values.
    
    Usage:
        
        result=GetKappa(mol)
        
        Input: mol is a molecule object.
        
        Output: result is a dcit form containing 6 kappa values.
    #################################################################
    """
    res=OrderedDict()
    for k, func in  _Kappa.items():
        res.update({k:func(mol)})
    return res
################################################################
if __name__ =='__main__':
    
    import pandas as pd
    from tqdm import tqdm
    
    smis = ['C'*(i+1) for i in range(100)]
    x = []
    for index, smi in tqdm(enumerate(smis), ascii=True):
        m = Chem.MolFromSmiles(smi)
        x.append(GetKappa(m))
        
    pd.DataFrame(x)


