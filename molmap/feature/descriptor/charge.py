#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 18:04:35 2019

@author: charleshen

@note: this code derived from PybioMed<https://github.com/gadsbyfly/PyBioMed>, with a major modified

"""


from rdkit import Chem
from rdkit.Chem import rdPartialCharges as GMCharge

import numpy


##############################################################################
iter_step=12


                    
def _CalculateElementMaxPCharge(mol,AtomicNum=6):
    """
    #################################################################
    **Internal used only**
    
    Most positive charge on atom with atomic number equal to n
    #################################################################
    """
    Hmol=Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol,iter_step)
    res=[]
    for atom in Hmol.GetAtoms():
        if atom.GetAtomicNum()==AtomicNum:
            res.append(float(atom.GetProp('_GasteigerCharge')))
            
    if res==[]:
        return 0
    else:
        return max(res)

def _CalculateElementMaxNCharge(mol,AtomicNum=6):
    """
    #################################################################
    **Internal used only**
    
    Most negative charge on atom with atomic number equal to n
    #################################################################
    """
    Hmol=Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol,iter_step)
    res=[]
    for atom in Hmol.GetAtoms():
        if atom.GetAtomicNum()==AtomicNum:
            res.append(float(atom.GetProp('_GasteigerCharge')))
    if res==[]:
        return 0
    else:
        return min(res)


def CalculateHMaxPCharge(mol):
    """
    #################################################################
    Most positive charge on H atoms
    
    -->QHmax
    
    Usage:
    
        result=CalculateHMaxPCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    return _CalculateElementMaxPCharge(mol,AtomicNum=1)


def CalculateCMaxPCharge(mol):
    """
    #################################################################
    Most positive charge on C atoms
    
    -->QCmax

    Usage:
    
        result=CalculateCMaxPCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    return _CalculateElementMaxPCharge(mol,AtomicNum=6)


def CalculateNMaxPCharge(mol):
    """
    #################################################################
    Most positive charge on N atoms
    
    -->QNmax

    Usage:
    
        result=CalculateNMaxPCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    return _CalculateElementMaxPCharge(mol,AtomicNum=7)


def CalculateOMaxPCharge(mol):
    """
    #################################################################
    Most positive charge on O atoms
    
    -->QOmax

    Usage:
    
        result=CalculateOMaxPCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    return _CalculateElementMaxPCharge(mol,AtomicNum=8)

def CalculateHMaxNCharge(mol):
    """
    #################################################################
    Most negative charge on H atoms
  
    -->QHmin

    Usage:
    
        result=CalculateHMaxNCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    return _CalculateElementMaxNCharge(mol,AtomicNum=1)


def CalculateCMaxNCharge(mol):
    """
    #################################################################
    Most negative charge on C atoms
    
    -->QCmin

    Usage:
    
        result=CalculateCMaxNCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    return _CalculateElementMaxNCharge(mol,AtomicNum=6)


def CalculateNMaxNCharge(mol):
    """
    #################################################################
    Most negative charge on N atoms
    
    -->QNmin

    Usage:
    
        result=CalculateNMaxNCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    return _CalculateElementMaxNCharge(mol,AtomicNum=7)


def CalculateOMaxNCharge(mol):
    """
    #################################################################
    Most negative charge on O atoms
    
    -->QOmin

    Usage:
    
        result=CalculateOMaxNCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    return _CalculateElementMaxNCharge(mol,AtomicNum=8)

def CalculateAllMaxPCharge(mol):
    """
    #################################################################
    Most positive charge on ALL atoms
    
    -->Qmax

    Usage:
    
        result=CalculateAllMaxPCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    Hmol=Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol,iter_step)
    res=[]
    for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))
    if res==[]:
        return 0
    else:
        return max(res)


def CalculateAllMaxNCharge(mol):
    """
    #################################################################
    Most negative charge on all atoms
    
    -->Qmin

    Usage:
    
        result=CalculateAllMaxNCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    Hmol=Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol,iter_step)
    res=[]
    for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))
    if res==[]:
        return 0
    else:
        return min(res)


def _CalculateElementSumSquareCharge(mol,AtomicNum=6):
    """
    #################################################################
    **Internal used only**
    
    Ths sum of square Charges on all atoms with atomicnumber equal to n
    #################################################################
    """
    Hmol=Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol,iter_step)
    res=[]
    for atom in Hmol.GetAtoms():
        if atom.GetAtomicNum()==AtomicNum:
            res.append(float(atom.GetProp('_GasteigerCharge')))
    if res==[]:
        return 0
    else:
        return sum(numpy.square(res))


def CalculateHSumSquareCharge(mol):
    
    """
    #################################################################
    The sum of square charges on all H atoms
    
    -->QHss

    Usage:
    
        result=CalculateHSumSquareCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    return _CalculateElementSumSquareCharge(mol,AtomicNum=1)


def CalculateCSumSquareCharge(mol):
    """
    #################################################################
    The sum of square charges on all C atoms
    
    -->QCss

    Usage:
    
        result=CalculateCSumSquareCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    return _CalculateElementSumSquareCharge(mol,AtomicNum=6)


def CalculateNSumSquareCharge(mol):
    """
    #################################################################
    The sum of square charges on all N atoms
    
    -->QNss

    Usage:
    
        result=CalculateNSumSquareCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    return _CalculateElementSumSquareCharge(mol,AtomicNum=7)

def CalculateOSumSquareCharge(mol):
    """
    #################################################################
    The sum of square charges on all O atoms
    
    -->QOss

    Usage:
    
        result=CalculateOSumSquareCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    return _CalculateElementSumSquareCharge(mol,AtomicNum=8)

def CalculateAllSumSquareCharge(mol):
    """
    #################################################################
    The sum of square charges on all atoms
    
    -->Qass

    Usage:
    
        result=CalculateAllSumSquareCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    Hmol=Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol,iter_step)
    res=[]
    for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))
            
    if res==[]:
        return 0
    else:
        return sum(numpy.square(res))

def CalculateTotalPCharge(mol):
    """
    #################################################################
    The total postive charge
    
    -->Tpc

    Usage:
    
        result=CalculateTotalPCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    Hmol=Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol,iter_step)
    res=[]
    for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))
            
    if res==[]:
        return 0
    else:
        cc=numpy.array(res,'d')
        return sum(cc[cc>0])

def CalculateMeanPCharge(mol):
    """
    #################################################################
    The average postive charge
    
    -->Mpc
    
    Usage:
    
        result=CalculateMeanPCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    Hmol=Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol,iter_step)
    res=[]
    for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))
            
    if res==[]:
        return 0
    else:
        cc=numpy.array(res,'d')
        return numpy.mean(cc[cc>0])


def CalculateTotalNCharge(mol):
    """
    #################################################################
    The total negative charge
    
    -->Tnc
    
    Usage:
    
        result=CalculateTotalNCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    Hmol=Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol,iter_step)
    res=[]
    for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))
            
    if res==[]:
        return 0
    else:
        cc=numpy.array(res,'d')
        return sum(cc[cc<0])


def CalculateMeanNCharge(mol):
    """
    #################################################################
    The average negative charge
    
    -->Mnc
    
    Usage:
    
        result=CalculateMeanNCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    Hmol=Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol,iter_step)
    res=[]
    for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))
            
    if res==[]:
        return 0
    else:
        cc=numpy.array(res,'d')
        return numpy.mean(cc[cc<0])


def CalculateTotalAbsoulteCharge(mol):
    """
    #################################################################
    The total absolute charge
    
    -->Tac
    
    Usage:
    
        result=CalculateTotalAbsoulteCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    Hmol=Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol,iter_step)
    res=[]
    for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))
            
    if res==[]:
        return 0
    else:
        cc=numpy.array(res,'d')
        return sum(numpy.absolute(cc))

def CalculateMeanAbsoulteCharge(mol):
    """
    #################################################################
    The average absolute charge
    
    -->Mac
    
    Usage:
    
        result=CalculateMeanAbsoulteCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    Hmol=Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol,iter_step)
    res=[]
    for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))
            
    if res==[]:
        return 0
    else:
        cc=numpy.array(res,'d')
        return numpy.mean(numpy.absolute(cc))

def CalculateRelativePCharge(mol):
    """
    #################################################################
    The partial charge of the most positive atom divided by
    
    the total positive charge.
    
    -->Rpc
    
    Usage:
    
        result=CalculateRelativePCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    Hmol=Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol,iter_step)
    res=[]
    for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))
            
    if res==[]:
        return 0
    else:
        cc=numpy.array(res,'d')
        if sum(cc[cc>0])==0:
            return 0
        else:
            return max(res)/sum(cc[cc>0])

def CalculateRelativeNCharge(mol):
    
    """
    #################################################################
    The partial charge of the most negative atom divided
    
    by the total negative charge.
    
    -->Rnc
    
    Usage:
    
        result=CalculateRelativeNCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """
    Hmol=Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol,iter_step)
    res=[]
    for atom in Hmol.GetAtoms():
            res.append(float(atom.GetProp('_GasteigerCharge')))
            
    if res==[]:
        return 0
    else:
        cc=numpy.array(res,'d')
        if sum(cc[cc<0])==0:
            return 0
        else:
            return min(res)/sum(cc[cc<0])

def CalculateLocalDipoleIndex(mol):
    """
    #################################################################
    Calculation of local dipole index (D)
    
    -->LDI
    
    Usage:
    
        result=CalculateLocalDipoleIndex(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """

    GMCharge.ComputeGasteigerCharges(mol,iter_step)
    res=[]
    for atom in mol.GetAtoms():
        res.append(float(atom.GetProp('_GasteigerCharge')))
    cc = [numpy.absolute(res[x.GetBeginAtom().GetIdx()]-res[x.GetEndAtom().GetIdx()]) for x in mol.GetBonds()]
    B=len(mol.GetBonds())
    
    if B == 0:
        return 0
    return sum(cc)/B
        
def CalculateSubmolPolarityPara(mol):
    """
    #################################################################
    Calculation of submolecular polarity parameter(SPP)
    
    -->SPP
    
    Usage:
    
        result=CalculateSubmolPolarityPara(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a numeric value.
    #################################################################
    """

    return CalculateAllMaxPCharge(mol)-CalculateAllMaxNCharge(mol)



_Charge={'SubmolPolarityPara':CalculateSubmolPolarityPara,
         'LocalDipoleIndex':CalculateLocalDipoleIndex,

        'NChargeRelative':CalculateRelativeNCharge,
        'PChargeRelative':CalculateRelativePCharge,
         
         
        'ChargeMeanAbsoulte':CalculateMeanAbsoulteCharge,
        'ChargeTotalAbsoulte':CalculateTotalAbsoulteCharge,
         
         
        'NChargeMean':CalculateMeanNCharge,
        'NChargeTotal':CalculateTotalNCharge,
         
         
        'PChargeMean':CalculateMeanPCharge,
        'PChargeTotal':CalculateTotalPCharge,
         
         ## sum square
        'SumSquareChargeAll':CalculateAllSumSquareCharge,
        'SumSquareChargeO':CalculateOSumSquareCharge,
        'SumSquareChargeN':CalculateNSumSquareCharge,
        'SumSquareChargeC':CalculateCSumSquareCharge,
        'SumSquareChargeH':CalculateHSumSquareCharge,
         
         #all 
        'NChargeAllMax':CalculateAllMaxNCharge,
        'PChargeAllMax':CalculateAllMaxPCharge,
         
         #negative
        'NChargeOMax':CalculateOMaxNCharge,
        'NChargeNMax':CalculateNMaxNCharge,
        'NChargeCMax':CalculateCMaxNCharge,
        'NChargeHMax':CalculateHMaxNCharge,
         
         #positive
        'PChargeOMax':CalculateOMaxPCharge,
        'PChargeNMax':CalculateNMaxPCharge,
        'PChargeCMax':CalculateCMaxPCharge,
        'PChargeHMax':CalculateHMaxPCharge,
    }


from collections import OrderedDict
def GetCharge(mol):
    """
    #################################################################
    Get the dictionary of constitutional descriptors for given moelcule mol
    
    Usage:
    
        result=GetCharge(mol)
    
        Input: mol is a molecule object.
    
        Output: result is a dict form containing all charge descriptors.
    #################################################################
    """
    result=OrderedDict()
    for DesLabel in _Charge.keys():
        res = _Charge[DesLabel](mol)
        if abs(res) == numpy.inf:
            res = numpy.nan
        result[DesLabel]=res
    return result


_ChargeNames = list(_Charge.keys())
##############################################################################

if __name__ =='__main__':
    
    import pandas as pd
    from tqdm import tqdm
    
    smis = ['C'*(i+1) for i in range(100)]
    x = []
    for index, smi in tqdm(enumerate(smis), ascii=True):
        m = Chem.MolFromSmiles(smi)
        x.append(GetCharge(m))
        
    pd.DataFrame(x)

