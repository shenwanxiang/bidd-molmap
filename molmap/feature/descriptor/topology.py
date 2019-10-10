#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 18:04:35 2019

@note: the varies topology indexes are calculated 
"""



from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import GraphDescriptors as GD

import numba as nb
import numpy as np
import scipy

periodicTable = rdchem.GetPeriodicTable()

MINVALUE = 1e-8

        
@nb.njit()
def _graphdist_(Distance):
    """
    Calculation of graph distance index: Tigdi(log value),
    The graph distance index is defined as the squared sum of all graph distance counts
    """
    unique = np.unique(Distance)
    res = MINVALUE
    for i in unique:
        k1 = Distance==i
        temp = k1.sum()
        res += temp**2
    return np.log10(res)

def CalculateGraphDistance(mol):
    """
    Calculation of graph distance index: Tigdi(log value),
    The graph distance index is defined as the squared sum of all graph distance counts
    """    
    Distance= Chem.GetDistanceMatrix(mol)
    return _graphdist_(Distance)


@nb.njit
def _Xu(Distance, nAT, deltas):
    sigma=Distance.sum(axis=1)
    temp1=MINVALUE
    temp2=MINVALUE
    for i in range(nAT):
        temp1 += deltas[i]*((sigma[i])**2)
        temp2 += deltas[i]*(sigma[i])
    Xu=np.sqrt(nAT)*np.log(temp1/temp2)
    return Xu

def CalculateXuIndex(mol):
    """
    Calculation of Xu index
    """
    nAT=mol.GetNumAtoms(onlyExplicit = True)
    deltas=np.array([x.GetDegree() for x in mol.GetAtoms()])
    Distance= Chem.GetDistanceMatrix(mol)
    return _Xu(Distance, nAT, deltas)



def CalculateWeiner(mol):
    """
    Calculation of Weiner number in a molecule
    """
    dist = Chem.GetDistanceMatrix(mol)
    s = 1.0/2*dist.sum()
    if s == 0:
        s = MINVALUE
    return np.log10(s)


def CalculateMeanWeiner(mol):

    N=mol.GetNumAtoms( onlyExplicit = True)
    WeinerNumber=CalculateWeiner(mol)
    
    if (N == 1) | (N == 0):
        N = 2
    return 2.0*WeinerNumber/(N*(N-1))


def CalculateBalaban(mol):

    J = Chem.GraphDescriptors.BalabanJ(mol)
    return J
    
    
def CalculateDiameter(mol):
    """
    Calculation of diameter, which is Largest value
    in the distance matrix [Petitjean 1992].
    """
    Distance=Chem.GetDistanceMatrix(mol)
    res = Distance.max()
    if res == 0:
        res = MINVALUE
    return np.log10(res)


def CalculateRadius(mol):
    """
    Calculation of radius based on topology.
    It is :If ri is the largest matrix entry in row i of the distance
    matrix D,then the radius is defined as the smallest of the ri 
    [Petitjean 1992].
    """
    Distance=Chem.GetDistanceMatrix(mol)
    temp = Distance.max(axis=0)
    res = temp.min()
    if res == 0:
        res = MINVALUE
    return np.log10(res)
    
    
def CalculatePetitjean(mol):
    """
    Calculation of Petitjean based on topology.
    Value of (diameter - radius) / diameter as defined in [Petitjean 1992].
    """
    diameter=CalculateDiameter(mol)
    radius=CalculateRadius(mol)
    
    if diameter == 0:
        diameter = MINVALUE
    return (diameter-radius)/diameter



@nb.njit()
def _Gutman(Distance, deltas, nAT):
    res=MINVALUE
    for i in range(nAT):
        for j in range(i+1,nAT):
            res=res+deltas[i]*deltas[j]*Distance[i,j]
    return np.log10(res)
    
def CalculateGutmanTopo(mol):
    """
    Calculation of Gutman molecular topological index based on
    simple vertex degree
    """
    nAT=mol.GetNumAtoms(onlyExplicit = True)
    deltas= np.array([x.GetDegree() for x in mol.GetAtoms()])
    Distance= Chem.GetDistanceMatrix(mol)
    res = _Gutman(Distance, deltas, nAT)
    return res

    
def CalculatePolarityNumber(mol):
    """
    Calculation of Polarity number. 
    It is the number of pairs of vertexes at distance matrix equal to 3
    """
    Distance= Chem.GetDistanceMatrix(mol)
    k3 = Distance==3
    res=1./2*k3.sum()
    
    if res == 0:
        res = MINVALUE
    return np.log10(res)



def _GetPrincipleQuantumNumber(atNum):
    """
    Get the principle quantum number of atom with atomic
    number equal to atNum 
    """
    if atNum<=2:
        return 1
    elif atNum<=10:
        return 2
    elif atNum<=18:
        return 3
    elif atNum<=36:
        return 4
    elif atNum<=54:
        return 5
    elif atNum<=86:
        return 6
    else:
        return 7
    

def CalculatePoglianiIndex(mol):
    """Calculation of Poglicani index
    
    The Pogliani index (Dz) is the sum over all non-hydrogen atoms
    
    of a modified vertex degree calculated as the ratio
    
    of the number of valence electrons over the principal
    
    quantum number of an atom [L. Pogliani, J.Phys.Chem. 1996, 100, 18065-18077].
    """
    res=MINVALUE
    for atom in mol.GetAtoms():
        n=atom.GetAtomicNum()
        nV=periodicTable.GetNOuterElecs(n)
        mP=_GetPrincipleQuantumNumber(n)
        res += (nV+0.0)/mP
    return np.log10(res)


def CalculateIpc(mol):
    """
    This returns the information content of the coefficients 
    of the characteristic polynomial of the adjacency matrix 
    of a hydrogen-suppressed graph of a molecule. 
    'avg = 1' returns the information content divided by the total population. 
    From D. Bonchev & N. Trinajstic, J. Chem. Phys. vol 67, 4517-4533 (1977)
    
    log of log values for index
    """
    temp=GD.Ipc(mol)
    if temp <= 0:
        temp = MINVALUE
    
    res = np.log10(temp) + 8 + MINVALUE
    return np.log(res)


def CalculateBertzCT(mol):
    """ 
    #################################################################
    A topological index meant to quantify "complexity" of molecules.

    Consists of a sum of two terms, one representing the complexity
    
    of the bonding, the other representing the complexity of the
    
    distribution of heteroatoms.

    From S. H. Bertz, J. Am. Chem. Soc., vol 103, 3599-3601 (1981)
    
    ---->BertzCT(log value)
    
    Usage: 
        
        result=CalculateBertzCT(mol)
        
        Input: mol is a molecule object
        
        Output: result is a numeric value
    #################################################################
    """
    temp=GD.BertzCT(mol)
    if temp > 0:
        return np.log10(temp)
    else:
        return np.log10(MINVALUE)


def CalculateHarary(mol):
    """
    Calculation of Harary number
    """
    
    Distance=np.array(Chem.GetDistanceMatrix(mol),'d')
    
    X = 1.0/(Distance[Distance!=0])
    res = 1.0/2*(X.sum())
    if res == 0:
        res = MINVALUE
    return np.log10(res)
        
    
def CalculateSchiultz(mol):
    """
    #################################################################
    Calculation of Schiultz number
    
    ---->Tsch(log value)
    
    Usage: 
        
        result=CalculateSchiultz(mol)
        
        Input: mol is a molecule object
        
        Output: result is a numeric value
    #################################################################
    """
    Distance=np.array(Chem.GetDistanceMatrix(mol),'d')
    Adjacent=np.array(Chem.GetAdjacencyMatrix(mol),'d')
    VertexDegree=sum(Adjacent)
    res = sum(scipy.dot((Distance+Adjacent),VertexDegree))
    if res ==0:
        res = MINVALUE
    
    return np.log10(res)



def CalculateZagreb1(mol):
    """
    Calculation of Zagreb index with order 1 in a molecule
    """
    
    deltas=[x.GetDegree() for x in mol.GetAtoms()]
    
    res = sum(np.array(deltas)**2)
    
    if res == 0:
        res = MINVALUE
    return np.log10(res)


def CalculateZagreb2(mol):
    
    """
    #################################################################
    Calculation of Zagreb index with order 2 in a molecule
    
    ---->ZM2
    
    Usage: 
        
        result=CalculateZagreb2(mol)
        
        Input: mol is a molecule object
        
        Output: result is a numeric value
    #################################################################
    """
    ke = [x.GetBeginAtom().GetDegree()*x.GetEndAtom().GetDegree() for x in mol.GetBonds()]
    res = sum(ke)
    if res == 0:
        res = MINVALUE
    return np.log10(res)


def CalculateMZagreb1(mol):
    """
    #################################################################
    Calculation of Modified Zagreb index with order 1 in a molecule
    
    ---->MZM1
    
    Usage: 
        
        result=CalculateMZagreb1(mol)
        
        Input: mol is a molecule object
        
        Output: result is a numeric value
    #################################################################
    """
    deltas=[x.GetDegree() for x in mol.GetAtoms()]
    while 0 in deltas:
        deltas.remove(0)
    deltas=np.array(deltas,'d')
    res=sum((1./deltas)**2)
    
    if res == 0:
        res = MINVALUE
    return np.log10(res)
    

def CalculateMZagreb2(mol):
    """
    #################################################################
    Calculation of Modified Zagreb index with order 2 in a molecule
    
    ---->MZM2
    
    Usage: 
        
        result=CalculateMZagreb2(mol)
        
        Input: mol is a molecule object
        
        Output: result is a numeric value
    #################################################################
    """
    cc = [x.GetBeginAtom().GetDegree()*x.GetEndAtom().GetDegree() for x in mol.GetBonds()]
    while 0 in cc:
        cc.remove(0)
    cc = np.array(cc,'d')
    res = sum((1./cc)**2)
    if res == 0:
        res = MINVALUE
    return np.log10(res)



def CalculatePlatt(mol):
    """
    #################################################################
    Calculation of Platt number in a molecule
    
    ---->Platt
    
    Usage: 
        
        result=CalculatePlatt(mol)
        
        Input: mol is a molecule object
        
        Output: result is a numeric value
    #################################################################
    """
    cc = [x.GetBeginAtom().GetDegree()+x.GetEndAtom().GetDegree()-2 for x in mol.GetBonds()]
    res = sum(cc)
    if res == 0:
        res = MINVALUE
    return np.log10(res)



def CalculateSimpleTopoIndex(mol):
    """
    #################################################################
    Calculation of the logarithm of the simple topological index by Narumi,
    
    which is defined as the product of the vertex degree.
    
    ---->Sito
    
    Usage: 
        
        result=CalculateSimpleTopoIndex(mol)
        
        Input: mol is a molecule object
        
        Output: result is a numeric value
    #################################################################
    """
    deltas=[x.GetDegree() for x in mol.GetAtoms()]
    while 0 in deltas:
        deltas.remove(0)
    deltas=np.array(deltas,'d')
    
    res=np.prod(deltas)
    if res>0:
        return np.log10(res)
    else:
        return np.log10(MINVALUE)
    
def CalculateHarmonicTopoIndex(mol):
    """
    #################################################################
    Calculation of harmonic topological index proposed by Narnumi.
    
    ---->Hato
    
    Usage: 
        
        result=CalculateHarmonicTopoIndex(mol)
        
        Input: mol is a molecule object
        
        Output: result is a numeric value
    #################################################################
    """
    deltas=[x.GetDegree() for x in mol.GetAtoms()]
    while 0 in deltas:
        deltas.remove(0)
    
    deltas=np.array(deltas,'d')  
    nAtoms=mol.GetNumAtoms( onlyExplicit = True)
    
    if sum(1./deltas) == 0:
        res = MINVALUE
    else:
        res=nAtoms/sum(1./deltas)

    return res


def CalculateGeometricTopoIndex(mol):
    """
    #################################################################
    Geometric topological index by Narumi
    
    ---->Geto
    
    Usage: 
        
        result=CalculateGeometricTopoIndex(mol)
        
        Input: mol is a molecule object
        
        Output: result is a numeric value
    #################################################################
    """
    nAtoms=mol.GetNumAtoms(onlyExplicit = True)
    deltas=[x.GetDegree() for x in mol.GetAtoms()]
    while 0 in deltas:
        deltas.remove(0)
    deltas=np.array(deltas,'d')
    temp=np.prod(deltas)
    res=np.power(temp,1./nAtoms)

    return res    

def CalculateArithmeticTopoIndex(mol):
    """
    #################################################################
    Arithmetic topological index by Narumi
    
    ---->Arto
    
    Usage: 
        
        result=CalculateArithmeticTopoIndex(mol)
        
        Input: mol is a molecule object
        
        Output: result is a numeric value
    #################################################################
    """
    nAtoms=mol.GetNumAtoms(onlyExplicit = True)
    nBonds=mol.GetNumBonds()
    
    res=2.*nBonds/nAtoms
    return res



_Topology={'IndexWeiner':CalculateWeiner,
           'IndexAvgWeiner':CalculateMeanWeiner,
           'IndexBalabanJ':CalculateBalaban,
           'IndexBertzCT':CalculateBertzCT,
           'IndexGraphDistance':CalculateGraphDistance,
           'IndexXu':CalculateXuIndex,
           'IndexIpc':CalculateIpc,
           
           'IndexGutmanTopo':CalculateGutmanTopo,
           'IndexPogliani':CalculatePoglianiIndex,
           
           'IndexPolarity':CalculatePolarityNumber,
           'IndexHarary':CalculateHarary,

           'IndexSchiultz':CalculateSchiultz,
           'IndexZagreb1':CalculateZagreb1,
           'IndexZagreb2':CalculateZagreb2,
           
           'IndexMZagreb1':CalculateMZagreb1,
           'IndexMZagreb2':CalculateMZagreb2,
           
           'IndexPlatt':CalculatePlatt,
           'IndexDiameter':CalculateDiameter,
           'IndexRadius':CalculateRadius,
           'IndexPetitjean':CalculatePetitjean,
           'IndexSimpleTopo':CalculateSimpleTopoIndex,
           'IndexHarmonicTopo':CalculateHarmonicTopoIndex,
           'IndexGeometricTopo':CalculateGeometricTopoIndex,
           'IndexArithmeticTopo':CalculateArithmeticTopoIndex      
    }
    
    
    
_TopologyNames = list(_Topology.keys())

from collections import OrderedDict
def GetTopology(mol):
    """
    #################################################################
    Get the dictionary of constitutional descriptors for given
    
    moelcule mol
    
    Usage: 
        
        result=CalculateWeiner(mol)
        
        Input: mol is a molecule object
        
        Output: result is a dict form containing all topological indices.
    #################################################################
    """
    result=OrderedDict()
    for k, func in _Topology.items():
        result[k]= func(mol)
    return result



if __name__ =='__main__':
    
    import pandas as pd
    from tqdm import tqdm
    
    smis = ['C'*(i+1) for i in range(100)]
    x = []
    for index, smi in tqdm(enumerate(smis), ascii=True):
        m = Chem.MolFromSmiles(smi)
        x.append(GetTopology(m))
        
    pd.DataFrame(x)