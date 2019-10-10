#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 18:04:35 2019

@author: charleshen

@usage:graph descriptors

http://rdkit.org/docs/source/rdkit.Chem.rdmolops.html https://www.rdkit.org/docs/source/rdkit.Chem.GraphDescriptors.html#rdkit.Chem.GraphDescriptors.Chi0

Subgraphs are potentially branched, whereas paths (in our 
terminology at least) cannot be.  So, the following graph: 

     C--0--C--1--C--3--C
           |
           2
           |
           C

has 3 _subgraphs_ of length 3: (0,1,2),(0,1,3),(2,1,3)
but only 2 _paths_ of length 3: (0,1,3),(2,1,3)

ref.: http://rdkit.org/docs/source/rdkit.Chem.rdmolops.html

"""



from rdkit import Chem



def CalculatePath1(mol):
    return len(Chem.FindAllPathsOfLengthN(mol,1))

def CalculatePath2(mol):
    return len(Chem.FindAllPathsOfLengthN(mol,2))

def CalculatePath3(mol):
    return len(Chem.FindAllPathsOfLengthN(mol,3))

def CalculatePath4(mol):
    return len(Chem.FindAllPathsOfLengthN(mol,4))

def CalculatePath5(mol):
    return len(Chem.FindAllPathsOfLengthN(mol,5))

def CalculatePath6(mol):
    return len(Chem.FindAllPathsOfLengthN(mol,6))


def CalculateSubgraphPath1(mol):
    return len(Chem.FindAllSubgraphsOfLengthN(mol,1))

def CalculateSubgraphPath2(mol):
    return len(Chem.FindAllSubgraphsOfLengthN(mol,2))

def CalculateSubgraphPath3(mol):
    return len(Chem.FindAllSubgraphsOfLengthN(mol,3))

def CalculateSubgraphPath4(mol):
    return len(Chem.FindAllSubgraphsOfLengthN(mol,4))

def CalculateSubgraphPath5(mol):
    return len(Chem.FindAllSubgraphsOfLengthN(mol,5))

def CalculateSubgraphPath6(mol):
    return len(Chem.FindAllSubgraphsOfLengthN(mol,6))




def CalculateUniqueSubgraphPath1(mol):
    return len(Chem.FindUniqueSubgraphsOfLengthN(mol,1))

def CalculateUniqueSubgraphPath2(mol):
    return len(Chem.FindUniqueSubgraphsOfLengthN(mol,2))

def CalculateUniqueSubgraphPath3(mol):
    return len(Chem.FindUniqueSubgraphsOfLengthN(mol,3))

def CalculateUniqueSubgraphPath4(mol):
    return len(Chem.FindUniqueSubgraphsOfLengthN(mol,4))

def CalculateUniqueSubgraphPath5(mol):
    return len(Chem.FindUniqueSubgraphsOfLengthN(mol,5))

def CalculateUniqueSubgraphPath6(mol):
    return len(Chem.FindUniqueSubgraphsOfLengthN(mol,6))








_Path = {
        ## Paths
        'AllPathsOfLength1': CalculatePath1,
        'AllPathsOfLength2': CalculatePath2,
        'AllPathsOfLength3': CalculatePath3,
        'AllPathsOfLength4': CalculatePath4,
        'AllPathsOfLength5': CalculatePath5,
        'AllPathsOfLength6': CalculatePath6,

        ##subgraph
        'AllSubgraphsOfLength1': CalculateSubgraphPath1,
        'AllSubgraphsOfLength2': CalculateSubgraphPath2,
        'AllSubgraphsOfLength3': CalculateSubgraphPath3,
        'AllSubgraphsOfLength4': CalculateSubgraphPath4,
        'AllSubgraphsOfLength5': CalculateSubgraphPath5,
        'AllSubgraphsOfLength6': CalculateSubgraphPath6,
    
        #unique subgraph
        'UniqueSubgraphsOfLength1': CalculateUniqueSubgraphPath1,
        'UniqueSubgraphsOfLength2': CalculateUniqueSubgraphPath2,
        'UniqueSubgraphsOfLength3': CalculateUniqueSubgraphPath3,
        'UniqueSubgraphsOfLength4': CalculateUniqueSubgraphPath4,
        'UniqueSubgraphsOfLength5': CalculateUniqueSubgraphPath5,
        'UniqueSubgraphsOfLength6': CalculateUniqueSubgraphPath6,
        }

_PathNames  = list(_Path.keys())

from collections import OrderedDict
def GetPath(mol):
    """
    Calculation of all path values.
    
    Usage:
        
        result=GetPath(mol)
        
        Input: mol is a molecule object.
        
        Output: result is a dcit form containing 6 kappa values.

    """
    res=OrderedDict()
    for k, func in  _Path.items():
        res.update({k:func(mol)})
    return res


def _GetHTMLDoc():
    """
    #################################################################
    Write HTML documentation for this module.
    #################################################################
    """
    import pydoc,os
    name = os.path.basename(__file__).replace('.py', '')
    pydoc.writedoc(name)  
    
    
################################################################
if __name__ =='__main__':

    
    import pandas as pd
    from tqdm import tqdm
    
    smis = ['C'*(i+1) for i in range(100)]
    x = []
    for index, smi in tqdm(enumerate(smis), ascii=True):
        m = Chem.MolFromSmiles(smi)
        x.append(GetPath(m))
        
    pd.DataFrame(x)