"""
topological fingerprint

"""


import numpy as np
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem import  DataStructs

_type = 'topological-based'



def GetRDkitFPs(mol, nBits = 2048, return_bitInfo = False):
    """
    #################################################################
    Calculate Daylight-like fingerprint or topological fingerprint
    
    (1024 bits).
    
    Usage:
        
        result=CalculateDaylightFingerprint(mol)
        
        Input: mol is a molecule object.
        
        Output: result is a tuple form. The first is the number of 
        
        fingerprints. The second is a dict form whose keys are the 
        
        position which this molecule has some substructure. The third
        
        is the DataStructs which is used for calculating the similarity.
    #################################################################
    """
    
    bitInfo = {}
    fp = RDKFingerprint(mol, fpSize = nBits, bitInfo = bitInfo)
    arr = np.zeros((0,),  dtype=np.bool)
    DataStructs.ConvertToNumpyArray(fp, arr)    
    if return_bitInfo:
        return arr, return_bitInfo
    return arr

