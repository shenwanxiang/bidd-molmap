from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import DataStructs
import numpy as np

_type = 'topological-based'


def GetAtomPairFPs(mol, nBits = 2048, binary = True):
    '''
    atompairs fingerprints
    '''
    fp = Pairs.GetHashedAtomPairFingerprint(mol, nBits = nBits)
    if binary:
        arr = np.zeros((0,),  dtype=np.bool)
    else:
        arr = np.zeros((0,),  dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr