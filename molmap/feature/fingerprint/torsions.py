from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem import DataStructs
import numpy as np

_type = 'topological-based'


def GetTorsionFPs(mol, nBits = 2048, binary = True):
    '''
    atompairs fingerprints
    '''
    fp = Torsions.GetHashedTopologicalTorsionFingerprint(mol, nBits = nBits)
    if binary:
        arr = np.zeros((0,),  dtype=np.bool)
    else:
        arr = np.zeros((0,),  dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr