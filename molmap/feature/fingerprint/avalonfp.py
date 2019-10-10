from rdkit.Chem import DataStructs
from rdkit.Avalon.pyAvalonTools import GetAvalonFP as GAFP
import numpy as np

_type = 'topological-based'


def GetAvalonFPs(mol, nBits=2048):

    '''
    Avalon_fingerprints: https://pubs.acs.org/doi/pdf/10.1021/ci050413p
    '''

    fp = GAFP(mol, nBits = nBits)
    arr = np.zeros((0,),  dtype=np.bool)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr