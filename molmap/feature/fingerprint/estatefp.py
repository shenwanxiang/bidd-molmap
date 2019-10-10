from rdkit.Chem.EState import Fingerprinter
import numpy as np


_type = 'Estate-based'

def GetEstateFPs(mol):
    '''
    79 bits Estate fps
    '''
    x = Fingerprinter.FingerprintMol(mol)[0]
    return x.astype(np.bool)