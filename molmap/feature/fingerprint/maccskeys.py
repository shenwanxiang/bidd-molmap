from rdkit.Chem import AllChem
from rdkit.Chem import  DataStructs
import numpy as np

_type = 'SMARTS-based'

def GetMACCSFPs(mol):

    '''
    166 bits
    '''

    fp =  AllChem.GetMACCSKeysFingerprint(mol)

    arr = np.zeros((0,),  dtype=np.bool)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
