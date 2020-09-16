from rdkit.Chem import AllChem
from rdkit.Chem import  DataStructs
import numpy as np
import pandas as pd
import os

_type = 'SMARTS-based'

file_path = os.path.dirname(__file__)

def GetMACCSFPs(mol):

    '''
    166 bits
    '''

    fp =  AllChem.GetMACCSKeysFingerprint(mol)

    arr = np.zeros((0,),  dtype=np.bool)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def GetMACCSFPInfos():
    return pd.read_excel(os.path.join(file_path, 'maccskeys.xlsx'))