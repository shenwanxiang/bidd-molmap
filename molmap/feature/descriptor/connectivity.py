"""

The calculation of molecular connectivity indices based on its topological

structure(Chi). You can get molecular connectivity descriptors.

"""
from mordred import Calculator, descriptors
import numpy as np

_calc = Calculator(descriptors.Chi)

_ConnectivityNames = [str(i) for i in _calc.descriptors]


def GetConnectivity(mol):
    r = _calc(mol)
    r = r.fill_missing(0)
    return r.asdict()
    
    


###############################################################################
if __name__ =='__main__':
    
    smis = ['CCCC','CCCCC','CCCCCC','CC(N)C(=O)O','CC(N)C(=O)[O-].[Na+]']
    smi5=['CCCCCC','CCC(C)CC','CC(C)CCC','CC(C)C(C)C','CCCCCN','c1ccccc1N']

    for index, smi in enumerate(smis):
        m = Chem.MolFromSmiles(smi)
        GetConnectivity(m)
