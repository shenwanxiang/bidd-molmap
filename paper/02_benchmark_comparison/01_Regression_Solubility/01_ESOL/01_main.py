from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem import MolSurf
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski

from rdkit import Chem

from collections import namedtuple
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas(ascii=True)



# this code copyed from https://github.com/PatWalters/solubility/blob/master/esol.py, with a minor modification
class ELogS:
    '''
    ESOL:  Estimating Aqueous Solubility Directly from Molecular Structure 
    John S. Delaney, J. Chem. Inf. Comput. Sci., 2004, 44, 1000 - 1005
    https://pubs.acs.org/doi/abs/10.1021/ci034243x 
    '''
    
    def __init__(self):
        self.aromatic_query = Chem.MolFromSmarts("a")
        self.Descriptor = namedtuple("Descriptor", "mw logp rotors ap")

    def calc_ap(self, mol):
        """
        Calculate aromatic proportion #aromatic atoms/#atoms total
        :param mol: input molecule
        :return: aromatic proportion
        """
        matches = mol.GetSubstructMatches(self.aromatic_query)
        return len(matches) / mol.GetNumAtoms()

    def calc_esol_descriptors(self, mol):
        """
        Calcuate mw,logp,rotors and aromatic proportion (ap)
        :param mol: input molecule
        :return: named tuple with descriptor values
        """
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        rotors = Lipinski.NumRotatableBonds(mol)
        ap = self.calc_ap(mol)
        return self.Descriptor(mw=mw, logp=logp, rotors=rotors, ap=ap)


    def call(self, mol):
        """
        Calculate ESOL based on descriptors in the Delaney paper, coefficients refit for the RDKit using the
        routine refit_esol below
        :param mol: input molecule
        :return: predicted solubility
        """
        intercept = 0.26121066137801696
        coef = {'mw': -0.0066138847738667125, 'logp': -0.7416739523408995, 
                'rotors': 0.003451545565957996, 'ap': -0.42624840441316975}
        desc = self.calc_esol_descriptors(mol)
        logs = intercept + coef["logp"] * desc.logp + coef["mw"] * desc.mw + coef["rotors"] * desc.rotors \
               + coef["ap"] * desc.ap
        return logs
    
    
    

esol_calculator = lambda x:ELogS().call(Chem.MolFromSmiles(x))


df_train = pd.read_csv('../train.csv', index_col = 0)
df_valid = pd.read_csv('../valid.csv',  index_col = 0)
df_test = pd.read_csv('../test.csv',  index_col = 0)
df_etc = pd.read_csv('../etc.csv')
task = 'measured log solubility in mols per litre'
df_etc[task] = df_etc.Exp_LogS


if __name__ == '__main__':
    from sklearn.metrics import mean_squared_error as mse
    results = []
    
    
    for seed in [7, 77, 777]:
        best_train_pred_y = df_train.smiles.progress_apply(esol_calculator).tolist()
        best_valid_pred_y = df_valid.smiles.progress_apply(esol_calculator).tolist()
        best_test_pred_y = df_test.smiles.progress_apply(esol_calculator).tolist()
        best_etc_pred_y = df_etc.smiles.progress_apply(esol_calculator).tolist()


        final_res = {'train_rmse':np.sqrt(mse(df_train[task], best_train_pred_y)), 
                     'valid_rmse':np.sqrt(mse(df_valid[task], best_valid_pred_y)), 
                     'test_rmse':np.sqrt(mse(df_test[task], best_test_pred_y)), 
                     'etc_rmse':np.sqrt(mse(df_etc[task], best_etc_pred_y)),

                     'train_pred_y':best_train_pred_y,
                     'valid_pred_y':best_valid_pred_y,                     
                     'test_pred_y':best_test_pred_y,
                     'etc_pred_y':best_etc_pred_y, 

                     '# trainable params': np.nan,
                     'random_seed':seed, 
                     'best_epoch': np.nan,
                     'batch_size':np.nan,
                     'lr': np.nan,
                    }

        results.append(final_res)

    pd.DataFrame(results).to_csv('../results/results_01_ESOL_model.csv')