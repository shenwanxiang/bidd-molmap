# -*- coding: utf-8 -*-
from rdkit import Chem
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas(ascii=True)

from molmap.feature import descriptor

f = descriptor.property.FilterItLogS
fsol_calculator = lambda x:f(Chem.MolFromSmiles(x))



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
        best_train_pred_y = df_train.smiles.progress_apply(fsol_calculator).tolist()
        best_valid_pred_y = df_valid.smiles.progress_apply(fsol_calculator).tolist()
        best_test_pred_y = df_test.smiles.progress_apply(fsol_calculator).tolist()
        best_etc_pred_y = df_etc.smiles.progress_apply(fsol_calculator).tolist()


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

    pd.DataFrame(results).to_csv('../results/results_02_FSOL_model.csv')
