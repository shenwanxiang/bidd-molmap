from molmap.model import RegressionEstimator, MultiClassEstimator, MultiLabelEstimator
from molmap import loadmap, dataset,feature
from molmap.show import imshow_wrap
from chembench import load_data, dataset

from sklearn.utils import shuffle 
from joblib import load, dump
import numpy as np
import pandas as pd
import os

bitsinfo = feature.fingerprint.Extraction().bitsinfo
fp_types = bitsinfo.Subtypes.unique()

esol = dataset.load_ESOL()
lipop = dataset.load_Lipop()
FreeSolv = dataset.load_FreeSolv()
PDBF = dataset.load_PDBF()

datasets = [esol, lipop, PDBF, FreeSolv] #malaria

tmp_feature_dir = '/raid/shenwanxiang/10_FP_effect/tempignore'
if not os.path.exists(tmp_feature_dir):
    os.makedirs(tmp_feature_dir)
    
mps = []
fp_save_folder = '/raid/shenwanxiang/FP_maps'
for fp_type in fp_types:
    mp = loadmap(os.path.join(fp_save_folder, '%s.mp' % fp_type))
    mps.append(mp)
    
    
Regression_res = []
## classification
for data in datasets:
    
    task_name = data.task_name
    task_type = data.task_type
    _, induces = load_data(task_name)
    smiles = data.x
    Y = data.y
    
    for mp, fp_type in zip(mps, fp_types):
        
        print(fp_type)
        
        X2_name = "X2_%s_%s.data" % (task_name, fp_type)
        X2_name = os.path.join(tmp_feature_dir, X2_name)
        if not os.path.exists(X2_name):
            X2 = mp.batch_transform(smiles, scale = False, n_jobs = 16)
            dump(X2, X2_name)
        else:
            X2 = load(X2_name)
            
        for i, idx in enumerate(induces):
            train_idx, valid_idx, test_idx = idx

            X = X2[train_idx]
            y = Y[train_idx]

            X_valid = X2[valid_idx]
            y_valid = Y[valid_idx]

            X_test = X2[test_idx]
            y_test = Y[test_idx]          
    
            fmap_shape1 = X.shape[1:]
            n_outputs = Y.shape[1]

            clf = RegressionEstimator(n_outputs=n_outputs,  
                                      fmap_shape1 = fmap_shape1, 
                                      dense_layers = [128, 64],
                                      gpuid = 5,
                                      batch_size = 64,
                                     ) 
            
            clf.fit(X, y, X_valid, y_valid)
            
            train_rmses, train_r2s = clf._performance.evaluate(X,y)
            valid_rmses, valid_r2s = clf._performance.evaluate(X_valid,y_valid)            
            test_rmses, test_r2s = clf._performance.evaluate(X_test,y_test)
            

            final_res = {'task_name':task_name, 
                         'fp_type': fp_type, 
                         'repeat_id':i,
                         'train_rmse':np.nanmean(train_rmses), 
                         'valid_rmse':np.nanmean(valid_rmses),                      
                         'test_rmse':np.nanmean(test_rmses),
                         
                         'train_r2':np.nanmean(train_r2s), 
                         'valid_r2':np.nanmean(valid_r2s),                      
                         'test_r2':np.nanmean(test_r2s),                        
                        }
            
            Regression_res.append(final_res)
            
pd.DataFrame(Regression_res).to_csv('./regression_results_gpu5.csv')