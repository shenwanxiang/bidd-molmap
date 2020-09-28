from molmap.model import RegressionEstimator, MultiClassEstimator, MultiLabelEstimator
from molmap import loadmap, feature
from molmap.show import imshow_wrap
from chembench import load_data, dataset

from sklearn.utils import shuffle 
from joblib import load, dump
import numpy as np
import pandas as pd
import os


bitsinfo = feature.fingerprint.Extraction().bitsinfo
fp_types = bitsinfo.Subtypes.unique()


tox21 = dataset.load_Tox21()
toxcast = dataset.load_ToxCast()
sider = dataset.load_SIDER()
clintox = dataset.load_ClinTox()
muv = dataset.load_MUV()

datasets = [muv, tox21, toxcast, sider, clintox]
MASK = -1


tmp_feature_dir = '/raid/shenwanxiang/10_FP_effect/tempignore'
if not os.path.exists(tmp_feature_dir):
    os.makedirs(tmp_feature_dir)
    
mps = []
fp_save_folder = '/raid/shenwanxiang/FP_maps'
for fp_type in fp_types:
    mp = loadmap(os.path.join(fp_save_folder, '%s.mp' % fp_type))
    mps.append(mp)
    
    
classification_res = []
## classification
for data in datasets:
    
    task_name = data.task_name
    task_type = data.task_type
    _, induces = load_data(task_name)
    smiles = data.x
    Y = pd.DataFrame(data.y).fillna(MASK).values

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

            clf = MultiLabelEstimator(n_outputs=n_outputs,  
                                      fmap_shape1 = fmap_shape1, 
                                      dense_layers = [128, 64],
                                      gpuid = 7,
                                     ) 
            
            clf.fit(X, y, X_valid, y_valid)
            
            train_aucs = clf._performance.evaluate(X,y)
            valid_aucs = clf._performance.evaluate(X_valid,y_valid)            
            test_aucs = clf._performance.evaluate(X_test,y_test)
            

            final_res = {'task_name':task_name, 
                         'fp_type': fp_type, 
                         'repeat_id':i,
                         'train_auc':np.nanmean(train_aucs), 
                         'valid_auc':np.nanmean(valid_aucs),                      
                         'test_auc':np.nanmean(test_aucs)}
            
            classification_res.append(final_res)
            
pd.DataFrame(classification_res).to_csv('./classification_results_gpu7.csv')