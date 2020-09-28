#### !/usr/bin/env python
# coding: utf-8
from molmap.model import RegressionEstimator, MultiClassEstimator, MultiLabelEstimator
from molmap import loadmap, dataset
from molmap.show import imshow_wrap
import molmap

from sklearn.utils import shuffle 
from joblib import load, dump
import numpy as np
import pandas as pd
import os

from chembench import dataset


## random
freesolv = dataset.load_FreeSolv()
esol = dataset.load_ESOL()
lipop = dataset.load_Lipop()
malaria = dataset.load_Malaria()


gpuid = 6
result_file = 'freesolv_esol_malaria.csv'

random_seeds = [2, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


file_path = "/raid/shenwanxiang/08_Robustness/dataset_induces/split" #split
tmp_feature_dir = '/raid/shenwanxiang/08_Robustness/tempignore' #feature path



with open(result_file, 'w+') as f:
    f.write('task_name, seed, valid_rmse, test_rmse\n')

# the dense layers for these multi outputs tasks

res = []
for data in [freesolv, esol, lipop, malaria]:
    task_name = data.task_name
    task_type = data.task_type

    X1_name = os.path.join(tmp_feature_dir, 'X1_%s.data' % task_name)
    X2_name = os.path.join(tmp_feature_dir, 'X2_%s.data' % task_name)
    X1 = load(X1_name)
    X2 = load(X2_name)

    fmap_shape1 = X1.shape[1:] 
    fmap_shape2 = X2.shape[1:] 

    Y = data.y
    df = data.df
    n_outputs = Y.shape[1]
    
    batch_size = 128
    
    if len(Y) < 2000:
        batch_size = 16 #set a smaller batch size for low-data task
        
    for seed in random_seeds:

        train_path = os.path.join(file_path, task_name,"%s" % seed, "train.csv")
        valid_path = os.path.join(file_path, task_name,"%s" % seed, "val.csv")
        test_path = os.path.join(file_path, task_name,"%s" % seed, "test.csv")

        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        test_df = pd.read_csv(test_path)

        train_idx = df[df.smiles.isin(train_df.smiles)].index.tolist()
        valid_idx = df[df.smiles.isin(valid_df.smiles)].index.tolist()
        test_idx = df[df.smiles.isin(test_df.smiles)].index.tolist()

        print(len(train_idx), len(valid_idx), len(test_idx))

        X_train = (X1[train_idx], X2[train_idx])
        y_train = Y[train_idx]

        X_valid = (X1[valid_idx], X2[valid_idx])
        y_valid = Y[valid_idx]

        X_test = (X1[test_idx], X2[test_idx])
        y_test = Y[test_idx]    

        clf = RegressionEstimator(n_outputs,
                                      fmap_shape1,
                                      fmap_shape2, 
                                      batch_size = batch_size,
                                      dense_layers = [256, 128, 32],
                                      gpuid = gpuid, 
                                 )             
        clf.fit(X_train,y_train, X_valid, y_valid)

        train_rmses, train_r2s = clf._performance.evaluate(X_train,y_train)
        valid_rmses, valid_r2s = clf._performance.evaluate(X_valid,y_valid)            
        test_rmses, test_r2s = clf._performance.evaluate(X_test,y_test)
        
        train_rmse = np.nanmean(train_rmses)
        valid_rmse = np.nanmean(valid_rmses)
        test_rmse = np.nanmean(test_rmses)
        
        final_res = {'seed': seed,
                     "task_name": task_name,
                     'train_rmse':train_rmse, 
                     'valid_rmse':valid_rmse,                      
                     'test_rmse':test_rmse,}
        print(final_res)
        
        with open(result_file, 'a+') as f:
            f.write('%s, %s, %s, %s\n' % (task_name, seed, valid_rmse, test_rmse))

        res.append(final_res)
        
pd.DataFrame(res).to_csv(result_file + '.bak.csv')