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

gpuid = 4
result_file = 'bace_bbbp_hiv.csv'

random_seeds = [2, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


file_path = "/raid/shenwanxiang/08_Robustness/dataset_induces/split" #split
tmp_feature_dir = '/raid/shenwanxiang/08_Robustness/tempignore' #feature path

bace = dataset.load_BACE()
bbbp = dataset.load_BBBP()
hiv = dataset.load_HIV()

with open(result_file, 'w+') as f:
    f.write('task_name, seed, valid_auc, test_auc\n')
    
    
res = []
for data in [bace, bbbp, hiv]:
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
        
        if task_name == 'HIV':
            patience = 20 #speed reason only, early topping looking ahead
            
        else:
            patience = 50

        clf = MultiLabelEstimator(n_outputs,
                                      fmap_shape1,
                                      fmap_shape2, 
                                      batch_size = 128,
                                      dense_layers = [256, 128, 32],
                                      gpuid = gpuid, 
                                      patience = patience,
                                      monitor = 'val_auc',
                                     )             
        clf.fit(X_train,y_train, X_valid, y_valid)

        train_aucs = clf._performance.evaluate(X_train,y_train)
        valid_aucs = clf._performance.evaluate(X_valid,y_valid)            
        test_aucs = clf._performance.evaluate(X_test,y_test)
        
        train_auc = np.nanmean(train_aucs)
        valid_auc = np.nanmean(valid_aucs)
        test_auc = np.nanmean(test_aucs)
        
        final_res = {'seed': seed,
                     "task_name": task_name,
                     'train_auc':train_auc, 
                     'valid_auc':valid_auc,                      
                     'test_auc':test_auc,}
        print(final_res)
        
        with open(result_file, 'a+') as f:
            f.write('%s, %s, %s, %s\n' % (task_name, seed, valid_auc, test_auc))

        res.append(final_res)
        
pd.DataFrame(res).to_csv(result_file + '.bak.csv')