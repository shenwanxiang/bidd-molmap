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

def get_pos_weights(trainY):
    """pos_weights: neg_n / pos_n """
    dfY = pd.DataFrame(trainY)
    pos = dfY == 1
    pos_n = pos.sum(axis=0)
    neg = dfY == 0
    neg_n = neg.sum(axis=0)
    pos_weights = (neg_n / pos_n).values
    neg_weights = (pos_n / neg_n).values
    return pos_weights, neg_weights

## random
clintox = dataset.load_ClinTox() #[128] 2 outputs 
sider = dataset.load_SIDER() #[256] #27 outputs
toxcast = dataset.load_ToxCast() # [1024] # 617 outputs
tox21 = dataset.load_Tox21() #[256] #12 outputs

dense_layers_config = {
    'ToxCast': [1024], # 617 outputs
    'SIDER': [128], # 27 outputs
    'Tox21':[256], # 12  outputs
    'ClinTox':[128], #2 outputs
    'ChEMBL': [2048], # 1310 outputs
    'PCBA': [256], #128 outputs
    'MUV':[512], #17 outputs
}

gpuid = 5
result_file = 'sider_toxcast_tox21.csv'

random_seeds = [2, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


file_path = "/raid/shenwanxiang/08_Robustness/dataset_induces/split" #split
tmp_feature_dir = '/raid/shenwanxiang/08_Robustness/tempignore' #feature path



with open(result_file, 'w+') as f:
    f.write('task_name, seed, valid_auc, test_auc\n')

# the dense layers for these multi outputs tasks

res = []
for data in [toxcast, sider, tox21]:
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
    
    Y = pd.DataFrame(Y).fillna(-1).values #mask nan value with -1
    
    dense_layers = dense_layers_config[task_name]
    
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

        pos_weights, neg_weights = get_pos_weights(y_train)
        loss = lambda y_true, y_pred: molmap.model.loss.weighted_cross_entropy(y_true,y_pred, pos_weights, MASK = -1)
    
        clf = MultiLabelEstimator(n_outputs,
                                      fmap_shape1,
                                      fmap_shape2, 
                                      batch_size = 128,
                                      dense_layers = dense_layers,
                                      gpuid = gpuid, 
                                      patience = 30, #speed only
                                      monitor = 'val_auc',
                                      loss = loss,
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