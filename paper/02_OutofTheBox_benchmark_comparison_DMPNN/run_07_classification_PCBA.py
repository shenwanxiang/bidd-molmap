#!/usr/bin/env python
# coding: utf-8

# In[1]:


from molmap import model as molmodel
import molmap
import matplotlib.pyplot as plt

import pandas as pd
from tqdm import tqdm
from joblib import load, dump
tqdm.pandas(ascii=True)
import numpy as np

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="6"
  

np.random.seed(123)
tf.compat.v1.set_random_seed(123)

#tmp_feature_dir = './tmpignore'
tmp_feature_dir = '/raid/shenwanxiang/tempignore'

if not os.path.exists(tmp_feature_dir):
    os.makedirs(tmp_feature_dir)


# In[2]:


mp1 = molmap.loadmap('../descriptor.mp')
mp2 = molmap.loadmap('../fingerprint.mp')


# In[3]:


task_name = 'PCBA'
from chembench import load_data
df, induces = load_data(task_name)
print(len(induces[0][0]), len(induces[0][1]), len(induces[0][2]), df.shape)



nan_idx = df[df.smiles.isna()].index.to_list()


MASK = -1
smiles_col = df.columns[0]
values_col = df.columns[1:]
Y = df[values_col].astype('float').fillna(MASK).values
if Y.shape[1] == 0:
    Y = Y.reshape(-1, 1)
    Y = Y.astype('float32')


batch = 30
xs = np.array_split(df.smiles.to_list(), batch)



X1_name_all = os.path.join(tmp_feature_dir, 'X1_%s.data' % (task_name))
X2_name_all = os.path.join(tmp_feature_dir, 'X2_%s.data' % (task_name))

if os.path.exists(X1_name_all) & os.path.exists(X2_name_all):
    X1 = load(X1_name_all)
    X2 = load(X2_name_all)
else:
    ## descriptors
    X1s = []
    for i, batch_smiles in tqdm(enumerate(xs), ascii=True):
        ii = str(i).zfill(2)
        X1_name = os.path.join(tmp_feature_dir, 'X1_%s_%s.data' % (task_name, ii))
        print('save to %s' % X1_name)
        if not os.path.exists(X1_name):
            X1 = mp1.batch_transform(batch_smiles, n_jobs = 8)
            X1 = X1.astype('float32')
            dump(X1, X1_name)
            
        else:
            X1 = load(X1_name)
        X1s.append(X1)
        del X1
        
    X1 = np.concatenate(X1s)
    del X1s
    
    dump(X1, X1_name_all)
    
    ## fingerprint
    X2s = []      
    for i, batch_smiles in tqdm(enumerate(xs), ascii=True):
        ii = str(i).zfill(2)
        X2_name = os.path.join(tmp_feature_dir, 'X2_%s_%s.data' % (task_name, ii))
        if not os.path.exists(X2_name):
            X2 = mp2.batch_transform(batch_smiles, n_jobs = 8)
            X2 = X2.astype('float32')
            dump(X2, X2_name)
            
        else:
            X2 = load(X2_name)
            
        X2s.append(X2)
        del X2
        
    X2 = np.concatenate(X2s)
    del X2s
    dump(X2, X2_name_all)


# In[10]:


molmap1_size = X1.shape[1:]
molmap2_size = X2.shape[1:]
X1.shape


# In[11]:


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

prcs_metrics = ['MUV', 'PCBA']


# In[12]:


epochs = 800
patience = 10 #early stopping, dual to large computation cost, the larger dataset  set small waitig patience for early stopping
dense_layers = [512]  #128 outputs

batch_size = 128
lr = 1e-4
weight_decay = 0

monitor = 'val_auc'
dense_avf = 'relu'
last_avf = None #sigmoid in loss

if task_name in prcs_metrics:
    metric = 'PRC'
else:
    metric = 'ROC'


# In[ ]:

results = []
for i, split_idxs in enumerate(induces):

    train_idx, valid_idx, test_idx = split_idxs
    
    train_idx = list(set(train_idx) - set(nan_idx))
    valid_idx = list(set(valid_idx) - set(nan_idx))
    test_idx = list(set(test_idx) - set(nan_idx))
    
    
    print(len(train_idx), len(valid_idx), len(test_idx))

    trainX = (X1[train_idx], X2[train_idx])
    trainY = Y[train_idx]

    validX = (X1[valid_idx], X2[valid_idx])
    validY = Y[valid_idx]

    testX = (X1[test_idx], X2[test_idx])
    testY = Y[test_idx]            

    pos_weights, neg_weights = get_pos_weights(trainY)
    #loss = lambda y_true, y_pred: molmodel.loss.weighted_cross_entropy(y_true,y_pred, pos_weights, MASK = MASK)
    loss = molmodel.loss.cross_entropy
    model = molmodel.net.DoublePathNet(molmap1_size, molmap2_size, 
                                       n_outputs=Y.shape[-1], 
                                       dense_layers=dense_layers, 
                                       dense_avf = dense_avf, 
                                       last_avf=last_avf)

    opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #
    #import tensorflow_addons as tfa
    #opt = tfa.optimizers.AdamW(weight_decay = 0.001,learning_rate=0.001)
    model.compile(optimizer = opt, loss = loss)

    performance = molmodel.cbks.CLA_EarlyStoppingAndPerformance((trainX, trainY), 
                                                                   (validX, validY), 
                                                                   patience = patience, 
                                                                   criteria = monitor,
                                                                   metric = metric,
                                                                  )
    model.fit(trainX, trainY, batch_size=batch_size, 
          epochs=epochs, verbose= 0, shuffle = True, 
          validation_data = (validX, validY), 
          callbacks=[performance]) 


    best_epoch = performance.best_epoch
    trainable_params = model.count_params()
    
    train_aucs = performance.evaluate(trainX, trainY)            
    valid_aucs = performance.evaluate(validX, validY)            
    test_aucs = performance.evaluate(testX, testY)

    final_res = {
                     'task_name':task_name,            
                     'train_auc':np.nanmean(train_aucs), 
                     'valid_auc':np.nanmean(valid_aucs),                      
                     'test_auc':np.nanmean(test_aucs), 
                     'metric':metric,
                     '# trainable params': trainable_params,
                     'best_epoch': best_epoch,
                     'batch_size':batch_size,
                     'lr': lr,
                     'weight_decay':weight_decay
                    }
    
    print(final_res)
    del model, trainX, validX, testX
    
    results.append(final_res)


pd.DataFrame(results).to_csv('./results/%s.lr0.001.csv' % task_name)

