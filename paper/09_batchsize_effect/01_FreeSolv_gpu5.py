#!/usr/bin/env python
# coding: utf-8

from molmap.model import RegressionEstimator, MultiClassEstimator, MultiLabelEstimator
from molmap import loadmap, dataset
from molmap.show import imshow_wrap

from sklearn.utils import shuffle 
from joblib import load, dump
import numpy as np
import pandas as pd
import os


def Rdsplit(df, random_state = 1, split_size = [0.8, 0.1, 0.1]):

    base_indices = np.arange(len(df)) 
    base_indices = shuffle(base_indices, random_state = random_state) 
    nb_test = int(len(base_indices) * split_size[2]) 
    nb_val = int(len(base_indices) * split_size[1]) 
    test_idx = base_indices[0:nb_test] 
    valid_idx = base_indices[(nb_test):(nb_test+nb_val)] 
    train_idx = base_indices[(nb_test+nb_val):len(base_indices)] 
    
    print(len(train_idx), len(valid_idx), len(test_idx)) 
    
    return train_idx, valid_idx, test_idx 



data = dataset.load_FreeSolv()

task_name = data.task_name
smiles = data.x
df = data.data


# In[5]:


from chembench import load_data
_, induces = load_data(task_name)


# In[6]:


mp1 = loadmap('../descriptor.mp')
mp2 = loadmap('../fingerprint.mp')


# In[7]:


tmp_feature_dir = '/raid/shenwanxiang/09_batchsize_effect/tempignore'
if not os.path.exists(tmp_feature_dir):
    os.makedirs(tmp_feature_dir)
    
X1_name = os.path.join(tmp_feature_dir, 'X1_%s.data' % task_name)
X2_name = os.path.join(tmp_feature_dir, 'X2_%s.data' % task_name)
if not os.path.exists(X1_name):
    X1 = mp1.batch_transform(smiles, n_jobs = 8)
    dump(X1, X1_name)
else:
    X1 = load(X1_name)

if not os.path.exists(X2_name): 
    X2 = mp2.batch_transform(smiles, n_jobs = 8)
    dump(X2, X2_name)
else:
    X2 = load(X2_name)


# In[8]:


fmap_shape1 = X1.shape[1:] 
fmap_shape2 = X2.shape[1:] 


# In[9]:


Y = data.y
n_outputs = Y.shape[1]


# induces = []
# for random_state in range(10):
#     induce = Rdsplit(data.data, random_state)
#     induces.append(induce)

# from sklearn.model_selection import KFold

# induces = []
# for random_state in [2, 32, 128, 512, 1024]:
#     kf = KFold(n_splits=5, shuffle = True, random_state=random_state)
#     for tr, ts in kf.split(range(len(df))):
#         induces.append([tr, ts])


batch_sizes  = [8, 64, 128]


res = []
for batch_size in batch_sizes:

    c1 = []
    for idx in induces:

        train_idx, valid_idx, test_idx  = idx

        X = (X1[train_idx], X2[train_idx])
        y = Y[train_idx]

        X_valid = (X1[valid_idx], X2[valid_idx])
        y_valid = Y[valid_idx]

        X_test = (X1[test_idx], X2[test_idx])
        y_test = Y[test_idx]    

        clf = RegressionEstimator(n_outputs=n_outputs,  
                                  fmap_shape1 = fmap_shape1, 
                                  fmap_shape2 = fmap_shape2,
                                  batch_size = batch_size,
                                  dense_layers = [256, 128, 32],
                                  gpuid = 5,
                                  epochs = 800,
                                  patience = 50,
                                 ) 

        clf.fit(X, y, X_valid, y_valid)

        rmse_list, r2_list = clf._performance.evaluate(X_test,y_test)
        rmse_list_, r2_list_ = clf._performance.evaluate(X_valid,y_valid)
        
        dfp = pd.DataFrame(clf._performance.history)
        dfp = dfp.set_index('epoch')        
        c1.append({'batch_size': batch_size, 
                   'process':dfp, 
                   'valid_rmse': rmse_list_[0], 
                   'valid_r2':r2_list_[0],
                   
                   'test_rmse': rmse_list[0], 
                   'test_r2':r2_list[0]})
        
    res.append(c1)



dump((batch_sizes, res), './%s.res' % task_name)
for i in res:
    x = pd.DataFrame(i).test_rmse.mean()    
    print(x)



