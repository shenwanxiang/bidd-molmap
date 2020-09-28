import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import load, dump
import time

from molmap import dataset
from molmap import loadmap
from molmap import model as molmodel
import molmap

#use GPU, if negative value, CPUs will be used
import tensorflow as tf
#import tensorflow_addons as tfa
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

## fix random seed to get repeatale results
seed = 123
tqdm.pandas(ascii=True)
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)


mp1 = molmap.loadmap('../descriptor.mp')
mp2 = molmap.loadmap('../fingerprint.mp')


task_name = 'Tox21'
from chembench import load_data
df, induces = load_data(task_name)
i = 0


MASK = -1
smiles_col = df.columns[0]
values_col = df.columns[1:]
Y = df[values_col].astype('float').fillna(MASK).values
if Y.shape[1] == 0:
    Y = Y.reshape(-1, 1)
    
tmp_feature_dir = './tmpignore'

X1_name = os.path.join(tmp_feature_dir, 'X1_%s.data' % task_name)
X2_name = os.path.join(tmp_feature_dir, 'X2_%s.data' % task_name)
if not os.path.exists(X1_name):
    X1 = mp1.batch_transform(df.smiles, n_jobs = 8)
    dump(X1, X1_name)
else:
    X1 = load(X1_name)

if not os.path.exists(X2_name): 
    X2 = mp2.batch_transform(df.smiles, n_jobs = 8)
    dump(X2, X2_name)
else:
    X2 = load(X2_name)
    
train_idx, valid_idx, test_idx = induces[i]

molmap1_size = X1.shape[1:]
molmap2_size = X2.shape[1:]
trainY = Y[train_idx]
validY = Y[valid_idx]

epochs = 100
patience = 100000 #early stopping, 100 epochs to select best
dense_layers = [256, 128]
batch_size = 128
lr = 1e-4
weight_decay = 0

monitor = 'val_auc'
dense_avf = 'relu'
last_avf = None #sigmoid in loss


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



def Eva(n_neighbors,  min_dist):
    
    min_dist = min_dist    
    n_neighbors = n_neighbors

    print({'min_dist':min_dist, 'n_neighbors':n_neighbors})
    
    mp1_new =  loadmap('../descriptor.mp')
    mp1_new.fit(method = 'umap', min_dist = min_dist, n_neighbors = n_neighbors)    
    
    mp2_new =  loadmap('../fingerprint.mp')
    mp2_new.fit(method = 'umap', min_dist = min_dist, n_neighbors = n_neighbors)

    X1_new = mp1.rearrangement(X1, mp1_new)
    X2_new = mp2.rearrangement(X2, mp2_new)

    trainX = (X1_new[train_idx], X2_new[train_idx])
    validX = (X1_new[valid_idx], X2_new[valid_idx])

    pos_weights, neg_weights = get_pos_weights(trainY)
    loss = lambda y_true, y_pred: molmodel.loss.weighted_cross_entropy(y_true,y_pred, pos_weights, MASK = -1)

    model = molmodel.net.DoublePathNet(molmap1_size, molmap2_size, 
                                       n_outputs=Y.shape[-1], 
                                       dense_layers=dense_layers, 
                                       dense_avf = dense_avf, 
                                       last_avf=last_avf)

    opt = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #
    #import tensorflow_addons as tfa
    #opt = tfa.optimizers.AdamW(weight_decay = 0.1,learning_rate=0.001,beta1=0.9,beta2=0.999, epsilon=1e-08)
    model.compile(optimizer = opt, loss = loss)

    performance = molmodel.cbks.CLA_EarlyStoppingAndPerformance((trainX, trainY), 
                                                                   (validX, validY), 
                                                                   patience = patience, 
                                                                   criteria = monitor,
                                                                   metric = 'ROC',
                                                                  )
    model.fit(trainX, trainY, batch_size=batch_size, 
          epochs=epochs, verbose= 0, shuffle = True, 
          validation_data = (validX, validY), 
          callbacks=[performance]) 

    best_epoch = performance.best_epoch
    train_aucs = performance.evaluate(trainX, trainY)            
    valid_aucs = performance.evaluate(validX, validY)            

    train_best_auc = np.nanmean(train_aucs)    
    valid_best_auc = np.nanmean(valid_aucs)
    
    dfx = pd.DataFrame(performance.history)
    valid_best_loss = dfx[dfx.epoch == performance.best_epoch].val_loss.iloc[0]
    
    with open(log_file, 'a') as f:
        f.write(','.join([str(min_dist), str(n_neighbors),str(valid_best_loss), str(valid_best_auc), 
                          str(train_best_auc), str(best_epoch)]) + '\n')
        
    return [valid_best_auc, train_best_auc, best_epoch]

if __name__ == '__main__':
    
    flag = 'search'
    start_time = str(time.ctime()).replace(':','-').replace(' ','_')
    log_file = task_name + '_' + flag + '_' + start_time + '.log'

    with open(log_file,'a') as f:
        f.write(','.join(['min_dist', 'n_neighbors', 'valid_best_loss','valid_best_auc', 
                          'train_best_auc', 'best_epoch'])+'\n')
    
    n_neighbors_list = [10, 20, 30, 40, 50, 60, 70, 80, 90,  100]
    min_dist_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for n_neighbors in n_neighbors_list:
        for min_dist in min_dist_list:
            output = Eva(n_neighbors,  min_dist)