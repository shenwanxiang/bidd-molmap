import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import load, dump


from molmap import dataset
from molmap import loadmap
from molmap import model as molmodel
import molmap

#use GPU, if negative value, CPUs will be used
import tensorflow as tf
import tensorflow_addons as tfa
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"


seed = 777
tqdm.pandas(ascii=True)
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)


def get_attentiveFP_idx(df):
    """ attentiveFP dataset"""
    train, valid,test = load('./Lipop_attentiveFP.data')
    print('training set: %s, valid set: %s, test set %s' % (len(train), len(valid), len(test)))
    train_idx = df[df.smiles.isin(train.smiles)].index
    valid_idx = df[df.smiles.isin(valid.smiles)].index
    test_idx = df[df.smiles.isin(test.smiles)].index
    print('training set: %s, valid set: %s, test set %s' % (len(train_idx), len(valid_idx), len(test_idx)))
    return train_idx, valid_idx, test_idx 

#load dataset
data = dataset.load_Lipop()
df = data.data
Y = data.y

X_name =  './X.data'

if os.path.exists(X_name):
    X = load(X_name)
else:
    mp = loadmap('../descriptor_grid_split.mp')
    X = mp.batch_transform(data.x, n_jobs = 8)
    dump(X, X_name)
    
train_idx, valid_idx, test_idx = get_attentiveFP_idx(df)

trainX = X[train_idx]                
validX = X[valid_idx]
testX  = X[test_idx]

trainY = Y[train_idx]
validY = Y[valid_idx]
testY = Y[test_idx]


import time
start_time = str(time.ctime()).replace(':','-').replace(' ','_')
log_file = data.task_name + '_' + start_time + '.log'
with open(log_file,'a') as f:
    f.write(','.join(['learning_rate', 'weight_decay','batch_size', 'second_units','last_units','best_epoch','best_r2', 'best_rmse'])+'\n')
    
    
def Eva(learning_rate,  weight_decay, batch_size, second_units, last_units):
    
    learning_rate = round(learning_rate, 1)
    weight_decay = round(weight_decay, 1)
    
    batch_size = 2**int(batch_size)
    second_units = 2**int(second_units)
    last_units = 2**int(last_units)
    
    opt = tfa.optimizers.AdamW(weight_decay = 10**-weight_decay, 
                               learning_rate=10**-learning_rate,
                               beta_1=0.9, beta_2=0.999, epsilon=1e-08)  
    
    model = molmodel.net.SinglePathNet(trainX.shape[1:], 
                                       n_outputs=1, dense_layers=[second_units, last_units], 
                                       dense_avf='relu', last_avf='linear')
    
    model.compile(optimizer = opt, loss = 'mse')

    performace = molmodel.cbks.Reg_EarlyStoppingAndPerformance((trainX, trainY), 
                                                               (validX, validY), 
                                                               patience = 20, 
                                                               criteria = 'val_loss')
    model.fit(trainX, trainY, batch_size=batch_size, 
          epochs=800, verbose= 0, shuffle = True, 
          validation_data = (validX, validY), 
          callbacks=[performace]) 
    val_rmses, val_r2s = performace.evaluate(validX, validY)
    val_rmse = round(val_rmses[0], 3)
    val_r2 = round(val_r2s[0], 3)
    
    with open(log_file, 'a') as f:
        f.write(','.join([str(learning_rate), str(weight_decay), str(batch_size), str(second_units), str(last_units)]))
        f.write(','+ str(performace.best_epoch)+',' + str(val_r2) + ',' + str(val_rmse) + '\n')
        
    return -val_rmse


if __name__ == '__main__':
    
    from pyGPGO.covfunc import matern32
    from pyGPGO.acquisition import Acquisition
    from pyGPGO.surrogates.GaussianProcess import GaussianProcess
    from pyGPGO.GPGO import GPGO
    
    cov = matern32()
    gp = GaussianProcess(cov)
    acq = Acquisition(mode='UCB')
    param = {
             'learning_rate': ('cont', [2, 5]),
             'weight_decay': ('cont', [2, 6]),
             'batch_size': ('int', [2, 8]),
             'second_units': ('int',[6, 9]),
             'last_units': ('int', [4, 8])
             }
    
    gpgo = GPGO(gp, acq, Eva, param)
    gpgo.run(max_iter=30,init_evals=2)