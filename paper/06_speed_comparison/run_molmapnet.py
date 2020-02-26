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

import os
os.environ["CUDA_VISIBLE_DEVICES"]="6"


seed = 777
tqdm.pandas(ascii=True)
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

#load dataset
data = dataset.load_ESOL()
df = data.data
Y = data.y

valid_idx = df.sample(frac = 0.2).index.to_list()
train_idx = list(set(df.index) - set(valid_idx))

batch_size = 200

res = []
for epochs in [1, 10, 50, 100, 150, 300, 500]:

    start = time.time()

    mp = loadmap('../descriptor.mp')
    X = mp.batch_transform(data.x, n_jobs = 10)

    trainX = X[train_idx]                
    validX = X[valid_idx]
    trainY = Y[train_idx]
    validY = Y[valid_idx]

    performace = molmodel.cbks.Reg_EarlyStoppingAndPerformance((trainX, trainY), 
                                                               (validX, validY), 
                                                               patience = 10000000000, 
                                                               criteria = 'val_loss')

    model = molmodel.net.SinglePathNet(X.shape[1:], 
                                       n_outputs=1, dense_layers=[128, 32], 
                                       dense_avf='relu', last_avf='linear')

    opt = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #
    model.compile(optimizer = opt, loss = 'mse')
    model.fit(trainX, trainY, batch_size=batch_size, 
          epochs=epochs, verbose= 0, shuffle = True, 
          validation_data = (validX, validY), 
          callbacks=[performace]) 

    end = time.time()
    total = end - start
    
    print('total epoch: %s, total time: %s' % (epochs, total))
    res.append([epochs, total])
    
x = pd.DataFrame(res, columns = ['epochs', 'total_time(s)'])
x['average_time(s)'] =  x['total_time(s)'] / x['epochs']
x.to_csv('./molmapnet.csv')