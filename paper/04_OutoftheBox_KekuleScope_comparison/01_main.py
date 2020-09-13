from molmap.extend.kekulescope import dataset
from molmap.extend.kekulescope import featurizer
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



def split(df, random_state = 123, split_size = [0.7, 0.15, 0.15]):
    from sklearn.utils import shuffle 
    import numpy as np
    base_indices = np.arange(len(df)) 
    base_indices = shuffle(base_indices, random_state = random_state) 
    nb_test = int(len(base_indices) * split_size[2]) 
    nb_val = int(len(base_indices) * split_size[1]) 
    test_idx = base_indices[0:nb_test] 
    valid_idx = base_indices[(nb_test):(nb_test+nb_val)] 
    train_idx = base_indices[(nb_test+nb_val):len(base_indices)] 
    
    print(len(train_idx), len(valid_idx), len(test_idx)) 
    
    return train_idx, valid_idx, test_idx 





if __name__ == '__main__':
    
    
    epochs = 500
    patience = 30
    batch_size = 128
    lr=0.0001
    data_split_seed = 1
    
    mp1 = molmap.loadmap('../descriptor_grid_split.mp')
    mp2 = molmap.loadmap('../fingerprint_grid_split.mp')

    for cell_line in dataset.cell_lines:
        
        df = dataset.load_data(cell_line)
        df = df[~df.pIC50.isna()].reset_index(drop=True)
        train_idx, valid_idx, test_idx = split(df, random_state = data_split_seed)
        Y = df['pIC50'].astype('float').values.reshape(-1,1)
        
        X1_name = 'X1_%s.data' % cell_line
        X2_name = 'X2_%s.data' % cell_line
        if not os.path.exists(X1_name):
            X1 = mp1.batch_transform(df.smiles, n_jobs = 8)
            dump(X1, X1_name)
        else:
            X1 = load(X1_name)

        if not os.path.exists(X2_name): 
            X2 = mp2.batch_transform(df.smiles)
            dump(X2, X2_name)
        else:
            X2 = load(X2_name)
        
        trainX = (X1[train_idx], X2[train_idx])
        trainY = Y[train_idx]
        validX = (X1[valid_idx], X2[valid_idx])
        validY = Y[valid_idx]
        testX = (X1[test_idx], X2[test_idx])
        testY = Y[test_idx]
        
        molmap1_size = X1.shape[1:]
        molmap2_size = X2.shape[1:]
        
        results = []
        
        for rdseed in [7, 77, 777]:
            np.random.seed(rdseed)
            tf.compat.v1.set_random_seed(rdseed)
            
            performace = molmodel.cbks.Reg_EarlyStoppingAndPerformance((trainX, trainY), 
                                                                       (validX, validY), 
                                                                       patience = 20, 
                                                                       criteria = 'val_loss')

            model = molmodel.net.DoublePathNet(molmap1_size, molmap2_size, 
                                               n_outputs=1, dense_layers=[256, 128, 32], 
                                               dense_avf='relu', last_avf='linear')

            opt = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #
            #import tensorflow_addons as tfa
            #opt = tfa.optimizers.AdamW(weight_decay = 0.1,learning_rate=0.001,beta1=0.9,beta2=0.999, epsilon=1e-08)
            model.compile(optimizer = opt, loss = 'mse')
            model.fit(trainX, trainY, batch_size=batch_size, 
                  epochs=800, verbose= 0, shuffle = True, 
                  validation_data = (validX, validY), 
                  callbacks=[performace]) 
            
            best_epoch = performace.best_epoch
            trainable_params = model.count_params()
            
            train_rmses, train_r2s = performace.evaluate(trainX, trainY)            
            valid_rmses, valid_r2s = performace.evaluate(validX, validY)            
            test_rmses, test_r2s = performace.evaluate(testX, testY)

            
            final_res = {
                         'cell_line':cell_line,            
                         'train_rmse':np.nanmean(train_rmses), 
                         'valid_rmse':np.nanmean(valid_rmses),                      
                         'test_rmse':np.nanmean(test_rmses), 

                         'train_r2':np.nanmean(train_r2s), 
                         'valid_r2':np.nanmean(valid_r2s),                      
                         'test_r2':np.nanmean(test_r2s), 

                         '# trainable params': trainable_params,
                         'random_seed':rdseed, 
                         'best_epoch': best_epoch,
                         'batch_size':batch_size,
                         'lr': lr,
                         'data_split_seed':data_split_seed,
                        }

            results.append(final_res)
        pd.DataFrame(results).to_csv('./results/results_molmapnet_%s.csv' % cell_line)