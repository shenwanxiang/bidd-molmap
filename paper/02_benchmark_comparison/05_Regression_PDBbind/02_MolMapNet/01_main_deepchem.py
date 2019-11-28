from molmap import dataset
from molmap import loadmap
import molmap

import matplotlib.pyplot as plt
from joblib import dump, load
from tqdm import tqdm
import pandas as pd
tqdm.pandas(ascii=True)

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import MaxPool2D, GlobalMaxPool2D, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, Concatenate,Flatten, Dense, Dropout


#use the second GPU, if negative value, CPUs will be used
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def Inception(inputs, units = 8, strides = 1):
    """
    naive google inception block
    """
    x1 = Conv2D(units, 5, padding='same', activation = 'relu', strides = strides)(inputs)
    x2 = Conv2D(units, 3, padding='same', activation = 'relu', strides = strides)(inputs)
    x3 = Conv2D(units, 1, padding='same', activation = 'relu', strides = strides)(inputs)
    outputs = Concatenate()([x1, x2, x3])    
    return outputs


def SinglePathClassificationModel(molmap_shape,  n_outputs = 1, strides = 1):
    """molmap_shape: w, h, c"""
    
    assert len(molmap_shape) == 3
    inputs = Input(molmap_shape)
    
    conv1 = Conv2D(48, 13, padding = 'same', activation='relu', strides = 1)(inputs)
    
    conv1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(conv1) #p1
    
    incept1 = Inception(conv1, strides = 1, units = 32)
    
    incept1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(incept1) #p2
    
    incept2 = Inception(incept1, strides = 1, units = 64)
    
    #flatten
    flat1 = GlobalMaxPool2D()(incept2)   
    d1 = Dense(128,activation='relu')(flat1)
    d1 = Dense(64,activation='relu')(d1)
    outputs = Dense(n_outputs,activation='linear')(d1)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def DoublePathClassificationModel(molmap1_size, molmap2_size, n_outputs = 1):
    
    ## first inputs
    d_inputs1 = Input(molmap1_size)
    d_conv1 = Conv2D(48, 13, padding = 'same', activation='relu', strides = 1)(d_inputs1)
    d_pool1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(d_conv1) #p1
    d_incept1 = Inception(d_pool1, strides = 1, units = 32)
    d_pool2 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(d_incept1) #p2
    d_incept2 = Inception(d_pool2, strides = 1, units = 64)
    d_flat1 = GlobalMaxPool2D()(d_incept2)

    
    ## second inputs
    f_inputs1 = Input(molmap2_size)
    f_conv1 = Conv2D(48, 13, padding = 'same', activation='relu', strides = 1)(f_inputs1)
    f_pool1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(f_conv1) #p1
    f_incept1 = Inception(f_pool1, strides = 1, units = 32)
    f_pool2 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(f_incept1) #p2
    f_incept2 = Inception(f_pool2, strides = 1, units = 64)
    f_flat1 = GlobalMaxPool2D()(f_incept2)    
    
    ## concat
    merge = Concatenate()([d_flat1, f_flat1]) 
    d1 = Dense(128,activation='relu')(merge)
    d1 = Dense(64,activation='relu')(d1)
    outputs = Dense(n_outputs, activation='linear')(d1)
    
    model = tf.keras.Model(inputs=[d_inputs1, f_inputs1], outputs=outputs)
    
    return model




df_train = pd.read_csv('../train.csv', index_col = 0)
df_valid = pd.read_csv('../valid.csv', index_col = 0)
df_test = pd.read_csv('../test.csv', index_col = 0)

df = df_train.append(df_valid).append(df_test)
df = df.reset_index(drop=True)

train_idx = df.index[:len(df_train)]
valid_idx = df.index[len(df_train): len(df_train)+len(df_valid)]
test_idx = df.index[len(df_train)+len(df_valid): len(df_train)+len(df_valid) + len(df_test)]

task = '-logKd/Ki'
Y = df[[task]].values

trainY = Y[train_idx]
validY = Y[valid_idx]
testY = Y[test_idx]


print(len(df_train), len(df_valid), len(df_test))



# calculate feature
X1_name =  './descriptor_grid_split.data'
X2_name =  './fingerprint_grid_split.data'

if os.path.exists(X1_name):
    X1 = load(X1_name)
else:
    mp1 = loadmap('../../descriptor_grid_split.mp')
    X1 = mp1.batch_transform(df.smiles, n_jobs = 8)
    dump(X1, X1_name)

if os.path.exists(X2_name):
    X2 = load(X2_name)
else:
    mp2 = loadmap('../../fingerprint_grid_split.mp')
    X2 = mp2.batch_transform(df.smiles, n_jobs = 8)
    dump(X2, X2_name)





if __name__ == '__main__':
    
    
    epochs = 500
    patience = 20
    batch_size = 128
    learning_rate=0.0001

    model_types = ['descriptor', 'fingerprint', 'both']    
    
    from cbks import RegressionPerformance, EarlyStoppingAtMinLoss
    
    for mtype in model_types:
        results = []
        for seed in [7, 77, 777]:
            np.random.seed(seed)
            tf.set_random_seed(seed)                           
 
            molmap1_size = X1.shape[1:]
            molmap2_size = X2.shape[1:]

            if mtype == 'descriptor':
                model = SinglePathClassificationModel(molmap1_size, n_outputs = Y.shape[1])
                trainX = X1[train_idx]  
                validX = X1[valid_idx]
                testX = X1[test_idx]

                
            elif mtype == 'fingerprint':
                model = SinglePathClassificationModel(molmap2_size, n_outputs = Y.shape[1])
                trainX = X2[train_idx] 
                validX = X2[valid_idx]
                testX = X2[test_idx]
              

            else:
                model = DoublePathClassificationModel(molmap1_size, molmap2_size, n_outputs = Y.shape[1])
                trainX = (X1[train_idx], X2[train_idx])    
                validX = (X1[valid_idx], X2[valid_idx])   
                testX = (X1[test_idx], X2[test_idx])                    

                

            earlystop = EarlyStoppingAtMinLoss(patience=patience, criteria = 'val_loss')
            performace = RegressionPerformance((trainX, trainY), (validX, validY))
            lr = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #
            model.compile(optimizer = lr, loss = 'mse')
            
            model.fit(trainX, trainY, batch_size=batch_size, 
                      epochs=epochs, verbose= 0, shuffle = True, 
                      validation_data = (validX, validY), 
                      callbacks=[performace, earlystop]) 

            train_perf = performace.evaluate(trainX, trainY)
            valid_perf = performace.evaluate(validX, validY)            
            test_perf = performace.evaluate(testX, testY)
       

            best_train_pred_y = list(performace.model.predict(trainX).reshape(-1,))
            best_valid_pred_y = list(performace.model.predict(validX).reshape(-1,))              
            best_test_pred_y = list(performace.model.predict(testX).reshape(-1,))            

            
            
            best_epoch = earlystop.best_epoch
            paras = model.count_params()
            
            
            final_res = {'train_rmse':np.nanmean(train_perf[0]), 
                         'valid_rmse':np.nanmean(valid_perf[0]), 
                         'test_rmse':np.nanmean(test_perf[0]),                          

                         'train_pred_y':best_train_pred_y,
                         'valid_pred_y':best_valid_pred_y,                         
                         'test_pred_y':best_test_pred_y,

                         '# trainable params': paras,
                         'random_seed':seed, 
                         'best_epoch': best_epoch,
                         'batch_size':batch_size,
                         'lr': learning_rate,
                        }
            results.append(final_res)
        
        pd.DataFrame(results).to_csv('../results/results_MolMapNet_%s.csv' % mtype)
