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
#import tensorflow_addons as tfa
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

## fix random seed to get repeatale results
seed = 123
tqdm.pandas(ascii=True)
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)



def get_attentiveFP_idx(df):
    """ attentiveFP dataset"""
    train, valid,test = load('../ESOL_train_valid_test.data')
    print('training set: %s, valid set: %s, test set %s' % (len(train), len(valid), len(test)))
    train_idx = df[df.smiles.isin(train.smiles)].index
    valid_idx = df[df.smiles.isin(valid.smiles)].index
    test_idx = df[df.smiles.isin(test.smiles)].index
    print('training set: %s, valid set: %s, test set %s' % (len(train_idx), len(valid_idx), len(test_idx)))
    return train_idx, valid_idx, test_idx 

#load dataset
data = dataset.load_ESOL()
df = data.data
Y = data.y


task_name = 'ESOL'
tmp_feature_dir = './tmpignore'
if not os.path.exists(tmp_feature_dir):
    os.makedirs(tmp_feature_dir)
mp1 = loadmap('../../descriptor.mp')


X1_name = os.path.join(tmp_feature_dir, 'X1_%s.data' % task_name)
if not os.path.exists(X1_name):
    X1 = mp1.batch_transform(df.smiles, n_jobs = 8)
    dump(X1, X1_name)
else:
    X1 = load(X1_name)



train_idx, valid_idx, test_idx = get_attentiveFP_idx(df)
trainY = Y[train_idx]
validY = Y[valid_idx]



import time
start_time = str(time.ctime()).replace(':','-').replace(' ','_')
log_file = data.task_name + '_' + start_time + '.log'

with open(log_file,'a') as f:
    f.write(','.join(['n_neighbors', 'min_dist', 'valid_best_loss', 'valid_best_rmse', 
                      'train_best_rmse', 'best_epoch'])+'\n')

def Eva(n_neighbors,  min_dist):
    
    min_dist = min_dist    
    n_neighbors = n_neighbors

    print({'min_dist':min_dist, 'n_neighbors':n_neighbors})
    mp_new =  loadmap('../../descriptor.mp')
    mp_new.fit(method = 'umap', min_dist = min_dist, n_neighbors = n_neighbors)
    X_new = mp1.rearrangement(X1, mp_new)
    
    trainX = X_new[train_idx]
    validX = X_new[valid_idx]

    opt = tf.keras.optimizers.Adam(lr = 1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #    
    model = molmodel.net.SinglePathNet(trainX.shape[1:], 
                                       n_outputs=1, dense_layers=[128, 32], 
                                       dense_avf='tanh', last_avf='linear')
    
    model.compile(optimizer = opt, loss = 'mse')
    performance = molmodel.cbks.Reg_EarlyStoppingAndPerformance((trainX, trainY), 
                                                               (validX, validY), 
                                                               patience = 1000000, #find best epoch in total 500 epochs
                                                               criteria = 'val_loss')
    model.fit(trainX, trainY, batch_size = 128, 
          epochs=500, verbose= 0, shuffle = True, 
          validation_data = (validX, validY), 
          callbacks=[performance]) 
    
    #performance.model.set_weights(performance.best_weights) #set best model as the final model
    
    valid_rmse, valid_r2 = performance.evaluate(validX, validY)
    train_rmse, train_r2 = performance.evaluate(trainX, trainY)
    
    valid_best_rmse = np.nanmean(valid_rmse)
    train_best_rmse = np.nanmean(train_rmse)
    valid_best_loss = perfomrance.best
    
    best_epoch = performance.best_epoch
    
    with open(log_file, 'a') as f:
        f.write(','.join([str(min_dist), str(n_neighbors), str(valid_best_loss), str(valid_best_rmse), 
                          str(train_best_rmse), str(best_epoch)]) + '\n')
        
    return [valid_best_loss, valid_best_rmse, train_best_rmse, best_epoch]


if __name__ == '__main__':

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    n_neighbors_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105]
    min_dist_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,0.55, 0.6, 0.65, 0.7,0.75, 0.8, 0.85, 0.9, 0.95]
    res = []
    for n_neighbors in n_neighbors_list:
        for min_dist in min_dist_list:
            output = Eva(n_neighbors,  min_dist)
            x = [n_neighbors, min_dist]
            x.extend(output)
            res.append(x)
    df = pd.read_csv(log_file)
    x = df.valid_best_rmse.values.reshape(len(n_neighbors_list), len(min_dist_list))
    n_neighbors_list = df.n_neighbors.unique()
    min_dist_list = df.min_dist.unique()

    dfr = pd.DataFrame(x, index = min_dist_list, columns = n_neighbors_list).round(3)
    fig, ax = plt.subplots(figsize = (12,10))
    sns.set(font_scale = 1.4)
    sns.heatmap(dfr, cmap='rainbow',
                     vmin = 0.50, vmax = 0.56, 
                     square=True, 
                     annot = True, fmt = '.3f',
                     ax = ax,
                     linewidths = 0.05, linecolor= 'white',
                     alpha = 0.9,
                     annot_kws = {'size': 11},
                     cbar_kws = {'fraction':0.046, 'pad': 0.02,  'label': 'RMSE of Validation Set'}
                    )
    ax.set_xlabel('min_dist', size = 15)
    ax.set_ylabel('n_neighbors', size = 15)
    plt.savefig(log_file + '.png', dpi = 400, bbox_inches ='tight')