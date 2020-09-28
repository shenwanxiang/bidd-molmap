import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import load, dump
import matplotlib.pyplot as plt


from molmap import loadmap
from molmap.model import RegressionEstimator, MultiClassEstimator, MultiLabelEstimator
from molmap import loadmap, dataset
from molmap.show import imshow_wrap
import molmap
import os

## fix random seed to get repeatale results
seed = 123
tqdm.pandas(ascii=True)
np.random.seed(seed)

from chembench import dataset, load_data
#load dataset
data = dataset.load_BACE()
df = data.df
Y = data.y


task_name = data.task_name


train_df = pd.read_csv('./train.csv')
valid_df = pd.read_csv('./val.csv')
test_df = pd.read_csv('./test.csv')


train_idx = df[df.smiles.isin(train_df.smiles)].index
valid_idx = df[df.smiles.isin(valid_df.smiles)].index
test_idx = df[df.smiles.isin(test_df.smiles)].index

trainY = Y[train_idx]
validY = Y[valid_idx]
testY =  Y[test_idx]


print(len(train_idx), len(valid_idx), len(test_idx))


mp2 = loadmap('../fingerprint.mp')

tmp_feature_dir = '/raid/shenwanxiang/08_Robustness/tempignore' #feature path
if not os.path.exists(tmp_feature_dir):
    os.makedirs(tmp_feature_dir)

X2_name = os.path.join(tmp_feature_dir, 'X2_%s.data' % task_name)
if not os.path.exists(X2_name):
    X2 = mp2.batch_transform(df.smiles, n_jobs = 8)
    dump(X2, X2_name)
else:
    X2 = load(X2_name)
    
    
    
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



pos_weights, neg_weights = get_pos_weights(trainY)
loss = lambda y_true, y_pred: molmap.model.loss.weighted_cross_entropy(y_true,y_pred, pos_weights, MASK = -1)

def Eva(n_neighbors,  min_dist, log_file):
    
    min_dist = min_dist    
    n_neighbors = n_neighbors

    print({'min_dist':min_dist, 'n_neighbors':n_neighbors})
    mp_new =  loadmap('../fingerprint.mp')
    mp_new.fit(method = 'umap', min_dist = min_dist, n_neighbors = n_neighbors)
    X_new = mp2.rearrangement(X2, mp_new)
    
    trainX = X_new[train_idx]
    validX = X_new[valid_idx]
    testX = X_new[test_idx]
    
    clf = MultiLabelEstimator(n_outputs = 1,
                              fmap_shape1 = trainX.shape[1:],
                              batch_size = 128,
                              dense_layers = [128, 32],
                              gpuid = "0",
                              patience = 1000000, #find best epoch in total 200 epochs
                              monitor = 'val_auc',
                              epochs = 200)             
    
    clf.fit(trainX, trainY, validX, validY) 

    best_epoch = clf._performance.best_epoch
    train_aucs = clf._performance.evaluate(trainX, trainY)            
    valid_aucs = clf._performance.evaluate(validX, validY)            
    test_aucs = clf._performance.evaluate(testX, testY)     
    
    train_best_auc = np.nanmean(train_aucs)    
    valid_best_auc = np.nanmean(valid_aucs)
    test_auc = np.nanmean(test_aucs)
    
    dfx = pd.DataFrame(clf._performance.history)
    valid_best_loss = dfx[dfx.epoch == clf._performance.best_epoch].val_loss.iloc[0]

    with open(log_file, 'a') as f:
        f.write(','.join([str(min_dist), str(n_neighbors),str(valid_best_loss), str(valid_best_auc), 
                          str(train_best_auc), str(best_epoch), str(test_auc)]) + '\n')
        
    return [valid_best_auc, train_best_auc, best_epoch]



if __name__ == '__main__':

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    import time

    n_neighbors_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105]
    min_dist_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,0.55, 0.6, 0.65, 0.7,0.75, 0.8, 0.85, 0.9, 0.95]

    flag = 'bace_search'
    start_time = str(time.ctime()).replace(':','-').replace(' ','_')
    log_file = task_name + '_' + flag + '_' + start_time + '.log'

    with open(log_file,'a') as f:
        f.write(','.join([ 'min_dist','n_neighbors', 'valid_best_loss','valid_best_auc', 
                          'train_best_auc', 'best_epoch', 'test_auc'])+'\n')

    for n_neighbors in n_neighbors_list:
        for min_dist in min_dist_list:
            output = Eva(n_neighbors,  min_dist, log_file)


















