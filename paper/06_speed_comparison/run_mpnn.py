import numpy as np
import tensorflow as tf
from joblib import load,dump
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
from deepchem.models import GraphConvModel, MPNNModel
import deepchem as dc
from deepchem.molnet.preset_hyper_parameters import hps
from copy import deepcopy
import os
import time
from molmap import dataset
os.environ["CUDA_VISIBLE_DEVICES"]="7"


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
df.iloc[valid_idx].to_csv('./valid.csv')
df.iloc[train_idx].to_csv('./train.csv')

featurizer = dc.feat.graph_features.WeaveFeaturizer()
metric = dc.metrics.Metric(dc.metrics.rms_score, np.mean)

def count_params(model):
    trainable_paras = 0
    for layer in model.layers.values():
        x = model.get_layer_variable_values(layer)
        for  i in x:
            if type(i) == np.ndarray:
                if len(i.shape) == 1:
                    trainable_paras += i.shape[0]
                else:
                    a, b = i.shape
                    trainable_paras += a*b
    return trainable_paras


def featurize_data(tasks, smiles_col, featurizer, dataset_file, normalize = True):
    loader = dc.data.CSVLoader(tasks=tasks, smiles_field=smiles_col, featurizer=featurizer)
    dataset = loader.featurize(dataset_file, shard_size=8192)
    move_mean = True
    if normalize:
        transformers = [dc.trans.NormalizationTransformer(
            transform_y=True, dataset=dataset, move_mean=move_mean)]
    else:
        transformers = []
    for transformer in transformers:
        dataset = transformer.transform(dataset)
    return dataset, transformers


if __name__ == '__main__':

    batch_size = 200
    lr = 0.003183415042748088
    n_atom_feat = 75
    n_pair_feat = 14
    T = 1
    M = 2
    
    res = []
    for epochs in [1, 10, 50, 100, 150, 300, 500]:
        start = time.time()
        
        tasks = ['measured log solubility in mols per litre']
        smiles_col = 'smiles'

        train_dataset, transformers =  featurize_data(tasks = tasks, smiles_col = smiles_col, 
                                                   featurizer =featurizer, dataset_file='./train.csv')

        valid_dataset, _ =  featurize_data(tasks = tasks, smiles_col = smiles_col, 
                                                   featurizer =featurizer, dataset_file='./valid.csv')
        model = dc.models.MPNNModel(
                                    len(tasks),
                                    n_atom_feat=n_atom_feat,
                                    n_pair_feat=n_pair_feat,
                                    T=T,
                                    M=M,
                                    batch_size=batch_size,
                                    learning_rate=lr,
                                    use_queue=False,
                                    mode="regression")
        for i in range(epochs):
            train_dataset.shuffle_each_shard()
            model.fit(train_dataset, nb_epoch=1, verbos = 0)
            valid_scores = model.evaluate(valid_dataset, [metric], transformers)
            print(valid_scores)
                
        end = time.time()
        total = end - start


        print('total epoch: %s, total time: %s' % (epochs, total))
        res.append([epochs, total])
    
    x = pd.DataFrame(res, columns = ['epochs', 'total_time(s)'])
    x['average_time(s)'] =  x['total_time(s)'] / x['epochs']
    x.to_csv('./deepchem_mpnn.csv')