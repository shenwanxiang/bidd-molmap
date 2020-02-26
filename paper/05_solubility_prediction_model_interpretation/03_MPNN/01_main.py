import numpy as np
import tensorflow as tf
from joblib import load,dump
from rdkit import Chem
import pandas as pd

from deepchem.models import GraphConvModel, MPNNModel
import deepchem as dc
from deepchem.molnet.preset_hyper_parameters import hps
from copy import deepcopy
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"



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


featurizer = dc.feat.graph_features.WeaveFeaturizer()
metric = dc.metrics.Metric(dc.metrics.rms_score, np.mean)


tasks = ['measured log solubility in mols per litre']
smiles_col = 'smiles'

# tasks, delaney_datasets, transformers = dc.molnet.load_delaney(featurizer='Weave', split='random')
# train_dataset, valid_dataset, test_dataset = delaney_datasets
train_dataset, transformers =  featurize_data(tasks = tasks, smiles_col = smiles_col, 
                                           featurizer =featurizer, dataset_file='../train.csv')

valid_dataset, _ =  featurize_data(tasks = tasks, smiles_col = smiles_col, 
                                           featurizer =featurizer, dataset_file='../valid.csv')

test_dataset, _ =  featurize_data(tasks = tasks, smiles_col = smiles_col, 
                                           featurizer =featurizer, dataset_file='../test.csv')

etc_dataset, _ =  featurize_data(tasks = ['Exp_LogS'], smiles_col = smiles_col, 
                                           featurizer =featurizer, dataset_file='../etc.csv')



if __name__ == '__main__':

    
    
    batch_size = 32
    lr = 0.003183415042748088
    n_atom_feat = 75
    n_pair_feat = 14
    T = 1
    M = 2

    patience = 30
    epochs = 800

    results = []
    best_epochs = []
    for seed in [7,77,777]:
        np.random.seed(seed)
        tf.set_random_seed(seed)

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


        ## perform an early stopping strategy
        best_rmse = np.inf
        model_performace = {}
        wait = 0
        best_model = None
        best_epoch = 0
        for i in range(epochs):
            
            train_dataset.shuffle_each_shard()
            
            model.fit(train_dataset, nb_epoch=1, verbos = 0)
            valid_scores = model.evaluate(valid_dataset, [metric], transformers)
            valid_rmse = valid_scores.get('mean-rms_score')
            model_performace[i] = valid_rmse


            if  valid_rmse < best_rmse:
                best_rmse = valid_rmse
                print(best_rmse)
                wait = 0        
                best_train_rmse = model.evaluate(train_dataset, [metric], transformers).get('mean-rms_score')
                best_valid_rmse = model.evaluate(valid_dataset, [metric], transformers).get('mean-rms_score')        
                best_test_rmse = model.evaluate(test_dataset, [metric], transformers).get('mean-rms_score')
                best_etc_rmse = model.evaluate(etc_dataset, [metric], transformers).get('mean-rms_score')

                best_train_pred_y = list(model.predict(train_dataset, transformers).reshape(-1,))
                best_valid_pred_y = list(model.predict(valid_dataset, transformers).reshape(-1,))           
                best_test_pred_y = list(model.predict(test_dataset, transformers).reshape(-1,))                 
                best_etc_pred_y = list(model.predict(etc_dataset, transformers).reshape(-1,))             


                best_epoch = i
            else:
                wait += 1
                if wait >= patience:
                    print('early stopping at best_rmse : %s...' % best_rmse)
                    break

            

        final_res = {'train_rmse':best_train_rmse, 
                     'valid_rmse':best_valid_rmse, 
                     'test_rmse':best_test_rmse,                      
                     'etc_rmse':best_etc_rmse,

                     'train_pred_y':best_train_pred_y,
                     'valid_pred_y':best_valid_pred_y,                     
                     'test_pred_y':best_test_pred_y,
                     'etc_pred_y':best_etc_pred_y, 
                     
                     
                     '# trainable params': count_params(model),
                     'random_seed':seed, 
                     'best_epoch': best_epoch,
                     
                     'batch_size':batch_size,
                     'lr': lr,

                    }
        
        results.append(final_res)
    
    print(best_epochs)
    pd.DataFrame(results).to_csv('../results/results_03_MPNN.csv')
