import os, sys
import molmap
from molmap import dataset
path = os.path.join(os.path.dirname(os.path.dirname(molmap.__file__)), 
                    'paper/05_solubility_prediction_model_interpretation/04_AttentiveFP')
sys.path.insert(0, path)
sys.setrecursionlimit(50000)

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data


import time
import numpy as np
import gc


import pickle
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.nn.Module.dump_patches = True

import copy
import pandas as pd
#then import my own modules
from AttentiveFP import Fingerprint, Fingerprint_viz, save_smiles_dicts, get_smiles_dicts, get_smiles_array, moltosvg_highlight
from rdkit import Chem
# from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem import rdMolDescriptors, MolSurf
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import seaborn as sns; sns.set()
from IPython.display import SVG, display

seed = 777
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.set_device(6)

def train(model, dataset, optimizer, loss_function):
    model.train()
    np.random.seed(epoch)
    valList = np.arange(0,dataset.shape[0])
    #shuffle them
    np.random.shuffle(valList)
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch)   
    for counter, batch in enumerate(batch_list):
        batch_df = dataset.loc[batch,:]
        smiles_list = batch_df.cano_smiles.values
        y_val = batch_df[tasks[0]].values
        
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        atoms_prediction, mol_prediction = model(torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.cuda.LongTensor(x_atom_index),torch.cuda.LongTensor(x_bond_index),torch.Tensor(x_mask))
        
        model.zero_grad()
        loss = loss_function(mol_prediction, torch.Tensor(y_val).view(-1,1))     
        loss.backward()
        optimizer.step()
def eval(model, dataset):
    model.eval()
    test_MAE_list = []
    test_MSE_list = []
    valList = np.arange(0,dataset.shape[0])
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch) 
    for counter, batch in enumerate(batch_list):
        batch_df = dataset.loc[batch,:]
        smiles_list = batch_df.cano_smiles.values
#         print(batch_df)
        y_val = batch_df[tasks[0]].values
        
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        atoms_prediction, mol_prediction = model(torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.cuda.LongTensor(x_atom_index),torch.cuda.LongTensor(x_bond_index),torch.Tensor(x_mask))
        MAE = F.l1_loss(mol_prediction, torch.Tensor(y_val).view(-1,1), reduction='none')        
        MSE = F.mse_loss(mol_prediction, torch.Tensor(y_val).view(-1,1), reduction='none')
#         print(x_mask[:2],atoms_prediction.shape, mol_prediction,MSE)
        
        test_MAE_list.extend(MAE.data.squeeze().cpu().numpy())
        test_MSE_list.extend(MSE.data.squeeze().cpu().numpy())
    return np.array(test_MAE_list).mean(), np.array(test_MSE_list).mean()


task_name = 'solubility'
tasks = ['measured log solubility in mols per litre']
batch_size = 200



p_dropout= 0.2
fingerprint_dim = 200
weight_decay = 5 # also known as l2_regularization_lambda
learning_rate = 2.5
radius = 2
T = 2
per_task_output_units_num = 1 # for regression model
output_units_num = 1


#load dataset
data = dataset.load_ESOL()
df = data.data

valid_idx = df.sample(frac = 0.2).index.to_list()
train_idx = list(set(df.index) - set(valid_idx))


res = []
for epochs in [1, 10, 50, 100, 150, 300, 500]:
    
    start = time.time()
    
    smilesList = df.smiles.values
    print("number of all smiles: ",len(smilesList))
    atom_num_dist = []
    remained_smiles = []
    canonical_smiles_list = []
    for smiles in smilesList:
        try:        
            mol = Chem.MolFromSmiles(smiles)
            atom_num_dist.append(len(mol.GetAtoms()))
            remained_smiles.append(smiles)
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
        except:
            print(smiles)
            pass
    print("number of successfully processed smiles: ", len(remained_smiles))
    df["cano_smiles"] = canonical_smiles_list
    feature_dicts = save_smiles_dicts(smilesList, 'tmp')
    remained_df = df[df["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
    uncovered_idx = set(df.index) - set(remained_df.index)
    train_idx = set(train_idx) - set(uncovered_idx)
    valid_idx = set(valid_idx) - set(uncovered_idx)
    print(len(train_idx), len(valid_idx))
    
    train_df = remained_df.loc[train_idx].reset_index(drop=True)
    valid_df = remained_df.loc[valid_idx].reset_index(drop=True)

    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([canonical_smiles_list[0]],feature_dicts)
    num_atom_features = x_atom.shape[-1]
    num_bond_features = x_bonds.shape[-1]
    loss_function = nn.MSELoss()
    model = Fingerprint(radius, T, num_atom_features, num_bond_features,
                fingerprint_dim, output_units_num, p_dropout)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), 10**-learning_rate, weight_decay=10**-weight_decay)

    for epoch in range(epochs):
        train_MAE, train_MSE = eval(model, train_df)
        valid_MAE, valid_MSE = eval(model, valid_df)
        print(epoch, np.sqrt(train_MSE), np.sqrt(valid_MSE))
        train(model, train_df, optimizer, loss_function)

    end = time.time()
    total = end - start
    print('total epoch: %s, total time: %s' % (epochs, total))
    res.append([epochs, total])

x = pd.DataFrame(res, columns = ['epochs', 'total_time(s)'])
x['average_time(s)'] =  x['total_time(s)'] / x['epochs']
x.to_csv('./attentivFP.csv')