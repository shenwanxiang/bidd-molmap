import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data


import time
import numpy as np
import gc
import sys
sys.setrecursionlimit(50000)
import pickle
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# from tensorboardX import SummaryWriter
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

def eval2(model, dataset):
    model.eval()
    test_MAE_list = []
    test_MSE_list = []
    pred = []
    valList = np.arange(0,dataset.shape[0])
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch) 
    for counter, test_batch in enumerate(batch_list):
        batch_df = dataset.iloc[test_batch,:]
        smiles_list = batch_df.cano_smiles.values
#         print(batch_df)
        y_val = batch_df[tasks[0]].values
        
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        atoms_prediction, mol_prediction = model(torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.cuda.LongTensor(x_atom_index),torch.cuda.LongTensor(x_bond_index),torch.Tensor(x_mask))
        MAE = F.l1_loss(mol_prediction, torch.Tensor(y_val).view(-1,1), reduction='none')        
        MSE = F.mse_loss(mol_prediction, torch.Tensor(y_val).view(-1,1), reduction='none')
        pred.extend(mol_prediction.data.squeeze().cpu().numpy())
#         print(x_mask[:2],atoms_prediction.shape, mol_prediction,MSE)
        
        test_MAE_list.extend(MAE.data.squeeze().cpu().numpy())
        test_MSE_list.extend(MSE.data.squeeze().cpu().numpy())
    return np.sqrt(np.array(test_MSE_list).mean()), pred




task_name = 'Malaria Bioactivity'
tasks = ['Loge EC50']

raw_filename = "./malaria-processed.csv"
feature_filename = raw_filename.replace('.csv','.pickle')
filename = raw_filename.replace('.csv','')
prefix_filename = raw_filename.split('/')[-1].replace('.csv','')
smiles_tasks_df = pd.read_csv(raw_filename, names = ["Loge EC50", "smiles"])
smilesList = smiles_tasks_df.smiles.values
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
smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)]
# print(smiles_tasks_df)
smiles_tasks_df['cano_smiles'] =canonical_smiles_list



if os.path.isfile(feature_filename):
    feature_dicts = pickle.load(open(feature_filename, "rb" ))
else:
    feature_dicts = save_smiles_dicts(smilesList,filename)
# feature_dicts = get_smiles_dicts(smilesList)
remained_df = smiles_tasks_df[smiles_tasks_df["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
uncovered_df = smiles_tasks_df.drop(remained_df.index)



df_train = pd.read_csv('../train.csv', index_col = 0)
df_valid = pd.read_csv('../valid.csv', index_col = 0)
df_test = pd.read_csv('../test.csv', index_col = 0)

train_df = remained_df[remained_df.smiles.isin(df_train.smiles)]
valid_df = remained_df[remained_df.smiles.isin(df_valid.smiles)]
test_df = remained_df[remained_df.smiles.isin(df_test.smiles)]
print(len(train_df), len(valid_df), len(test_df))

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)






if __name__ == '__main__': 

    results = []
    random_seeds = [7,77,777]
    
    for seed in random_seeds:    
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        start_time = str(time.ctime()).replace(':','-').replace(' ','_')
        batch_size = 200
        epochs = 800
        p_dropout= 0.03
        fingerprint_dim = 200

        weight_decay = 4.3 # also known as l2_regularization_lambda
        learning_rate = 4
        radius = 2
        T = 1
        per_task_output_units_num = 1 # for regression model
        output_units_num = len(tasks) * per_task_output_units_num


        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([canonical_smiles_list[0]],feature_dicts)
        num_atom_features = x_atom.shape[-1]
        num_bond_features = x_bonds.shape[-1]
        loss_function = nn.MSELoss()
        model = Fingerprint(radius, T, num_atom_features, num_bond_features,
                    fingerprint_dim, output_units_num, p_dropout)
        model.cuda()

        optimizer = optim.Adam(model.parameters(), 10**-learning_rate, weight_decay=10**-weight_decay)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        best_param ={}
        best_param["train_epoch"] = 0
        best_param["valid_epoch"] = 0
        best_param["train_MSE"] = 9e8
        best_param["valid_MSE"] = 9e8

        for epoch in range(800):
            train_MAE, train_MSE = eval(model, train_df)
            valid_MAE, valid_MSE = eval(model, valid_df)

            if train_MSE < best_param["train_MSE"]:
                best_param["train_epoch"] = epoch
                best_param["train_MSE"] = train_MSE
            if valid_MSE < best_param["valid_MSE"]:
                best_param["valid_epoch"] = epoch
                best_param["valid_MSE"] = valid_MSE
                if valid_MSE < 1.1:
                     torch.save(model, 'saved_models/model_'+prefix_filename+'_'+start_time+'_'+str(epoch)+'.pt')
            if (epoch - best_param["train_epoch"] >2) and (epoch - best_param["valid_epoch"] >18):        
                break
            print(epoch, train_MSE, valid_MSE)

            train(model, train_df, optimizer, loss_function)

        # evaluate model
        best_model = torch.load('saved_models/model_'+prefix_filename+'_'+start_time+'_'+str(best_param["valid_epoch"])+'.pt')     

        best_model_dict = best_model.state_dict()
        best_model_wts = copy.deepcopy(best_model_dict)

        model.load_state_dict(best_model_wts)
        (best_model.align[0].weight == model.align[0].weight).all()
        test_MAE, test_MSE = eval(model, test_df)
        print("best epoch:",best_param["valid_epoch"],"\n","test RMSE:",np.sqrt(test_MSE))

        
        best_train_rmse, best_train_pred_y  = eval2(model, train_df)
        best_valid_rmse, best_valid_pred_y = eval2(model, valid_df)
        best_test_rmse, best_test_pred_y = eval2(model, test_df)

        
        final_res = {'train_rmse':best_train_rmse, 
                     'valid_rmse':best_valid_rmse,                      
                     'test_rmse':best_test_rmse, 

                     'train_pred_y':best_train_pred_y,
                     'valid_pred_y':best_valid_pred_y,                     
                     'test_pred_y':best_test_pred_y,

                     '# trainable params': params,
                     'random_seed':seed, 
                     'best_epoch': best_param["valid_epoch"],
                     
                     'batch_size':batch_size,
                     'lr': 10**-learning_rate,

                    }

        results.append(final_res)

    pd.DataFrame(results).to_csv('../results/results_AttentiveFP.csv')














