from __future__ import division, print_function

import cairosvg

import torch
from profilehooks import profile
from sklearn.metrics import mean_squared_error
from math import sqrt
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models
from torchvision import transforms as data_augmentations
import os, glob, time
import copy
import joblib, sys
import numpy as np
import scipy
from scipy import stats
from scipy import spatial
import os,sys, os.path
from collections import defaultdict
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.rdBase
from rdkit import DataStructs
from rdkit.DataStructs import BitVectToText
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG
import IPython
#IPython.core.display.set_matplotlib_formats('svg')
from IPython.core.display import SVG
from torch.autograd import Variable
import multiprocessing
import pandas as pd

def count_trainable_params(model_ft):
    model_parameters = filter(lambda p: p.requires_grad, model_ft.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])    
    return params


def split(df, random_state = 1, split_size = [0.7, 0.15, 0.15]):
    from sklearn.utils import shuffle
    base_indices = np.arange(len(df))
    base_indices = shuffle(base_indices, random_state  = random_state)
    nb_test = int(len(base_indices) * split_size[2])
    nb_val = int(len(base_indices) * split_size[1])

    test_idx = base_indices[1:nb_test]
    valid_idx = base_indices[(nb_test+1):(nb_test+nb_val)]
    train_idx = base_indices[(nb_test+nb_val+1):len(base_indices)]

    print(len(train_idx), len(valid_idx), len(test_idx))
    return train_idx, valid_idx, test_idx

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    start_time = time.time()
    # use the lines below to use more than 1 GPU
    #model = torch.nn.DataParallel(model, device_ids=[0, 1 , 2, 3])
    model.cuda()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000.0
    best_epoch = 0
    early = 0
    
    for epoch in range(num_epochs):
        time_epoch = time.time()

        # cyclical learning rate
        if epoch %   nb_epochs_training_per_cycle == 0:
            optimizer = optim.SGD(model.parameters(), lr= lr)
            scheduler = lr_scheduler.StepLR(optimizer, step_size= step_size_lr_decay, gamma= drop_factor_lr)

        print('Epoch {}/{} {}'.format(epoch, num_epochs - 1, early))
        print('-' * 10)
        if early >=  epochs_early_stop:
            model.load_state_dict(best_model_wts)
    
            time_elapsed = time.time() - start_time
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val loss: {:4f}'.format(best_loss))
            return model, best_epoch

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            epoch_losses=0.0
            deno=0.0
            if phase == 'train':
                scheduler.step()
                model.train()  
            else:
                model.eval()

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()
                labels = labels.type(torch.FloatTensor)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    aa = time.time()
                    outputs = model(inputs)
                    preds=outputs.squeeze(1)
                    preds = preds.type(torch.FloatTensor)
                    loss = criterion(preds, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        aa = time.time()
                        loss.backward()
                        optimizer.step()

                del inputs; del outputs
                epoch_losses += loss.data * len(preds)
                deno +=len(preds)
                del preds

            epoch_loss = epoch_losses / deno
            print('{} Loss: {:.4f} {}'.format(phase, epoch_loss, deno))
            del deno

            #torch.cuda.empty_cache()

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                early=0
            if phase == 'val' and epoch_loss > best_loss:
                early+=1

            # stop if there is no convergence....
            #if phase == 'val' and best_loss > 2 and epoch >= 50:
            #    model.load_state_dict(best_model_wts)
            #    return model

            # now predict for test set
            if phase == 'val':
                pred=[]
                obs=[]
                for inputs, labels in dataloaders['test']:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    outputs = model(inputs)
                    for i in range(len(labels)):
                        pred.append(outputs.data[i])
                        obs.append(labels.data[i])
                    del labels, outputs, inputs
                pred=np.asarray(pred)
                obs=np.asarray(obs)
                rms = sqrt(mean_squared_error( pred,obs)) 
                r2=scipy.stats.pearsonr(pred,obs)
                print('test rmse: %0.3f, test-r2: %0.3f' % (rms, r2[0]**2))
                del pred, obs, rms, r2

        print('Epoch complete in {:.0f}m {:.0f}s'.format( (time.time() - time_epoch) // 60, (time.time() - time_epoch) % 60))

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)
    
    return model, best_epoch



def eval(model_ft, tag = 'test'):
    model_ft.eval()
    pred=[]
    obs=[]
    for inputs, labels in dataloaders[tag]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_ft(inputs)
        for i in range(len(labels)):
            pred.append(outputs.data[i])
            obs.append(labels.data[i])
    pred=np.asarray(pred)
    obs=np.asarray(obs)
    rms = sqrt(mean_squared_error( pred,obs)) ##target.flatten() - y_pred.flatten() )
    r2=scipy.stats.pearsonr(pred,obs)[0]**2

    print('RMSE : %0.3f, r2: %0.3f' % (rms, r2))
    return rms, r2


if __name__ == '__main__':
    
    
    net = 'vgg19_bn'
    lr = 0.01
    step_size_lr_decay = 25
    drop_factor_lr = 0.6
    batch_size = 16
    nb_epochs_training_per_cycle = 200
    nb_epochs_training = 800
    epochs_early_stop = 50
    data_split_seed = 1
    
    from molmap.extend.kekulescope import dataset
    from load_images import *
    if not os.path.exists('./models'):
        os.makedirs('./models')
        
    for cell_line in dataset.cell_lines:

        df = dataset.load_data(cell_line)
        df = df[~df.pIC50.isna()].reset_index(drop=True)

        chembl_ids = df.chembel_ids.values
        activities =  df.pIC50.values
        my_smiles = df.smiles.values
        train_indices, val_indices, test_indices = split(my_smiles, random_state = data_split_seed)
        # divide training into: true training and validation
        activities_train = activities[train_indices]
        activities_test = activities[test_indices]
        activities_val = activities[val_indices]

        chembl_ids_train = chembl_ids[train_indices]
        chembl_ids_test = chembl_ids[test_indices]
        chembl_ids_val = chembl_ids[val_indices]
        
        my_smiles_train = my_smiles[train_indices]
        my_smiles_test = my_smiles[test_indices]
        my_smiles_val = my_smiles[val_indices]
        
        seed = data_split_seed
        #------------
        for tag in ['train', 'val', 'test']:
            if not os.path.exists('./images/%s/%s/%s/images' % (cell_line, seed, tag)):
                os.makedirs('./images/%s/%s/%s/images' % (cell_line, seed, tag))

        ###################################################################################
        svgs = glob.glob( "./images/{}/{}/train/images/*svg".format( cell_line,  seed) )
        pngs = glob.glob( "./images/{}/{}/train/images/*png".format( cell_line,  seed) )
        if len(svgs) == 0 and len(pngs) == 0:
            for i,mm in enumerate(my_smiles_train):
                mol_now=[Chem.MolFromSmiles(my_smiles_train[i])]
                koko=Chem.Draw.MolsToGridImage([x for x in mol_now], molsPerRow=1,useSVG=True)
                orig_stdout = sys.stdout
                f = open('./images/{}/{}/train/images/{}.svg'.format( cell_line, seed,chembl_ids_train[i]), 'w')
                sys.stdout = f
                print(koko.data)
                sys.stdout = orig_stdout
                f.close()
        else:
            print("SVGs ready")
        svgs = glob.glob( "./images/{}/{}/train/images/*svg".format( cell_line,  seed) )
        for svg in svgs:
            cairosvg.svg2png(url=svg, write_to=svg + '.png', dpi = 500, output_width=300, output_height=300)
            cmd = 'rm -rf %s' % svg
            os.system(cmd)    

        svgs = glob.glob( "./images/{}/{}/val/images/*svg".format( cell_line,  seed) )
        pngs = glob.glob( "./images/{}/{}/val/images/*png".format( cell_line,  seed) )
        if len(svgs) == 0 and len(pngs) == 0:
            for i,mm in enumerate(my_smiles_val):
                mol_now=[Chem.MolFromSmiles(my_smiles_val[i])]
                koko=Chem.Draw.MolsToGridImage([x for x in mol_now], molsPerRow=1,useSVG=True)
                orig_stdout = sys.stdout
                f = open('./images/{}/{}/val/images/{}.svg'.format( cell_line, seed,chembl_ids_val[i]), 'w')
                sys.stdout = f
                print(koko.data)
                sys.stdout = orig_stdout
                f.close()
        else:
            print("SVGs ready")
        svgs = glob.glob( "./images/{}/{}/val/images/*svg".format( cell_line,  seed) )
        for svg in svgs:
            cairosvg.svg2png(url=svg, write_to=svg + '.png', dpi = 500, output_width=300, output_height=300)
            cmd = 'rm -rf %s' % svg
            os.system(cmd)


        svgs = glob.glob( "./images/{}/{}/test/images/*svg".format( cell_line,  seed) )
        pngs = glob.glob( "./images/{}/{}/test/images/*png".format( cell_line,  seed) )
        if len(svgs) == 0 and len(pngs) == 0:
            for i,mm in enumerate(my_smiles_test):
                mol_now=[Chem.MolFromSmiles(my_smiles_test[i])]
                koko=Chem.Draw.MolsToGridImage([x for x in mol_now], molsPerRow=1,useSVG=True)
                orig_stdout = sys.stdout
                f = open('./images/{}/{}/test/images/{}.svg'.format( cell_line, seed,chembl_ids_test[i]), 'w')
                sys.stdout = f
                print(koko.data)
                sys.stdout = orig_stdout
                f.close()
        else:
            print("SVGs ready")
        svgs = glob.glob( "./images/{}/{}/test/images/*svg".format( cell_line,  seed) )
        for svg in svgs:
            cairosvg.svg2png(url=svg, write_to=svg + '.png', dpi = 500, output_width=300, output_height=300)
            cmd = 'rm -rf %s' % svg
            os.system(cmd)


        transform = {'train':  data_augmentations.Compose([
                     data_augmentations.Resize(224),
                     data_augmentations.ToTensor(),
                     data_augmentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]), 'val':  data_augmentations.Compose([
                     data_augmentations.Resize(224),
                     data_augmentations.ToTensor(),
                     data_augmentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),'test':  data_augmentations.Compose([
                     data_augmentations.Resize(224),
                     data_augmentations.ToTensor(),
                     data_augmentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                }

        data_dir="./images/{}/{}/".format(cell_line,  seed)

        paths_labels_train=[]
        for i,x in enumerate(activities_train):
            path_now = './images/{}/{}/train/images/{}.svg.png'.format( cell_line, seed,chembl_ids_train[i])
            now = (path_now , x)
            paths_labels_train.append(now)

        paths_labels_val=[]
        for i,x in enumerate(activities_val):
            path_now = './images/{}/{}/val/images/{}.svg.png'.format( cell_line, seed,chembl_ids_val[i])
            now = (path_now , x)
            paths_labels_val.append(now)

        paths_labels_test=[]
        for i,x in enumerate(activities_test):
            path_now = './images/{}/{}/test/images/{}.svg.png'.format( cell_line, seed,chembl_ids_test[i])
            now = (path_now , x)
            paths_labels_test.append(now)

        workers=multiprocessing.cpu_count()
        shuffle=False
        ## use the custom functions to load the data
        trainloader = torch.utils.data.DataLoader(
                    ImageFilelist(paths_labels= paths_labels_train,
                    transform=transform['train']),
                    batch_size= batch_size, shuffle=shuffle,
                    num_workers=workers) 

        valloader = torch.utils.data.DataLoader(
                    ImageFilelist(paths_labels= paths_labels_val,
                    transform=transform['val']),
                    batch_size= batch_size, shuffle=shuffle,
                    num_workers=workers) 

        testloader = torch.utils.data.DataLoader(
                    ImageFilelist(paths_labels= paths_labels_test,
                    transform=transform['test']),
                    batch_size= batch_size, shuffle=shuffle,
                    num_workers=workers) 
        
        dataloaders = {'train': trainloader, 'val':valloader, 'test':testloader}
        ###################################################################################
        
        results = []
        for rdseed in [7, 77, 777]:
            np.random.seed(rdseed)
            torch.manual_seed(rdseed)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            use_gpu = lambda x=True: torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() and x else torch.FloatTensor)
            use_gpu()
            model_ft = models.vgg19_bn(pretrained=True)
            modules=[]
            modules.append( nn.Linear(in_features=25088, out_features=4096, bias=True) )
            modules.append( nn.ReLU(inplace=True) )
            modules.append( nn.Dropout(p=0.5) )
            modules.append( nn.Linear(in_features=4096, out_features=1000, bias=True) )
            modules.append( nn.ReLU(inplace=True) )
            modules.append( nn.Dropout(p=0.5) )
            modules.append( nn.Linear(in_features=1000, out_features=200, bias=True) )
            modules.append( nn.ReLU(inplace=True) )
            modules.append( nn.Dropout(p=0.5) )
            modules.append( nn.Linear(in_features=200, out_features=100, bias=True) )
            modules.append( nn.ReLU(inplace=True) )
            modules.append( nn.Dropout(p=0.5) )
            modules.append( nn.Linear(in_features=100, out_features=1, bias=True) )
            classi = nn.Sequential(*modules)
            model_ft.classifier = classi

            optimizer_ft = optim.SGD(model_ft.parameters(), lr= lr)#, momentum=0.95) #, nesterov=True)
            model_ft = model_ft.to(device)
            criterion = torch.nn.MSELoss()
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size= step_size_lr_decay, gamma= drop_factor_lr)

            model_ft, best_epoch = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs= nb_epochs_training)
            
            torch.save(model_ft, 'models/model_{}_{}_{}_{}.pt'.format(net, cell_line, rdseed, lr))
            train_rmse, train_r2 = eval(model_ft, tag = 'train')
            valid_rmse, valid_r2 = eval(model_ft, tag = 'val')
            test_rmse, test_r2 = eval(model_ft, tag = 'test')
            trainable_params = count_trainable_params(model_ft)

            final_res = {
                         'cell_line':cell_line,            
                        'train_rmse':train_rmse, 
                         'valid_rmse':valid_rmse,                      
                         'test_rmse':test_rmse, 

                         'train_r2':train_r2, 
                         'valid_r2':valid_r2,                      
                         'test_r2':test_r2, 

                         '# trainable params': trainable_params,
                         'random_seed':rdseed, 
                         'best_epoch': best_epoch,

                         'batch_size':batch_size,
                         'lr': lr,
                         'data_split_seed':seed,
                        }

            results.append(final_res)
            
        pd.DataFrame(results).to_csv('../results/results_kekulescope_%s.csv' % cell_line)