import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(ascii=True)
import os


#import deepchem as dc
#_, data, _ = dc.molnet.load_tox21(featurizer = 'Raw', split='random')
#__file__ = '/home/shenwanxiang/Research/bidd-molmap/molmap/dataset/__init__.py'

class data(object):
    
    def __init__(self, df, smiles_col, target_cols, task_name, task_type, description):
                
        assert type(target_cols) == list
        
        
        
        self.data = df
        self.x = df[smiles_col].values        
        self.y = df[target_cols].values.astype(float)
        self.y_cols = target_cols
        self.task_name = task_name
        self.task_type = task_type
        self.description = description
        self.n_samples = len(df)
        
        print('total samples: %s' % self.n_samples)
        
        
    def split(self, method = 'random', random_state = 32):
        """
        parameters
        --------------------
        method" {'random', 'sccafold'}
        
        """
        pass
        


################################ regression task  #####################################
def load_malaria():
    
    description = """the Malaria dataset includes 9998 compounds that experimentally measured EC50 values of a sulfide-resistant strain of Plasmodium falciparum, which is the
                    source of malaria"""
    
    task_name = 'malaria'
    task_type = 'regression'
    filename = os.path.join(os.path.dirname(__file__), 'malaria-processed.csv')
    df = pd.read_csv(filename)
    target_cols = ['activity']
    smiles_col = 'smiles'
    return data(df, smiles_col,target_cols, task_name, task_type, description)    

    
    
    
def load_IVPK():
    task_name = 'IVPK'
    task_type = 'regression'
    description = """THE IVPK dataset contains 1352 drugs of human intravenous pharmacokinetic data, 
                    including VD, CL, Fu, MRT and T1/2"""
    
    filename = os.path.join(os.path.dirname(__file__), 'IVPK.csv')
    df = pd.read_csv(filename)
    target_cols = ['human VDss (L/kg)', 'human CL (mL/min/kg)',
                   'fraction unbound \nin plasma (fu)', 'MRT (h)',
                   'terminal  t1/2 (h)']
    smiles_col = 'SMILES'
    return data(df, smiles_col,target_cols, task_name, task_type, description)


################################ classification task #####################################

def load_HIV():
    task_name = 'HIV'
    task_type = 'classification'
    description = """The HIV dataset conatins 41127 compounds and their binnary ability to inhibit HIV replication."""

    
    filename = os.path.join(os.path.dirname(__file__), 'HIV.csv')
    df = pd.read_csv(filename)    
    target_cols = ['HIV_active']
    smiles_col = 'smiles'
    return data(df, smiles_col, target_cols, task_name, task_type, description)    


def load_Tox21():
    task_name = 'Tox21'
    task_type = 'classification'
    description = """The Tox21 dataset contains 8014 compounds and corresponding toxicity data against 12 targets."""

    filename = os.path.join(os.path.dirname(__file__), 'tox21.csv')
    df = pd.read_csv(filename)
    target_cols = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    smiles_col = 'smiles'
    return data(df, smiles_col, target_cols, task_name, task_type, description)




def load_SIDER():
    task_name = 'SIDER'
    task_type = 'classification'
    description = """The SIDER dataset contains 1427 marketed drugs and their adverse drug reactions (ADR) against 27 System-Organs Class."""

    
    filename = os.path.join(os.path.dirname(__file__), 'sider.csv')
    df = pd.read_csv(filename)
    target_cols = ['SIDER1', 'SIDER2', 'SIDER3', 'SIDER4', 'SIDER5', 'SIDER6', 'SIDER7',
                   'SIDER8', 'SIDER9', 'SIDER10', 'SIDER11', 'SIDER12', 'SIDER13',
                   'SIDER14', 'SIDER15', 'SIDER16', 'SIDER17', 'SIDER18', 'SIDER19',
                   'SIDER20', 'SIDER21', 'SIDER22', 'SIDER23', 'SIDER24', 'SIDER25',
                   'SIDER26', 'SIDER27']
    smiles_col = 'smiles'
    return data(df, smiles_col, target_cols, task_name, task_type, description)


def load_CYP450():
    task_name = 'CYP450'
    task_type = 'classification'
    description = """The CYP450 dataset contains 16896 compounds against of five main CYP450 isozymes: 1A2, 2C9, 2C19, 2D6, and 3A4. This data should split training and test set by aids"""
    filename = os.path.join(os.path.dirname(__file__), 'cyp450-processed.csv')
    df = pd.read_csv(filename)
    target_cols = ['label_1a2', 'label_2c9', 'label_2c19', 'label_2d6', 'label_3a4']
    smiles_col = 'smiles'
    return data(df, smiles_col, target_cols, task_name, task_type, description)