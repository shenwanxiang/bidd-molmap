import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(ascii=True)
import os
from rdkit import Chem

def to_smiles(mol):
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

def to_mol(smiles):
    return Chem.MolFromSmiles(smiles)




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
        
        


################################ regression task  #####################################
# def load_malaria(check_smiles = False):
    
#     description = """the Malaria dataset includes 9998 compounds that experimentally measured EC50 values of a sulfide-resistant strain of Plasmodium falciparum, which is the
#                     source of malaria"""
    
#     task_name = 'malaria'
#     task_type = 'regression'
#     filename = os.path.join(os.path.dirname(__file__), 'malaria-processed.csv')
#     df = pd.read_csv(filename)
#     target_cols = ['activity']
#     smiles_col = 'smiles'
#     N = len(df)
#     if check_smiles:
#         mols = df[smiles_col].apply(to_mol)
#         df = df.iloc[mols[~mols.isna()].index]
#         df = df.reset_index(drop=True)
#         M = len(df)
#         if N != M:
#             print("%s invalid smiles are removed" % (N-M))
#     return data(df, smiles_col, target_cols, task_name, task_type, description)

    
# def load_Lipop(check_smiles = False):
    
#     description = """The Lipop (Lipophilicity) dataset has 4200 compounds and and their corresponding experimental lipophilicity values."""
    
#     task_name = 'Lipop'
#     task_type = 'regression'
#     filename = os.path.join(os.path.dirname(__file__), 'Lipophilicity.csv')
#     df = pd.read_csv(filename)
#     target_cols = ['exp']
#     smiles_col = 'smiles' 
#     N = len(df)
#     if check_smiles:
#         mols = df[smiles_col].apply(to_mol)
#         df = df.iloc[mols[~mols.isna()].index]
#         df = df.reset_index(drop=True)
#         M = len(df)
#         if N != M:
#             print("%s invalid smiles are removed" % (N-M))
#     return data(df, smiles_col, target_cols, task_name, task_type, description)


def load_ESOL(check_smiles = False):
    
    description = """The ESOL dataset includes 1128 compounds and their experimental water solubility """
    
    task_name = 'ESOL'
    task_type = 'regression'
    filename = os.path.join(os.path.dirname(__file__), 'delaney-processed.csv')
    df = pd.read_csv(filename)
    target_cols = ['measured log solubility in mols per litre']
    smiles_col = 'smiles'
    N = len(df)
    if check_smiles:
        mols = df[smiles_col].apply(to_mol)
        df = df.iloc[mols[~mols.isna()].index]
        df = df.reset_index(drop=True)
        M = len(df)
        if N != M:
            print("%s invalid smiles are removed" % (N-M))
    return data(df, smiles_col, target_cols, task_name, task_type, description)


def load_FreeSolv():
    
    description = """The FreeSolv dataset contains 642 small molecules' experimental hydration free energy in water """
    
    task_name = 'FreeSolv'
    task_type = 'regression'
    filename = os.path.join(os.path.dirname(__file__), 'FreeSolv.csv')
    df = pd.read_csv(filename)
    target_cols = ['expt']
    smiles_col = 'smiles'
    return data(df, smiles_col,target_cols, task_name, task_type, description)        
    
    
    
# def load_CEP():
    
#     description = """The CEP (Clean Energy Project) has 29978 compounds and corresponding CEP values """
    
#     task_name = 'CEP'
#     task_type = 'regression'
#     filename = os.path.join(os.path.dirname(__file__), 'cep-processed.csv')
#     df = pd.read_csv(filename)
#     target_cols = ['PCE']
#     smiles_col = 'smiles'
#     return data(df, smiles_col,target_cols, task_name, task_type, description)        
    


    

    
# def load_IVPK():
#     task_name = 'IVPK'
#     task_type = 'regression'
#     description = """THE IVPK dataset contains 1352 drugs of human intravenous pharmacokinetic data, 
#                     including VD, CL, Fu, MRT and T1/2"""
    
#     filename = os.path.join(os.path.dirname(__file__), 'IVPK.csv')
#     df = pd.read_csv(filename)
#     df = df.rename(columns={'SMILES':'smiles'})
#     target_cols = ['human VDss (L/kg)', 'human CL (mL/min/kg)',
#                    'fraction unbound \nin plasma (fu)', 'terminal  t1/2 (h)']
#     smiles_col = 'smiles'
#     return data(df, smiles_col,target_cols, task_name, task_type, description)


################################ classification task #####################################



def load_BACE(check_smiles = False):
    task_name = 'BACE'
    task_type = 'classification'
    description = """The BACE dataset contains 1513 inhibitors with their binary inhibition labels for the target of BACE-1"""

    filename = os.path.join(os.path.dirname(__file__), 'bace.csv')
    df = pd.read_csv(filename)
    df = df.rename(columns = {'mol': 'smiles'})
    target_cols = ['Class']
    smiles_col = 'smiles'
    N = len(df)
    if check_smiles:
        mols = df[smiles_col].apply(to_mol)
        df = df.iloc[mols[~mols.isna()].index]
        df = df.reset_index(drop=True)
        M = len(df)
        if N != M:
            print("%s invalid smiles are removed" % (N-M))
    return data(df, smiles_col, target_cols, task_name, task_type, description)


def load_ClinTox():
    task_name = 'ClinTox'
    task_type = 'classification'
    description = """The ClinTox dataset contains 1484 drugs or compounds, the labels are FDA approval status and clinical traial toxicity results."""

    filename = os.path.join(os.path.dirname(__file__), 'clintox.csv')
    df = pd.read_csv(filename)
    target_cols = ['FDA_APPROVED','CT_TOX']
    smiles_col = 'smiles'
    return data(df, smiles_col, target_cols, task_name, task_type, description)


# def load_BBBP(check_smiles = False):
#     task_name = 'BBBP'
#     task_type = 'classification'
#     description = """The BBBP dataset contains 2050 compounds with their binary permeability properties of Blood-brain barrier"""

#     filename = os.path.join(os.path.dirname(__file__), 'BBBP.csv')
#     df = pd.read_csv(filename)
#     target_cols = ['BBBP']
#     smiles_col = 'smiles'
#     N = len(df)
#     if check_smiles:
#         mols = df[smiles_col].apply(to_mol)
#         df = df.iloc[mols[~mols.isna()].index]
#         df = df.reset_index(drop=True)
#         M = len(df)

#         if N != M:
#             print("%s invalid smiles are removed" % (N-M))
#     return data(df, smiles_col, target_cols, task_name, task_type, description)



# def load_HIV(check_smiles = False):
#     task_name = 'HIV'
#     task_type = 'classification'
#     description = """The HIV dataset conatins 41127 compounds and their binnary ability to inhibit HIV replication."""

#     filename = os.path.join(os.path.dirname(__file__), 'HIV.csv')
#     df = pd.read_csv(filename)    
#     target_cols = ['HIV_active']
#     smiles_col = 'smiles'
#     N = len(df)
#     if check_smiles:
#         mols = df[smiles_col].apply(to_mol)
#         df = df.iloc[mols[~mols.isna()].index]
#         df = df.reset_index(drop=True)
#         M = len(df)

#         if N != M:
#             print("%s invalid smiles are removed" % (N-M))
#     return data(df, smiles_col, target_cols, task_name, task_type, description)


# def load_MUV():
#     task_name = 'MUV'
#     task_type = 'classification'
#     description = """The MUV(Maximum Unbiased Validation) dataset contains 93087 compounds with 17 challenging tasks which is specifically designed for validation of virtual screening techniques."""

    
#     filename = os.path.join(os.path.dirname(__file__), 'muv.csv')
#     df = pd.read_csv(filename)
#     target_cols =   ['MUV-466',
#                      'MUV-548',
#                      'MUV-600',
#                      'MUV-644',
#                      'MUV-652',
#                      'MUV-689',
#                      'MUV-692',
#                      'MUV-712',
#                      'MUV-713',
#                      'MUV-733',
#                      'MUV-737',
#                      'MUV-810',
#                      'MUV-832',
#                      'MUV-846',
#                      'MUV-852',
#                      'MUV-858',
#                      'MUV-859']
#     smiles_col = 'smiles'
#     return data(df, smiles_col, target_cols, task_name, task_type, description)    



# def load_Tox21():
#     task_name = 'Tox21'
#     task_type = 'classification'
#     description = """The Tox21 dataset contains 7831 compounds and corresponding toxicity data against 12 targets."""

#     filename = os.path.join(os.path.dirname(__file__), 'tox21.csv')
#     df = pd.read_csv(filename)
#     target_cols = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
#                     'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
#     smiles_col = 'smiles'
#     return data(df, smiles_col, target_cols, task_name, task_type, description)




# def load_SIDER():
#     task_name = 'SIDER'
#     task_type = 'classification'
#     description = """The SIDER dataset contains 1427 marketed drugs and their adverse drug reactions (ADR) against 27 System-Organs Class."""

    
#     filename = os.path.join(os.path.dirname(__file__), 'sider.csv')
#     df = pd.read_csv(filename)
#     target_cols = ['SIDER1', 'SIDER2', 'SIDER3', 'SIDER4', 'SIDER5', 'SIDER6', 'SIDER7',
#                    'SIDER8', 'SIDER9', 'SIDER10', 'SIDER11', 'SIDER12', 'SIDER13',
#                    'SIDER14', 'SIDER15', 'SIDER16', 'SIDER17', 'SIDER18', 'SIDER19',
#                    'SIDER20', 'SIDER21', 'SIDER22', 'SIDER23', 'SIDER24', 'SIDER25',
#                    'SIDER26', 'SIDER27']
#     smiles_col = 'smiles'
#     return data(df, smiles_col, target_cols, task_name, task_type, description)





# def load_CYP450():
#     task_name = 'CYP450'
#     task_type = 'classification'
#     description = """The CYP450 dataset contains 16896 compounds against of five main CYP450 isozymes: 1A2, 2C9, 2C19, 2D6, and 3A4. This data should split training and test set by aids"""
#     filename = os.path.join(os.path.dirname(__file__), 'cyp450-processed.csv')
#     df = pd.read_csv(filename)
#     target_cols = ['label_1a2', 'label_2c9', 'label_2c19', 'label_2d6', 'label_3a4']
#     smiles_col = 'smiles'
#     return data(df, smiles_col, target_cols, task_name, task_type, description)



# def load_ToxCast():
#     task_name = 'ToxCast'
#     task_type = 'classification'
#     description = """The ToxCast dataset contains 8597 compounds and corresponding binary toxicity levels over 600 experiments."""

#     filename = os.path.join(os.path.dirname(__file__), 'toxcast_data.csv')
#     df = pd.read_csv(filename)
#     target_cols = df.columns[1:].tolist()
#     smiles_col = 'smiles'
#     return data(df, smiles_col, target_cols, task_name, task_type, description)

