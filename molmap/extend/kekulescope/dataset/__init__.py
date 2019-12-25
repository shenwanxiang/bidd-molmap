import pandas as pd
from rdkit import Chem
import os


cell_lines = ['A2780', 
            'CCRF-CEM',
            'DU-145', 
            'HCT-15',
            'KB',
            'LoVo',
            'PC-3',
            'SK-OV-3']

targets = ['A2a',
         'ABL1',
         'Acetylcholinesterase',
         'Androgen',
         'Aurora-A',
         'B-raf',
         'COX-1',
         'COX-2',
         'Cannabinoid',
         'Carbonic',
         'Caspase',
         'Coagulation',
         'Dihydrofolate',
         'Dopamine',
         'Ephrin',
         'Estrogen',
         'Glucocorticoid',
         'Glycogen',
         'HERG',
         'HL-60',
         'JAK2',
         'K562',
         'L1210',
         'LCK',
         'MDA-MB-231',
         'Monoamine',
         'Vanilloid',
         'erbB1',
         'opioid']


def load_data(name):
    filename = os.path.join(os.path.dirname(__file__), name, name + '.sdf')
    suppl = Chem.SDMolSupplier(filename)
    mols = [x for x in suppl if x is not None]
    my_smiles=[Chem.MolToSmiles(submol) for submol in mols]
    chembl_ids=[m.GetProp("ChEMBL_ID") for m in mols]
    activities =[float(m.GetProp("pIC50")) for m in mols]

    df = pd.DataFrame([my_smiles, chembl_ids, activities], index = ['smiles', 'chembel_ids', 'pIC50']).T
    return df