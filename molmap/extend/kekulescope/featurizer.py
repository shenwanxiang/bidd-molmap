from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np


Draw.DrawingOptions.bondLineWidth = 1.3
Draw.DrawingOptions.atomLabelFontSize = 18


def Smiles2Img(smiles, size = (224,224), addHs = True):

    mol = Chem.MolFromSmiles(smiles)
    if addHs:
        mol = Chem.AddHs(mol)
    img = Draw.MolToImage(mol,  kekulize=True, size= size, fitImage = True)
    img = img.convert('RGB')
    
    return img