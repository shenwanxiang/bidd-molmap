#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 20:29:36 2019

@author: charleshen

The calculation of 63 molecular constitutional indices based on its topological
structure.

Including MQNs(molecular quantum numbers.) Nguyen et al. ChemMedChem 4:1803-5 (2009)
"""




from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def CalculateHydrogenNumber(mol):

    i=0
    Hmol=Chem.AddHs(mol)
    for atom in Hmol.GetAtoms():
        if atom.GetAtomicNum()==1:
            i=i+1
            
    return i

def CalculateHalogenNumber(mol):

    i=0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum()==9 or atom.GetAtomicNum()==17 or atom.GetAtomicNum()==35 or atom.GetAtomicNum()==53:
            i=i+1
    return i
            

       
def CalculateAromaticBondNumber(mol):
    i=0
    for bond in mol.GetBonds():
        if bond.GetBondType().name=='AROMATIC':
            i=i+1
            
    return i

def CalculateAllAtomNumber(mol):
    return Chem.AddHs(mol).GetNumAtoms()
        




_MQN_NAMES = [## Atoms
            'NumCarbonAtoms',
            'NumFluorineAtoms',
            'NumChlorineAtoms',
            'NumBromineAtoms',
            'NumIodineAtoms',
            'NumSulfurAtoms',
            'NumPhosphorousAtoms',
            'NumAcyclicNitrogenAtoms',
            'NumCyclicNitrogenAtoms',
            'NumAcyclicOxygenAtoms',
            'NumCyclicOxygenAtoms',
            'NumHeavyAtoms',

            ##Bonds
            'NumAcyclicSingleBonds',
            'NumAcyclicDoubleBonds',
            'NumAcyclicTripleBonds',
            'NumCyclicSingleBonds',
            'NumCyclicDoubleBonds',
            'NumCyclicTripleBonds',
            'NumRotatableBonds',

            ##Polarity
            'NumHydrogenBondAcceptorSites',
            'NumHydrogenBondAcceptorAtoms',
            'NumHydrogenBondDonorSites',
            'NumHydrogenBondDonorAtoms',
            'NegativeCharges',
            'PositiveCharges',

            #nodes
            'AcyclicSingleValentNodes',
            'AcyclicDivalentNodes',
            'AcyclicTrivalentNodes',
            'AcyclicTetravalentNodes',
            'CyclicDivalentNodes',
            'CyclicTrivalentNodes',
            'CyclicTetravalentNodes',

            #Rings
            '3MemberedRings',
            '4MemberedRings',
            '5MemberedRings',
            '6MemberedRings',
            '7MemberedRings',
            '8MemberedRings',
            '9MemberedRings',
            '10MemberedRings',

            'NodesSharedby2Rings',
            'EdgesSharedby2Rings'] 

def _MolQuantumNumbers(m):
    '''
    MQN : (molecular quantum numbers) Nguyen et al. ChemMedChem 4:1803-5 (2009)
    '''   
    
    X = rdMolDescriptors.MQNs_(m)
    return dict(zip(_MQN_NAMES, X))



_constitutional={    
                 ## Atoms
                'NumHydrogenAtoms':CalculateHydrogenNumber,
                'NumHalogenAtoms':CalculateHalogenNumber,
                'NumHeteroAtoms':Chem.rdMolDescriptors.CalcNumHeteroatoms,                 
                'NumBridgeheadAtoms': Chem.rdMolDescriptors.CalcNumBridgeheadAtoms,
                'NumSpiroAtoms': Chem.rdMolDescriptors.CalcNumSpiroAtoms,
                'NumAllAtoms':CalculateAllAtomNumber,
                 
                 ## Rings
                'NumRings':Chem.rdMolDescriptors.CalcNumRings,                 
                'NumAromaticRings': Chem.rdMolDescriptors.CalcNumAromaticRings,                 
                'NumSaturatedRings':Chem.rdMolDescriptors.CalcNumSaturatedRings,
                'NumAliphaticRings': Chem.rdMolDescriptors.CalcNumAliphaticRings,
                 
                 ## Bonds            
                'NumAromaticBonds':CalculateAromaticBondNumber,
                'NumAmideBonds': Chem.rdMolDescriptors.CalcNumAmideBonds,

                #cycles
                'NumHeteroCycles':Chem.rdMolDescriptors.CalcNumHeterocycles,                  
                'NumAliphaticCarbocycles': Chem.rdMolDescriptors.CalcNumAliphaticCarbocycles,
                'NumAliphaticHeterocycles': Chem.rdMolDescriptors.CalcNumAliphaticHeterocycles,
                'NumAromaticCarbocycles':Chem.rdMolDescriptors.CalcNumAromaticCarbocycles,
                'NumAromaticHeterocycles':Chem.rdMolDescriptors.CalcNumAromaticHeterocycles,
                'NumSaturatedCarbocycles': Chem.rdMolDescriptors.CalcNumSaturatedCarbocycles,
                'NumSaturatedHeterocycles': Chem.rdMolDescriptors.CalcNumSaturatedHeterocycles, 
                
                #centers
                'NumAtomStereoCenters': Chem.rdMolDescriptors.CalcNumAtomStereoCenters,
                'NumUnspecifiedAtomStereoCenters': Chem.rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters,
                }


from collections import OrderedDict
def GetConstitution(mol):
    """
    #################################################################
    Get the dictionary of constitutional descriptors for given moelcule mol
    
    Usage:
        
        result=GetConstitutional(mol)
        
        Input: mol is a molecule object.
        
        Output: result is a dict form containing all constitutional values.
    #################################################################
    """
    result=OrderedDict()
    for k, func in _constitutional.items():
        result[k]= func(mol)
        
    _MQN = _MolQuantumNumbers(mol)
    result.update(_MQN)
    
    return result


_ConstitutionNames = list(_constitutional.keys())
_ConstitutionNames.extend(_MQN_NAMES)


#############################################################

if __name__ =='__main__':
    
    import pandas as pd
    from tqdm import tqdm
    
    smis = ['C'*(i+1) for i in range(100)]
    x = []
    for index, smi in tqdm(enumerate(smis), ascii=True):
        m = Chem.MolFromSmiles(smi)
        x.append(GetConstitution(m))
        
    pd.DataFrame(x)

    
