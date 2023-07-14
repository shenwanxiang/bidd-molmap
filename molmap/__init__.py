__version__ = '1.3.9.7'

## to custom map object
from molmap.agg import AggMolMap

## cpds feature map
from molmap.map import MolMap
from molmap.cpd import GlobCpdMolMap

## pdb feature map
from molmap.pdb import PDB2Fmap, PDB2Img

## aa sequence feature map
from molmap.seq import LocalAASeqMolMap
from molmap.seq import GlobAASeqMolMap

## na sequence feature map
from molmap.seq import GlobNASeqMolMap


from joblib import load as loadmap

