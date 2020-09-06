"""
MinHashed Atom-pair Fingerprint, MAP
orignal paper: Capecchi, Alice, Daniel Probst, and Jean-Louis Reymond. "One molecular fingerprint to rule them all: drugs, biomolecules, and the metabolome." Journal of Cheminformatics 12.1 (2020): 1-15. orignal code: https://github.com/reymond-group/map4, thanks their orignal work

A small bug is fixed: https://github.com/reymond-group/map4/issues/6
"""

_type = 'topological-based'


import itertools
from collections import defaultdict

import tmap as tm
from mhfp.encoder import MHFPEncoder
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdmolops import GetDistanceMatrix


def to_smiles(mol):
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


class MAP4Calculator:

    def __init__(self, dimensions=2048, radius=2, is_counted=False, is_folded=False, fold_dimensions = 2048):
        """
        MAP4 calculator class
        """
        self.dimensions = dimensions
        self.radius = radius
        self.is_counted = is_counted
        self.is_folded = is_folded
        self.fold_dimensions = fold_dimensions
        
        if self.is_folded:
            self.encoder = MHFPEncoder(dimensions)
        else:
            self.encoder = tm.Minhash(dimensions)

    def calculate(self, mol):
        """Calculates the atom pair minhashed fingerprint
        Arguments:
            mol -- rdkit mol object
        Returns:
            tmap VectorUint -- minhashed fingerprint
        """
        
        atom_env_pairs = self._calculate(mol)
        if self.is_folded:
            return self._fold(atom_env_pairs)
        return self.encoder.from_string_array(atom_env_pairs)

    def calculate_many(self, mols):
        """ Calculates the atom pair minhashed fingerprint
        Arguments:
            mols -- list of mols
        Returns:
            list of tmap VectorUint -- minhashed fingerprints list
        """

        atom_env_pairs_list = [self._calculate(mol) for mol in mols]
        if self.is_folded:
            return [self._fold(pairs) for pairs in atom_env_pairs_list]
        return self.encoder.batch_from_string_array(atom_env_pairs_list)

    def _calculate(self, mol):
        return self._all_pairs(mol, self._get_atom_envs(mol))

    def _fold(self, pairs):
        fp_hash = self.encoder.hash(set(pairs))
        return self.encoder.fold(fp_hash, self.fold_dimensions)

    def _get_atom_envs(self, mol):
        atoms_env = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            for radius in range(1, self.radius + 1):
                if idx not in atoms_env:
                    atoms_env[idx] = []
                atoms_env[idx].append(MAP4Calculator._find_env(mol, idx, radius))
        return atoms_env

    @classmethod
    def _find_env(cls, mol, idx, radius):
        env = rdmolops.FindAtomEnvironmentOfRadiusN(mol, radius, idx)
        atom_map = {}

        submol = Chem.PathToSubmol(mol, env, atomMap=atom_map)
        if idx in atom_map:
            smiles = Chem.MolToSmiles(submol, rootedAtAtom=atom_map[idx], canonical=True, isomericSmiles=False)
            return smiles
        return ''

    def _all_pairs(self, mol, atoms_env):
        atom_pairs = []
        distance_matrix = GetDistanceMatrix(mol)
        num_atoms = mol.GetNumAtoms()
        shingle_dict = defaultdict(int)
        for idx1, idx2 in itertools.combinations(range(num_atoms), 2):
            dist = str(int(distance_matrix[idx1][idx2]))

            for i in range(self.radius):
                env_a = atoms_env[idx1][i]
                env_b = atoms_env[idx2][i]

                ordered = sorted([env_a, env_b])

                shingle = '{}|{}|{}'.format(ordered[0], dist, ordered[1])

                if self.is_counted:
                    shingle_dict[shingle] += 1
                    shingle += '|' + str(shingle_dict[shingle])

                atom_pairs.append(shingle.encode('utf-8'))
        return list(set(atom_pairs))

    

def GetMAP4(mol, nBits=2048, radius = 2, fold_dimensions = None): 
    
    """
    MAP4: radius=2
    """
    if fold_dimensions == None:
        fold_dimensions = nBits

    calc = MAP4Calculator(dimensions=nBits, radius=radius, is_counted=False, is_folded=True, fold_dimensions = fold_dimensions)
    
    arr = calc.calculate(mol)
    
    return arr.astype(bool)