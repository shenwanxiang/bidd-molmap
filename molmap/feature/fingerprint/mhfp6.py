"""
Probst, Daniel, and Jean-Louis Reymond. "A probabilistic molecular fingerprint for big data settings." Journal of cheminformatics 10.1 (2018): 66.'

orignal code: https://github.com/reymond-group/mhfp

"""

from mhfp.encoder import MHFPEncoder


def GetMHFP6(mol, nBits=2048, radius = 3): 
    """
    MHFP6: radius=3
    """
    encoder = MHFPEncoder(n_permutations = nBits)
    hash_values = encoder.encode_mol(mol, radius=radius, rings=True, kekulize=True, min_radius=1) 
    arr = encoder.fold(hash_values, nBits)
    return arr.astype(bool)