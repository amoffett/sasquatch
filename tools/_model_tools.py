import numpy as np
from GenomicTools.synteny import *
import pickle as pkl

def calculate_synteny_distributions(synteny_filenames):
    synteny_distributions = {}
    n = 0
    for fname in synteny_filenames: 
        spA, spB, blocks, labels = load_synteny_blocks(fname)
        h = np.unique([b.shape[0] for b in blocks],return_counts=True)
        synteny_distributions[(spA,spB)] = h
        n += 1
    return synteny_distributions

def fetch_data(spA, spB, data):
    if (spA,spB) in data.keys():
        return data[(spA,spB)]
    elif (spB,spA) in data.keys():
        return data[(spB,spA)]
    else:
        raise ValueError("Distribution for %s-%s comparison is not here..."%(spA,spB))

def find_N(species_list, b0):
    N = 0
    for sp in species_list:
        N = np.max([N,b0[sp].shape[0]])
    return N

def adjust_b0(species_list, b0, N):
    b0_adjusted = {}
    for sp in species_list:
        N_sp = b0[sp].shape[0]
        b0_adjusted[sp] = np.zeros(N)
        b0_adjusted[sp][range(N_sp)] = b0[sp]
    return b0_adjusted
