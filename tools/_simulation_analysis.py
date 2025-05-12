import numpy as np
import bz2

def load_ancestral_karyotype(filename, min_genes = 200):
    with bz2.open(filename, mode='rb') as f:
        ancestral_genome = [int(line.decode().split('\t')[0]) for line in f.readlines()]
    x_anc, y_anc = np.unique(ancestral_genome,return_counts=True)
    chrom_sizes = np.sort(y_anc[y_anc >= min_genes])[::-1]
    chrom_indices = np.arange(1,chrom_sizes.shape[0]+1)
    return chrom_indices, chrom_sizes

def genome_from_karyotype(karyotype):
    genome = []
    n = 1
    for chrom in karyotype:
        genome.append(list(range(n,n+chrom)))
        n += chrom
    return genome

def convolve_blocks(block1, block2, min_synteny = 1):
    block1 = np.array(block1)
    block2 = np.array(block2)
    l1 = block1.shape[0]
    l2 = block2.shape[0]
    if l1 > l2:
        b1 = np.copy(block2)
        L1 = b1.shape[0]
        b2 = np.copy(block1)
        L2 = b2.shape[0]
    else:
        b1 = np.copy(block1)
        L1 = b1.shape[0]
        b2 = np.copy(block2)
        L2 = b2.shape[0]

    convolution = []
    for l in range(min_synteny,L1):
        overlap = (b1[0:l] == b2[L2-l:L2])
        if np.sum(overlap) >= min_synteny:
            where_overlap = np.where(overlap)[0]
            convolution.append(b1[0:l][where_overlap])

    for l in range(L2-L1+1):
        overlap = (b1 == b2[L2-(L1+l):L2-l])
        if np.sum(overlap) >= min_synteny:
            where_overlap = np.where(overlap)[0]
            convolution.append(b1[where_overlap])

    for l in range(min_synteny,L1)[::-1]:
        overlap = (b1[L1-l:L1] == b2[0:l])
        if np.sum(overlap) >= min_synteny:
            where_overlap = np.where(overlap)[0]
            convolution.append(b1[L1-l:L1][where_overlap])

    return convolution

### TO BE FIXED...
def block_overlap(block1, block2, min_synteny = 1):
    convolution_forward = convolve_blocks(block1,block2,min_synteny=min_synteny)
    convolution_backward = convolve_blocks(block1,block2[::-1],min_synteny=min_synteny)
    
    if (len(convolution_forward) == 1) and (len(convolution_backward) == 1):
        convolution = convolution_forward
    else:
        convolution = convolution_forward + convolution_backward
        
    #convolution = convolution_forward + convolution_backward

    family = [set(c) for c in convolution]
    out_family = [set(c) for c in convolution]
    for set1 in family:
        for set2 in family:
            if set1 != set2:
                if len(set1.intersection(set2)) > 0:
                    N1 = len(set1)
                    N2 = len(set2)
                    if (set1 in out_family) and (set2 in out_family):
                        if N1 >= N2:
                            out_family.remove(set2)
                        else:
                            out_family.remove(set1)

    return out_family

def calculate_simulation_sb_distribution(GA, GB, min_synteny = 1):
    sim_block_sizes = []
    for chromA in GA:
        for chromB in GB:
            blocksAB = block_overlap(chromA,chromB,min_synteny=min_synteny)
            sim_block_sizes += [len(b) for b in blocksAB]
    h_sim = np.unique(sim_block_sizes,return_counts=True)
    return h_sim
