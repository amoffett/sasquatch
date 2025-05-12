import numpy as np
from scipy import stats
from sasquatch.tools import *

def simulate_neutral_genome_evolution(karyotype, T, kappa, gamma, theta, return_trajectory = False, seed = None):
    G = [genome_from_karyotype(karyotype)]
    N = len(G[0])
    K = np.sum([len(g)-1 for g in G[0]])
    q = 1 - np.exp(-theta)
    if seed == None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed=seed)
        
    t = 0
    t_list = [t]
    while t < T:
        G0 = G[-1]
        dt_kappa = stats.expon.rvs(scale=1/(kappa*K))
        dt_gamma = stats.expon.rvs(scale=1/(gamma*K))
        if dt_kappa < dt_gamma:
            cut = 1
            dt = dt_kappa
        else:
            cut = 2
            dt = dt_gamma        
        p_chrom = np.array([len(g)-1 for g in G0])
        p_chrom = p_chrom / p_chrom.sum()
        r_chrom = rng.random()
        chrom_choice = np.digitize(r_chrom,np.cumsum(p_chrom))
        cut_chrom = G0[chrom_choice]
        if cut == 1:
            cut_choice = np.random.choice(range(1,len(cut_chrom)))
            chrom_piece_A = cut_chrom[:cut_choice]
            chrom_piece_B = cut_chrom[cut_choice:]
            chrom_piece_to_fuse = np.random.choice([0,1])
            r_chrom_fuse = rng.random()
            chrom_choice_fusion = np.digitize(r_chrom_fuse,np.arange(1,N+1)/N)
            side_choice_fusion = np.random.choice([0,1])
            G1 = []
            for chrom in range(N):
                if (chrom == chrom_choice) and (chrom != chrom_choice_fusion):
                    if chrom_piece_to_fuse == 1:
                        G1.append(chrom_piece_A)
                    else:
                        G1.append(chrom_piece_B)
                elif (chrom == chrom_choice) and (chrom == chrom_choice_fusion):
                    if side_choice_fusion == 0:
                        if chrom_piece_to_fuse == 0:
                            G1.append(chrom_piece_A + chrom_piece_B)
                        else:
                            G1.append(chrom_piece_B + chrom_piece_A)
                    else:
                        if chrom_piece_to_fuse == 0:
                            G1.append(chrom_piece_B + chrom_piece_A)
                        else:
                            G1.append(chrom_piece_A + chrom_piece_B)                
                elif (chrom != chrom_choice) and (chrom == chrom_choice_fusion):
                    if side_choice_fusion == 0:
                        if chrom_piece_to_fuse == 0:
                            G1.append(chrom_piece_A + G0[chrom_choice_fusion])
                        else:
                            G1.append(chrom_piece_B + G0[chrom_choice_fusion])
                    else:
                        if chrom_piece_to_fuse == 0:
                            G1.append(G0[chrom_choice_fusion] + chrom_piece_A)
                        else:
                            G1.append(G0[chrom_choice_fusion] + chrom_piece_B) 
                else:
                    G1.append(G0[chrom])
        elif cut == 2:
            cut_choice = np.random.choice(range(0,len(cut_chrom)+1))
            chrom_piece_A = cut_chrom[:cut_choice]
            chrom_piece_BC = cut_chrom[cut_choice:]            
            p_cut_size = stats.geom.pmf(np.arange(1,len(chrom_piece_BC)+1),q)
            p_cut_size = p_cut_size / p_cut_size.sum()
            r_cut_size = rng.random()
            cut_size = np.digitize(r_cut_size,np.cumsum(p_cut_size)) + 1
            chrom_piece_B = chrom_piece_BC[:cut_size][::-1]
            chrom_piece_C = chrom_piece_BC[cut_size:]
            
            new_chrom_piece_A = chrom_piece_A + chrom_piece_C
            new_chrom_piece_B = chrom_piece_B
            insertion_choice = np.random.choice(range(0,len(new_chrom_piece_A)+1))
            new_chrom = new_chrom_piece_A[:insertion_choice] + new_chrom_piece_B + new_chrom_piece_A[insertion_choice:]
            G1 = []
            for chrom in range(N):
                if (chrom == chrom_choice):
                    G1.append(new_chrom)
                else:
                    G1.append(G0[chrom])
        
        G.append(G1)
        t += dt
        t_list.append(t)
    if t > T:
        t_list.append(t)
        G.append(G[-1])
        
    if return_trajectory == False:
        return t, G[-1]
    else:
        return t_list, G
