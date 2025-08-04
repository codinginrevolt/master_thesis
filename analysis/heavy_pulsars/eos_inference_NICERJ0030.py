import sys
import os
import tqdm
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import pandas as pd
import scipy.integrate as integrate

##########
# CONFIG #
##########

# path to the eos files
eos_path_macro = "../../EOS_analysis/eos/samples/MRL"

# outfile
outfile = "./eos_likelihood_NICER_PSRJ0030"

# M-R samples from the NICER data releases in https://doi.org/10.5281/zenodo.3386448 and https://doi.org/10.5281/zenodo.3473465
maryland_path = "/home/aya/work/hkoehn/EOS_analysis/inference_pulsars/data/J0030/J0030_RM_maryland.txt"
amsterdam_path = "/home/aya/work/hkoehn/EOS_analysis/inference_pulsars/data/J0030/A_NICER_VIEW_OF_PSR_J0030p0451/ST_PST/ST_PST__M_R.txt"

#==========================================================#


#load the radius-mass posterior samples from the data
maryland_samples = pd.read_csv(maryland_path, sep=" ", names=["R", "M", "weight"] , skiprows = 6)
if pd.isna(maryland_samples["weight"]).any():
	print("Warning: weights not properly specified, assuming constant weights instead.")
	maryland_samples["weight"] = np.ones(len(maryland_samples["weight"]))
amsterdam_samples = pd.read_csv(amsterdam_path, sep=" ", names=["weight", "twotimeslogl", "M", "R"])


#get a smooth kde interpolator for the posterior
maryland_posterior= stats.gaussian_kde([maryland_samples["M"], maryland_samples["R"]], weights = maryland_samples["weight"])
amsterdam_posterior= stats.gaussian_kde([amsterdam_samples["M"], amsterdam_samples["R"]], weights = amsterdam_samples["weight"])


def log_likelihood(EOSID: int):
        
        R, M = np.loadtxt(os.path.join(eos_path_macro, f'{EOSID}.dat'), usecols = [0,1], unpack=True) #loads the R-M curve
        R = np.atleast_1d(R)
        M = np.atleast_1d(M)
        m_vals = np.arange(0, M.max(), 0.02)
        r_vals = np.interp(m_vals, M, R)
        logy_maryland = maryland_posterior.logpdf(np.vstack([m_vals, r_vals]))
        logl_maryland = scipy.special.logsumexp(logy_maryland)+np.log(0.02)
        logy_amsterdam = amsterdam_posterior.logpdf(np.vstack([m_vals, r_vals]))
        logl_amsterdam = scipy.special.logsumexp(logy_amsterdam)+np.log(0.02)
        return logl_maryland, logl_amsterdam

def main():

    files = next(os.walk(eos_path_macro))[2]
    NEOS=len(files)
    
    
    #split the EOS workload differently
    splits = np.array_split(np.arange(1, NEOS+1), size)
    work = splits[rank].copy()
    save_maryland=[]
    save_amsterdam=[]
    
    print("This is processor ", rank, "and I am doing EOS from ", work[0], " to ", work[-1], ".")

    #loop over the EOSs and calculate the likelihood for each
    comm.Barrier()
    for EOSID in tqdm.tqdm(work):
        logl_maryland, logl_amsterdam = log_likelihood(EOSID)
        save_maryland.append(logl_maryland)
        save_amsterdam.append(logl_amsterdam)
    comm.Barrier()
    
    # gather the results at root
    save_maryland = comm.gather(save_maryland,root=0)
    save_amsterdam = comm.gather(save_amsterdam,root=0)
    
    if rank==0:
       logL_maryland = np.array([l for sublist in save_maryland for l in sublist]) #flatten the save list
       logL_maryland -= scipy.special.logsumexp(logL_maryland)
       logL_amsterdam = np.array([l for sublist in save_amsterdam for l in sublist])
       logL_amsterdam -= scipy.special.logsumexp(logL_amsterdam)
    
       L_maryland = np.exp(logL_maryland)
       L_amsterdam = np.exp(logL_amsterdam)
       L = 1/2*(L_maryland+L_amsterdam)
       np.savetxt(outfile, np.array([L, L_maryland, L_amsterdam]).T)
    
    else:
    	None
    
    
if __name__ == "__main__":
    main()