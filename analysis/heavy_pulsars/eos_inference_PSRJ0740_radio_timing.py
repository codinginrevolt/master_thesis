#!/home/koehn/anaconda3/envs/nmma/bin/python3
import sys
import os, os.path
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
outfile = "./eos_likelihood_PSRJ0740"

# gaussian approximation for the mass posterior of 2104.00880
data_posterior = scipy.stats.norm(loc =2.08, scale =0.07)

#==========================================================#

def log_likelihood(EOSID: int):
    R, M = np.loadtxt(os.path.join(eos_path_macro, f"{EOSID}.dat"), usecols=[0,1], unpack=True)
    R = np.atleast_1d(R)
    M = np.atleast_1d(M)
    MTOV = M.max()
    logl = data_posterior.logcdf(MTOV)
    return logl
 
def main():
     
    files = next(os.walk(eos_path_macro))[2]
    NEOS=len(files)
    
    
    # split the EOS workload differently
    splits = np.array_split(np.arange(1, NEOS+1), size)
    work = splits[rank].copy()
    save = []
    
    print("This is processor ", rank, "and I am doing EOS from ", work[0], " to ", work[-1], ".")

    # loop over the EOSs and calculate the likelihood for each
    comm.Barrier()
    for EOSID in work:
        log_l = log_likelihood(EOSID)
        save.append(np.exp(log_l))
    comm.Barrier()
    
    # gather the results at root
    save = comm.gather(save,root=0)
    
    if rank==0:
       L = np.array([l for sublist in save for l in sublist]) #flatten the save list
       L *= 1/np.sum(L)
       np.savetxt(outfile, L)
    
    else:
    	None
     

if __name__ == "__main__":
    main()
