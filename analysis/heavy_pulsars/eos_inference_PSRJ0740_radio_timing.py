import os, os.path
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import numpy as np
import scipy
import argparse



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

        outdir = os.path.dirname(outfile)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        
        np.savetxt(outfile, L)
    
    else:
    	None
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate EOS likelihoods from PSRJ0740 radio timing data")
    parser.add_argument("set_name", help="dataset name, e.g. SE_10ns_3")
    args = parser.parse_args()
    set_name = args.set_name

    ##########
    # CONFIG #
    ##########

    # path to the eos files
    eos_path_macro = f"/home/sam/thesis/code/results/tov_res/{set_name}/"

    # outfile
    outfile = f"/home/sam/thesis/code/results/pulsars/{set_name}/eos_likelihood_PSRJ0740"
    outdir = os.path.dirname(outfile)

    main()
