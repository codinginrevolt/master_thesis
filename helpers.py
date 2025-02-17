import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import loguniform

from pqcd.pQCD import pQCD

import dickandballs as db

def get_hype_samples():
    """
    Hyperparams for GP mean and kernel, as well as pqcd renormalisation scale
    """

    rng = np.random.default_rng()
    nu = rng.normal(1.25, 0.2)
    l = rng.normal(1, 0.5) # note that std here is different than the ones in the qcd papers

    cs2_hat = rng.normal(0.5,0.25)
    while cs2_hat<0 or cs2_hat>1: # sound speed shouldnt be negative or more than c
        cs2_hat = rng.normal(0.5,0.25)        

    X = loguniform.rvs(0.5, 2)


    return cs2_hat, nu, l, X

def get_pqcd(X: float, mu_low: float = 2.2, mu_high:float = 2.8, size: int = 100):
    """
    Retrieve pQCD calculations of number density (in n_sat) and sound speed squared using an X value (renormalisation scale)
    Default mu from 2.2 GeV to 3 GeV
    """

    from pqcd.pQCD import pQCD

    mu_grid = np.linspace(mu_low, mu_high, size)
        
    pQCD_temp = pQCD(X)
    n_pqcd  = np.vectorize(pQCD_temp.number_density)(mu_grid)/0.16 #nsat
    cs2_pqcd = np.vectorize(pQCD_temp.speed2)(mu_grid)
    
    return n_pqcd, cs2_pqcd

def get_phi(cs2):
    return -np.log(1/cs2 - 1)

def CI_to_sigma(width):
    """From CI 75% to a sigma value
    z*sigma = width/2
    TODO: automate z score so any CI can be taken"""
    z = 1.15
    sig = width/(2*z)
    return(sig)