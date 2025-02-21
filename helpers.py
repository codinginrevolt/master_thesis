import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

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

def CI_to_sigma(width, CI):
    """
    From any given CI to a sigma value.
    CI: confidence interval percentage (e.g., 75 for 75%)
    """
    z = stats.norm.ppf(1 - (1 - 75/100) / 2)
    sig = width/(2*z)
    return(sig)

def generate_samples(n_ceft, cs2_ceft_avg, phi_ceft_sigma):
    import dickandballs as db
    import helpers as hel

    cs2_hat, nu_hat, l_hat, X_hat = hel.get_hype_samples()

    kernel = db.Kernel('SE', sigma=nu_hat, l=l_hat)

    n_pqcd, cs2_pqcd = hel.get_pqcd(X_hat, size=100)

    x_train =  np.concatenate((n_ceft, n_pqcd))
    cs2_train =  np.concatenate((cs2_ceft_avg, cs2_pqcd))

    phi_pqcd_sigma = np.zeros_like(cs2_pqcd)
    phi_sigma_train = np.concatenate((phi_ceft_sigma, phi_pqcd_sigma))
    phi_train = hel.get_phi(cs2_train)
    train_noise = phi_sigma_train**2

    x_test = np.linspace(n_ceft[0], n_pqcd[-1], 200) # number density, starting val is ending val of n crust


    gp = db.GP(kernel, hel.get_phi(cs2_hat))
    gp.fit(x_train, x_test, phi_train, var_f = train_noise, stabilise=True)

    phi_test, sig = gp.posterior()

    return phi_test.flatter()