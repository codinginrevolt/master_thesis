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
    Returns n_pqcd in nsat units, cs2 unitless
    """

    from pqcd.pQCD import pQCD

    mu_grid = np.linspace(mu_low, mu_high, size)
        
    pQCD_temp = pQCD(X)
    n_pqcd  = np.vectorize(pQCD_temp.number_density)(mu_grid) /0.16 # in nsat
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

def generate_sample(n_ceft, cs2_ceft_avg, phi_ceft_sigma, x_test_end=10, point_nums=200):
    """
    input n must be in nsat, if used in conjunction with make_condition_eos() that is automatically the case
    out n in nsat
    """

    cs2_hat, nu_hat, l_hat, X_hat = get_hype_samples()

    kernel = db.Kernel('SE', sigma=nu_hat, l=l_hat)

    n_pqcd, cs2_pqcd = get_pqcd(X_hat, size=100) # nsat, unitless

    x_train =  np.concatenate((n_ceft, n_pqcd)) # nsat
    cs2_train =  np.concatenate((cs2_ceft_avg, cs2_pqcd))

    phi_pqcd_sigma = np.zeros_like(cs2_pqcd)
    phi_sigma_train = np.concatenate((phi_ceft_sigma, phi_pqcd_sigma))
    phi_train = get_phi(cs2_train)
    train_noise = phi_sigma_train**2

    x_test = np.linspace(x_train[0], x_test_end, point_nums) # number density in nsat, starting val is ending val of n crust, ending val is default 10 nsat


    gp = db.GP(kernel, get_phi(cs2_hat))
    gp.fit(x_train, x_test, phi_train, var_f = train_noise, stabilise=True)

    phi_test = gp.posterior(sampling=True)

    return phi_test.flatten(), x_test

def make_conditioning_eos():
    "returns n in nsat"

    ceft_lower= np.loadtxt('EOS/ceft/eos_ceft_lower.dat')
    n_ceft_lower, p_ceft_lower, e_ceft_lower = ceft_lower.T

    ceft_upper= np.loadtxt('EOS/ceft/eos_ceft_upper.dat')
    _, p_ceft_upper, e_ceft_upper = ceft_upper.T

    n_ceft = n_ceft_lower #fm^-3

    # average CEFT EOS
    e_ceft = (e_ceft_lower+e_ceft_upper)/2 # MeVfm^-3 
    p_ceft = (p_ceft_lower+p_ceft_upper)/2 # MeVfm^-3

    # seperating crust: n_crust = n_ceft[:428] last element overlap
    

    p_ceft_upper = p_ceft_upper[427:]
    p_ceft_lower = p_ceft_lower[427:]

    e_ceft_upper = e_ceft_upper[427:]
    e_ceft_lower = e_ceft_lower[427:]

    n_ceft = n_ceft[427:]
    e_ceft = e_ceft[427:]
    p_ceft = p_ceft[427:]

    # chemical potential
    e_ini = e_ceft[0] 
    p_ini = p_ceft[0] 
    n_ini = n_ceft[0]
    mu_ini = (e_ini + p_ini) / n_ini # MeV

    # sound speed
    cs2_ceft_lower = np.gradient(p_ceft_lower, e_ceft_lower) #dp/de
    cs2_ceft_upper = np.gradient(p_ceft_upper, e_ceft_upper) #dp/de
    cs2_ceft_avg = (cs2_ceft_upper+cs2_ceft_lower)/2

    # phi
    phi_ceft_lower = get_phi(cs2_ceft_lower)
    phi_ceft_upper = get_phi(cs2_ceft_upper)
    phi_ceft_width = phi_ceft_upper-phi_ceft_lower
    phi_ceft_sigma = CI_to_sigma(phi_ceft_width, 75)

    return n_ceft/0.16, cs2_ceft_avg, phi_ceft_sigma, e_ini, p_ini, mu_ini