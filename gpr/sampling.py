import numpy as np
from scipy.stats import loguniform

import kernels
import gaussianprocess
import prepare_pqcd as pp
from constants import get_phi

def get_hype_samples():
    """
    Hyperparams for GP mean and kernel, as well as pqcd renormalisation scale
    """

    rng = np.random.default_rng()
    nu = rng.normal(1.25, 0.2)

    while True:
        l = rng.normal(1.0, 0.5) # note that std here is different than the ones in the qcd papers
        if l>0: break


    while True:
        cs2_hat = rng.normal(0.5, 0.25)
        if 0 <= cs2_hat <= 1:  # sound speed should be between 0 and 1
            break


    X = loguniform.rvs(0.5, 2)


    return cs2_hat, nu, l, X

def get_hype_n_ceft_end():
    rng = np.random.default_rng()
    nc_hat = rng.uniform(1,2)
    return nc_hat

def generate_sample(n_ceft, cs2_ceft_avg, phi_ceft_sigma, n_crust, cs2_crust, x_test_end = 10, mu_low = 2.2, mu_high = 2.8, point_nums=200, ceft_end = 0):
    """
    input n must be in nsat, if used in conjunction with make_condition_eos() that is automatically the case
    out n in nsat
    """

    cs2_hat, nu_hat, l_hat, X_hat = get_hype_samples()

    kernel = kernels.Kernel('SE', sigma=nu_hat**0.5, l=l_hat)

    if ceft_end == 0:
        n_ceft_end_hat = get_hype_n_ceft_end()
    else:
        n_ceft_end_hat = ceft_end
    
    idx = np.searchsorted(n_ceft, n_ceft_end_hat)
    before_or_after = np.argmin([np.abs(n_ceft[idx-1]-n_ceft_end_hat), np.abs(n_ceft[idx]-n_ceft_end_hat)])
    if before_or_after == 1:
        idx = idx+1
    
    n_ceft = n_ceft[:idx]
    cs2_ceft_avg = cs2_ceft_avg[:idx]
    phi_ceft_sigma = phi_ceft_sigma[:idx]

    n_pqcd, cs2_pqcd = pp.get_pqcd(X_hat, mu_low, mu_high, size=100) # nsat, unitless

    x_train =  np.concatenate((n_crust[-10:-1],n_ceft, n_pqcd)) #nsat
    cs2_train =  np.concatenate((cs2_crust[-10:-1],cs2_ceft_avg, cs2_pqcd)) # crust excluding last element because it is already in ceft

    phi_pqcd_sigma = np.zeros_like(cs2_pqcd)
    phi_crust_sigma = np.zeros_like(cs2_crust[-10:-1])

    phi_sigma_train = np.concatenate((phi_crust_sigma, phi_ceft_sigma, phi_pqcd_sigma))
    phi_train = get_phi(cs2_train)
    train_noise = phi_sigma_train**2

    x_test = np.linspace(n_ceft[0], x_test_end, point_nums) # number density in nsat, starting val is ending val of n crust, ending val is default 10 nsat


    gp = gaussianprocess.GP(kernel, get_phi(cs2_hat))
    gp.fit(x_train, x_test, phi_train, var_f = train_noise, stabilise=True)

    phi_test = gp.posterior(sampling=True)

    phi_test = phi_test.flatten()
    phi_test[0] = get_phi(cs2_crust[-1]) # minor difference in gpr result and actual crust ending so replacing gpr val

    return phi_test, x_test, X_hat
