import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d

import kernels
import gaussianprocess
from constants import ns, crust_end, get_phi, crust_end_old
import prepare_pqcd as pp
import eos

from pathlib import Path
base_dir = Path(__file__).resolve().parent

def CI_to_sigma(width, CI = 75):
    """
    TODO: enable the same from any given CI to a sigma value.
    CI: confidence interval percentage (e.g., 75 for 75%)
    """
    z = norm.ppf(1 - (1 - CI/100) / 2)
    sig = np.abs(width)/(2*z)
    return(sig)


def smooth_cs2(n_ceft, cs2_ceft, a, b, c, d):
    """
    only needed for old chiEFT band
    """
    end = 200
    cs2_ceft_train = np.concatenate((cs2_ceft[0:a], cs2_ceft[b:c], cs2_ceft[d:end]))
    n_ceft_train = np.concatenate((n_ceft[0:a], n_ceft[b:c], n_ceft[d:end]))
    n_ceft_test = np.concatenate((n_ceft[a:b], n_ceft[c:d]))

    kern = kernels.Kernel('SE', sigma = 0.01, l = 0.01)
    gp = gaussianprocess.GP(kern)
    gp.fit(n_ceft_train, n_ceft_test, cs2_ceft_train, stabilise=True)

    cs2_ceft_test, _ = gp.posterior()
    cs2_ceft_test = cs2_ceft_test.flatten()

    new_cs2_ceft = cs2_ceft.copy()
    new_cs2_ceft[a:b] = cs2_ceft_test[:b-a]
    new_cs2_ceft[c:d] = cs2_ceft_test[b-a:]

    return new_cs2_ceft

def make_conditioning_eos():
    """
    returns 
     - n in nsat
     - cs2_ceft_avg, phi_ceft_sigma, e_ini, p_ini, mu_ini, n_crust/ns, e_crust, p_crust, cs2_crust
    """

    ####### loading the new ceft band #########
    ceft_95 = np.loadtxt(base_dir / 'EOS/chiEFT/chiEFT_band_95_percent_credibility.txt').T
    n, cs2_l, cs2_u = ceft_95
        
    n_ceft = n[crust_end:] # fm^-3
    cs2_l_ceft = cs2_l[crust_end:]
    cs2_u_ceft = cs2_u[crust_end:]

    ###### using GP to interpolate and make datapoints equidistant ######
    n_uniform = np.linspace(n_ceft[0], n_ceft[-1], 100)    
    kern = kernels.Kernel("SE", sigma=0.01, l=0.5)
    gp_l = gaussianprocess.GP(kern)
    gp_l.fit(n_ceft, n_uniform, cs2_l_ceft, stabilise=True)
    cs2_l_uniform = gp_l.posterior(sampling=True)

    gp_u = gaussianprocess.GP(kern)
    gp_u.fit(n_ceft, n_uniform, cs2_u_ceft, stabilise=True)
    cs2_u_uniform = gp_u.posterior(sampling=True)
                                
    n_ceft = n_uniform
    cs2_l_ceft = cs2_l_uniform.flatten()
    cs2_u_ceft = cs2_u_uniform.flatten()

    ####### crust eos #######
    crust = np.loadtxt(base_dir / 'EOS/chiEFT/crust.dat').T
    n_crust = crust[0]
    e_crust = crust[1]
    p_crust = crust[2]
    cs2_crust = crust[3]
    
    e_ini = e_crust[-1]
    p_ini = p_crust[-1]
    mu_ini = (p_ini + e_ini)/n_crust[-1]

    ######### conditioning eos ##########
    cs2_avg_ceft = (cs2_u_ceft+cs2_l_ceft)/2

    phi_l_ceft = get_phi(cs2_l_ceft)
    phi_u_ceft = get_phi(cs2_u_ceft)
    phi_width_ceft = phi_u_ceft-phi_l_ceft
    phi_sigma_ceft = CI_to_sigma(phi_width_ceft, 95)

    return n_ceft/ns, cs2_avg_ceft, phi_sigma_ceft, e_ini, p_ini, mu_ini, n_crust/ns, e_crust, p_crust, cs2_crust

def get_old_ini_vacuum(n=2.72e-14):
    ceft_lower_old = np.loadtxt(base_dir / 'EOS/ceft/eos_ceft_lower.dat')
    n_ceft_lower_old, p_ceft_lower_old, e_ceft_lower_old = ceft_lower_old.T

    ceft_upper_old= np.loadtxt(base_dir / 'EOS/ceft/eos_ceft_upper.dat')
    _, p_ceft_upper_old, e_ceft_upper_old = ceft_upper_old.T

    n_ceft_old, p_ceft_lower_old, e_ceft_lower_old = ceft_lower_old.T


    n_ceft_old = n_ceft_lower_old # fm^-3

    e_ceft_old = (e_ceft_lower_old+e_ceft_upper_old)/2
    p_ceft_old = (p_ceft_lower_old+p_ceft_upper_old)/2

    n_crust_old = n_ceft_old[:crust_end_old+1]
    e_crust_old = e_ceft_old[:crust_end_old+1]
    p_crust_old = p_ceft_old[:crust_end_old+1]

    e_interp_old = interp1d(n_crust_old, e_crust_old)
    p_interp_old = interp1d(n_crust_old, p_crust_old)

    e_ini = e_interp_old(n)
    p_ini = p_interp_old(n)

    return e_ini, p_ini

def make_conditioning_eos_old():
    """
    returns 
     - n in nsat
     - cs2_ceft_avg, phi_ceft_sigma, e_ini, p_ini, mu_ini, n_crust/ns, e_crust, p_crust, cs2_crust
    """
    ceft_lower= np.loadtxt(base_dir / 'EOS/ceft/eos_ceft_lower.dat')
    n_ceft_lower, p_ceft_lower, e_ceft_lower = ceft_lower.T

    ceft_upper= np.loadtxt(base_dir / 'EOS/ceft/eos_ceft_upper.dat')
    _, p_ceft_upper, e_ceft_upper = ceft_upper.T


    n_ceft = n_ceft_lower #fm^-3

    cs2_lower = np.gradient(p_ceft_lower, e_ceft_lower) #dp/de

    cs2_upper = np.gradient(p_ceft_upper, e_ceft_upper)

    cs2_avg = (cs2_upper+cs2_lower)/2

    # average CEFT EOS
    e_ceft = (e_ceft_lower+e_ceft_upper)/2 # MeVfm^-3 
    p_ceft = (p_ceft_lower+p_ceft_upper)/2 # MeVfm^-3

    # seperating crust: n_crust = n_ceft[:428] last element overlap
    n_crust = n_ceft[:crust_end_old+1]
    e_crust = e_ceft[:crust_end_old+1]
    p_crust = p_ceft[:crust_end_old+1]
    cs2_crust = cs2_avg[:crust_end_old+1]

    # seperating ceft proper
    p_ceft_upper = p_ceft_upper[crust_end_old:]
    p_ceft_lower = p_ceft_lower[crust_end_old:]

    e_ceft_upper = e_ceft_upper[crust_end_old:]
    e_ceft_lower = e_ceft_lower[crust_end_old:]

    n_ceft = n_ceft[crust_end_old:]
    e_ceft = e_ceft[crust_end_old:]
    p_ceft = p_ceft[crust_end_old:]

    # chemical potential
    e_ini = e_ceft[0] 
    p_ini = p_ceft[0] 
    n_ini = n_ceft[0]
    mu_ini = (e_ini + p_ini) / n_ini # MeV

    # sound speed
    cs2_ceft_lower = cs2_lower[crust_end_old:]
    cs2_ceft_lower = smooth_cs2(n_ceft, cs2_ceft_lower, 6,34,101,134)
    cs2_ceft_upper = cs2_upper[crust_end_old:] 
    cs2_ceft_upper = smooth_cs2(n_ceft, cs2_ceft_upper, 35,85,95,140)

    cs2_ceft_avg = (cs2_ceft_upper+cs2_ceft_lower)/2

    # phi
    phi_ceft_lower = get_phi(cs2_ceft_lower)
    phi_ceft_upper = get_phi(cs2_ceft_upper)
    phi_ceft_width = phi_ceft_upper-phi_ceft_lower
    phi_ceft_sigma = CI_to_sigma(phi_ceft_width, 75)

    return n_ceft/ns, cs2_ceft_avg, phi_ceft_sigma, e_ini, p_ini, mu_ini, n_crust/ns, e_crust, p_crust, cs2_crust

"""     return (
        n_ceft[::12] / ns,
        cs2_ceft_avg[::12],
        phi_ceft_sigma[::12],
        e_ini,
        p_ini,
        mu_ini,
        n_crust/ns,
        e_crust,
        p_crust,
        cs2_crust
    ) """

