import numpy as np
from scipy.stats import norm
import kernels
import gaussianprocess
from constants import ns, crust_end, get_phi
import prepare_pqcd as pp

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
    n_crust = n_ceft[:crust_end+1]
    e_crust = e_ceft[:crust_end+1]
    p_crust = p_ceft[:crust_end+1]
    cs2_crust = cs2_avg[:crust_end+1]

    # seperating ceft proper
    p_ceft_upper = p_ceft_upper[crust_end:]
    p_ceft_lower = p_ceft_lower[crust_end:]

    e_ceft_upper = e_ceft_upper[crust_end:]
    e_ceft_lower = e_ceft_lower[crust_end:]

    n_ceft = n_ceft[crust_end:]
    e_ceft = e_ceft[crust_end:]
    p_ceft = p_ceft[crust_end:]

    # chemical potential
    e_ini = e_ceft[0] 
    p_ini = p_ceft[0] 
    n_ini = n_ceft[0]
    mu_ini = (e_ini + p_ini) / n_ini # MeV

    # sound speed
    cs2_ceft_lower = cs2_lower[crust_end:]
    cs2_ceft_lower = smooth_cs2(n_ceft, cs2_ceft_lower, 6,34,101,134)
    cs2_ceft_upper = cs2_upper[crust_end:] 
    cs2_ceft_upper = smooth_cs2(n_ceft, cs2_ceft_upper, 35,85,95,140)

    cs2_ceft_avg = (cs2_ceft_upper+cs2_ceft_lower)/2

    # phi
    phi_ceft_lower = get_phi(cs2_ceft_lower)
    phi_ceft_upper = get_phi(cs2_ceft_upper)
    phi_ceft_width = phi_ceft_upper-phi_ceft_lower
    phi_ceft_sigma = CI_to_sigma(phi_ceft_width, 75)

    return n_ceft/ns, cs2_ceft_avg, phi_ceft_sigma, e_ini, p_ini, mu_ini, n_crust/ns, e_crust, p_crust, cs2_crust
