import numpy as np

from constants import ns, crust_end, get_phi
from prepare_ceft import smooth_cs2

from pathlib import Path
base_dir = Path(__file__).resolve().parent


def get_ceft_cs2():
    """
    Returns:
    n_ceft in nsat,
    cs2_ceft_avg,
    cs2_ceft_lower,
    cs2_ceft_upper,
    """        
    ceft_lower= np.loadtxt(base_dir / 'EOS/ceft/eos_ceft_lower.dat')
    n_ceft_lower, p_ceft_lower, e_ceft_lower = ceft_lower.T

    ceft_upper= np.loadtxt(base_dir / 'EOS/ceft/eos_ceft_upper.dat')
    _, p_ceft_upper, e_ceft_upper = ceft_upper.T


    n_ceft = n_ceft_lower #fm^-3

    cs2_lower = np.gradient(p_ceft_lower, e_ceft_lower) #dp/de
    cs2_upper = np.gradient(p_ceft_upper, e_ceft_upper)

    cs2_avg = (cs2_upper+cs2_lower)/2

    e_ceft = (e_ceft_lower+e_ceft_upper)/2 # MeVfm^-3 
    p_ceft = (p_ceft_lower+p_ceft_upper)/2 # MeVfm^-3

    # seperating ceft proper
    p_ceft_upper = p_ceft_upper[crust_end:]
    p_ceft_lower = p_ceft_lower[crust_end:]

    e_ceft_upper = e_ceft_upper[crust_end:]
    e_ceft_lower = e_ceft_lower[crust_end:]

    n_ceft = n_ceft[crust_end:]
    e_ceft = e_ceft[crust_end:]
    p_ceft = p_ceft[crust_end:]

    # sound speed
    cs2_ceft_lower = cs2_lower[crust_end:]
    cs2_ceft_lower = smooth_cs2(n_ceft, cs2_ceft_lower, 6,34,101,134)
    cs2_ceft_upper = cs2_upper[crust_end:] 
    cs2_ceft_upper = smooth_cs2(n_ceft, cs2_ceft_upper, 35,85,95,140)
    cs2_ceft_avg = cs2_avg[crust_end:]  

    return n_ceft/ns, cs2_ceft_avg, cs2_ceft_lower, cs2_ceft_upper

def get_ceft_phi():
    """
    Returns:
    n_ceft in nsat,
    phi_ceft_avg,
    phi_ceft_lower,
    phi_ceft_upper,
    """        
    ceft_lower= np.loadtxt(base_dir / 'EOS/ceft/eos_ceft_lower.dat')
    n_ceft_lower, p_ceft_lower, e_ceft_lower = ceft_lower.T

    ceft_upper= np.loadtxt(base_dir / 'EOS/ceft/eos_ceft_upper.dat')
    _, p_ceft_upper, e_ceft_upper = ceft_upper.T


    n_ceft = n_ceft_lower #fm^-3

    cs2_lower = np.gradient(p_ceft_lower, e_ceft_lower) #dp/de
    cs2_upper = np.gradient(p_ceft_upper, e_ceft_upper)
    
    cs2_avg = (cs2_upper+cs2_lower)/2

    e_ceft = (e_ceft_lower+e_ceft_upper)/2 # MeVfm^-3 
    p_ceft = (p_ceft_lower+p_ceft_upper)/2 # MeVfm^-3

    # seperating ceft proper
    p_ceft_upper = p_ceft_upper[crust_end:]
    p_ceft_lower = p_ceft_lower[crust_end:]

    e_ceft_upper = e_ceft_upper[crust_end:]
    e_ceft_lower = e_ceft_lower[crust_end:]

    n_ceft = n_ceft[crust_end:]
    e_ceft = e_ceft[crust_end:]
    p_ceft = p_ceft[crust_end:]

    # sound speed
    cs2_ceft_lower = cs2_lower[crust_end:]
    cs2_ceft_lower = smooth_cs2(n_ceft, cs2_ceft_lower, 6,34,101,134)
    cs2_ceft_upper = cs2_upper[crust_end:] 
    cs2_ceft_upper = smooth_cs2(n_ceft, cs2_ceft_upper, 35,85,95,140) 
    cs2_ceft_avg = cs2_avg[crust_end:]

    # phi
    phi_ceft_lower = get_phi(cs2_ceft_lower)
    phi_ceft_upper = get_phi(cs2_ceft_upper)
    phi_ceft_avg =  get_phi(cs2_ceft_avg)

    return n_ceft/ns, phi_ceft_avg, phi_ceft_lower, phi_ceft_upper


def get_n_test(n_end, numpoints):
    """
    Input: 
    n_end: in nsat, corrresponding to sample's termination point,
    numpoints: number of datapoints,

    Returns:
    n: in nsat, test array including crust number density and gpr test number density
    """

    ceft_lower= np.loadtxt(base_dir / 'EOS/ceft/eos_ceft_lower.dat')
    n_ceft_lower, _, _ = ceft_lower.T


    n_ceft = n_ceft_lower #fm^-3
    n_crust = n_ceft[:crust_end]/ns #nsat

    n_test = np.linspace(n_ceft[crust_end]/ns, n_end, numpoints)

    n = np.concatenate((n_crust, n_test))

    return n



