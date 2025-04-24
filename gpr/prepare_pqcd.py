import numpy as np

from pqcd.pQCD import pQCD
from constants import ns

def get_pqcd(X: float, mu_low: float = 2.2, mu_high:float = 2.8, size: int = 100):
    """
    Retrieve pQCD calculations of number density (in n_sat) and sound speed squared using an X value (renormalisation scale)
    Default mu from 2.2 GeV to 3 GeV
    Returns n_pqcd in nsat units, cs2 unitless
    """

    mu_grid = np.linspace(mu_low, mu_high, size)
        
    pQCD_temp = pQCD(X)
    n_pqcd  = np.vectorize(pQCD_temp.number_density)(mu_grid) /ns # in nsat
    cs2_pqcd = np.vectorize(pQCD_temp.speed2)(mu_grid)
    
    return n_pqcd, cs2_pqcd


def check_pqcd_connection(X_hat, e_end, p_end, n_end):
    pqcd_temp = pQCD(X_hat)
    weight = (int(pqcd_temp.constraints(e0=e_end/1000, p0=p_end/1000, n0=n_end*ns)))
    boolean = weight == 1  
    return boolean