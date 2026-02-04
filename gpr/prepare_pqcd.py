import numpy as np

from pqcd.pQCD import pQCD
from constants import ns


def get_pqcd(X: float, mu_low: float = 2.2, mu_high: float = 3, size: int = 100):
    """
    Retrieve pQCD calculations of number density (in n_sat) and sound speed squared using an X value (renormalisation scale)

    Parameters    
    ----------
    X : float,  
        Renormalisation scale, usually between 0.5 and 2
    mu_low : float, optional
        Lower bound of chemical potential grid in GeV, by default 2.2 GeV
    mu_high : float, optional
        Upper bound of chemical potential grid in GeV, by default 3 GeV
    size : int, optional
        Number of points in the chemical potential grid, by default 100
    
    Returns
    -------
    n_pqcd : np.ndarray
        Number density in units of n_sat
    cs2_pqcd : np.ndarray
        Sound speed squared (unitless)
    """

    mu_grid = np.linspace(mu_low, mu_high, size)

    pQCD_temp = pQCD(X)
    n_pqcd = np.vectorize(pQCD_temp.number_density)(mu_grid) / ns  # in nsat
    cs2_pqcd = np.vectorize(pQCD_temp.speed2)(mu_grid)

    return n_pqcd, cs2_pqcd


def check_pqcd_connection(X_hat, e_end, p_end, n_end):
    """
    Returns boolean signifying is connection to pQCD is possible consistently at a particular renormalisation scale X_hat.

    Parameters
    ----------
    X_hat : float
        Renormalisation scale, usually between 0.5 and 2
    e_end : float
        Energy density at which to connect to pQCD in MeV/fm³
    p_end : float
        Pressure at which to connect to pQCD in MeV/fm³
    n_end : float
        Number density at which to connect to pQCD in n_sat units
    
    Returns
    -------
    boolean : bool
    """
    pqcd_temp = pQCD(X_hat)
    weight = int(pqcd_temp.constraints(e0=e_end / 1000, p0=p_end / 1000, n0=n_end * ns))
    boolean = weight == 1
    return boolean


def get_chempot_train(mu0, mu_low: float = 2.2, mu_high: float = 3, size: int = 100):
    """
    Get log of chemical potential to use as integral conditioning points.

    Parameters
    ----------
    mu0 : float
        Reference chemical potential in MeV
    mu_low : float, optional
        Lower bound of chemical potential grid in GeV, by default 2.2 GeV
    mu_high : float, optional
        Upper bound of chemical potential grid in GeV, by default 3 GeV
    size : int, optional
        Number of points in the chemical potential grid, by default 100
    
    Returns
    -------
    int_train : np.ndarray
        Logarithm of chemical potential grid normalized by reference chemical potential
    """
    mu_grid = np.linspace(mu_low, mu_high, size) * 1000
    int_train = np.log(mu_grid / mu0)

    return int_train
