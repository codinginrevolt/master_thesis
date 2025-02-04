import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import dickandballs as db

def get_hype_samples():
    
    rng = np.random.default_rng()
    cs2_hat = rng.normal(0.5,0.25**2)
    l = rng.normal(1, 0.25**2)
    nu = rng.normal(1.25, 0.2**2)

    return cs2_hat, l, nu

def get_phi(cs2):
    return -np.log(1/cs2 - 1)

def CI_to_sigma(width):
    """From CI 75% to a sigma value
    z*sigma = width/2
    TODO: automate z score so any CI can be taken"""
    z = 1.15
    sig = width/(2*z)
    return(sig)