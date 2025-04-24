import numpy as np

ns = 0.16 # fm^-3 nsat

crust_end = 428

def get_phi(cs2):
    return -np.log(1/cs2 - 1)
