import numpy as np

ns = 0.16 # fm^-3 nsat

crust_end = 75 # index number, see notebook cheft.ipynb

crust_end_old = 248

def get_phi(cs2):
    return -np.log(1/cs2 - 1)
