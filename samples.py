import numpy as np
import multiprocessing as mp
from tqdm import tqdm

import dickandballs as db
import helpers as hel

def make_eos():
    ceft_lower= np.loadtxt('EOS/ceft/eos_ceft_lower.dat')
    n_ceft_lower, p_ceft_lower, e_ceft_lower = ceft_lower.T

    ceft_upper= np.loadtxt('EOS/ceft/eos_ceft_upper.dat')
    n_ceft_upper, p_ceft_upper, e_ceft_upper = ceft_upper.T

    n_ceft = n_ceft_lower/0.16 #n_sat

    # average CEFT EOS
    e_ceft = (e_ceft_lower+e_ceft_upper)/2
    p_ceft = (p_ceft_lower+p_ceft_upper)/2

    # seperating crust
    n_crust = n_ceft[:111]
    e_crust = e_ceft[:111]
    p_crust = p_ceft[:111]


    p_ceft_upper = p_ceft_upper[111:]
    p_ceft_lower = p_ceft_lower[111:]

    e_ceft_upper = e_ceft_upper[111:]
    e_ceft_lower = e_ceft_lower[111:]

    n_ceft = n_ceft[111:]
    e_ceft = e_ceft[111:]
    p_ceft = p_ceft[111:]

    # chemical potential
    mu_ceft = (e_ceft + p_ceft)/n_ceft

    e_ini = e_ceft[0]
    p_ini = p_ceft[0]
    n_ini = n_ceft[0]
    mu_ini = (e_ini + p_ini) / n_ini

    # sound speed
    cs2_ceft_lower = np.gradient(p_ceft_lower, e_ceft_lower) #dp/de
    cs2_ceft_upper = np.gradient(p_ceft_upper, e_ceft_upper) #dp/de
    cs2_ceft_avg = (cs2_ceft_upper+cs2_ceft_lower)/2
    cs2_ceft_width = cs2_ceft_upper-cs2_ceft_lower
    cs2_ceft_sigma = hel.CI_to_sigma(cs2_ceft_width, 75)

    # phi
    phi_ceft_lower = hel.get_phi(cs2_ceft_lower)
    phi_ceft_upper = hel.get_phi(cs2_ceft_upper)
    phi_ceft_width = phi_ceft_upper-phi_ceft_lower
    phi_ceft_avg = (phi_ceft_upper+phi_ceft_lower)/2
    phi_ceft_sigma = hel.CI_to_sigma(phi_ceft_width, 75)

    return n_ceft, cs2_ceft_avg, phi_ceft_sigma

samples_n = 50000

if __name__ == "__main__":
    n_ceft, cs2_ceft_avg, phi_ceft_sigma = make_eos()

    results_phi = []
    results_n = []
    for _ in tqdm(range(samples_n), desc="Generating samples"):
        phi, n = hel.generate_samples(n_ceft, cs2_ceft_avg, phi_ceft_sigma)
        results_phi.append(phi)
        results_n.append(n)

    # Convert to NumPy array and save
    final_array = np.array([results_n, results_phi])
    np.save("results/phi_samples/samples.npy", final_array)
    print("Saved results to samples.npy")

#TODO: make the script take arguments like the kernel and the num samples to generate
