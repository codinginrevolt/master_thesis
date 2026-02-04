import numpy as np
from constants import get_phi


def append_eos_results(
    results_e,
    results_p,
    results_cs2,
    epsilon,
    pressure,
    cs2,
    e_crust,
    p_crust,
    cs2_crust,
):
    """
    Appends the crust EOS data to the GPR results and stores them in the provided lists.
    """
    e_array = np.concatenate(
        (e_crust[:-1], epsilon)
    )  # last element of crust is first element of gpr result
    p_array = np.concatenate((p_crust[:-1], pressure))
    cs2_array = np.concatenate((cs2_crust[:-1], cs2))

    results_e.append(e_array)
    results_p.append(p_array)
    results_cs2.append(cs2_array)


def append_n_phi(results_n, results_phi, n_array, phi_array, n_crust, cs2_crust):
    """
    Appends the crust number density and phi data to the GPR results and stores them in the provided lists.
    """
    n_array = np.concatenate((n_crust[:-1], n_array))
    phi_array = np.concatenate((get_phi(cs2_crust[:-1]), phi_array))

    results_n.append(n_array)
    results_phi.append(phi_array)
