import numpy as np
from scipy.stats import loguniform

import kernels
import gaussianprocess
import prepare_pqcd as pp
from constants import get_phi


def get_hype_samples(kern: str | None = None):
    """
    Hyperparams for GP mean and kernel, as well as pqcd renormalisation scale
    returns cs2_hat, X_hat, sigma_hat, l_hat, alpha_hat
    """

    rng = np.random.default_rng()

    X = loguniform.rvs(0.5, 2)

    while True:
        cs2_hat = rng.normal(0.5, 0.25)
        if 0 <= cs2_hat <= 1:  # sound speed should be between 0 and 1
            break

    nu = rng.normal(1.25, 0.2)  # note that this is the std, not variance

    alpha = 0.0

    if kern is None:
        kern = "SE"

    match kern:
        case "SE":
            l = rng.uniform(0.5, 1.5)  # default (0.5,1.5)
        case "RQ":
            l = rng.uniform(0.5, 1.5)
            alpha = rng.uniform(0.1, 10)
        case "M32":
            l = rng.uniform(0.580, 1.914)  # default (0.580, 1.914) ; high: (1.914, 2.552) 
        case "M52":
            l = rng.uniform(0.638, 1.742)  # default (0.638,1.742) ; high: (1.742,2.322)
        case "GE":
            l = rng.uniform(0.713, 2.310)  # default (0.713, 2.310) ; high: (2.14, 3.084)
            alpha = rng.uniform(1.6, 1.95)

    return cs2_hat, X, nu, l, alpha


def get_hype_n_ceft_end():  # it was easier to create a new fn than to add it to the one above
    rng = np.random.default_rng()
    nc_hat = rng.uniform(1, 2)
    return nc_hat


def generate_sample(
    n_ceft,
    cs2_ceft_avg,
    phi_ceft_sigma,
    n_crust,
    cs2_crust,
    x_test_end=10,
    mu_low=2.2,
    mu_high=2.8,
    point_nums=200,
    ceft_end=0,
    kern="SE",
):
    """
    input n must be in nsat, if used in conjunction with make_condition_eos() that is automatically the case
    out n in nsat
    """

    (cs2_hat, X_hat, nu_hat, l_hat, alpha_hat) = get_hype_samples(kern)

    match kern:
        case "SE":
            kernel = kernels.Kernel("SE", sigma=nu_hat**0.5, l=l_hat)
        case "RQ":
            kernel = kernels.Kernel("RQ", sigma=nu_hat**0.5, l=l_hat, alpha=alpha_hat)
        case "M32":
            kernel = kernels.Kernel("M32", sigma=nu_hat**0.5, l=l_hat)
        case "M52":
            kernel = kernels.Kernel("M52", sigma=nu_hat**0.5, l=l_hat)
        case "GE":
            kernel = kernels.Kernel("GE", sigma=nu_hat**0.5, l=l_hat, gamma=alpha_hat)
        case _:
            raise ValueError("Invalid kernel value")

    if ceft_end == 0:
        n_ceft_end_hat = get_hype_n_ceft_end()
    else:
        n_ceft_end_hat = ceft_end

    idx = np.searchsorted(n_ceft, n_ceft_end_hat)
    idx_or_before = np.argmin(
        [np.abs(n_ceft[idx - 1] - n_ceft_end_hat), np.abs(n_ceft[idx] - n_ceft_end_hat)]
    )
    if idx_or_before == 1:
        idx = idx + 1
    n_ceft = n_ceft[:idx]
    cs2_ceft_avg = cs2_ceft_avg[:idx]
    phi_ceft_sigma = phi_ceft_sigma[:idx]

    (n_pqcd, cs2_pqcd) = pp.get_pqcd(X_hat, mu_low, mu_high, size=100)  # nsat, unitless

    x_train = np.concatenate((n_ceft, n_pqcd))  # nsat

    cs2_train = np.concatenate((cs2_ceft_avg, cs2_pqcd))
    phi_pqcd_sigma = np.zeros_like(cs2_pqcd)

    phi_sigma_train = np.concatenate((phi_ceft_sigma, phi_pqcd_sigma))

    phi_train = get_phi(cs2_train)
    train_noise = phi_sigma_train**2

    x_test = np.linspace(
        n_ceft[0], x_test_end, point_nums
    )  # number density in nsat, starting val is ending val of n crust, ending val is default 10 nsat

    gp = gaussianprocess.GP(kernel, get_phi(cs2_hat))
    gp.fit(x_train, x_test, phi_train, var_f=train_noise, stabilise=True)

    phi_test = gp.posterior(sampling=True)

    phi_test = phi_test.flatten()
    phi_test[0] = get_phi(
        cs2_crust[-1]
    )  # minor difference in gpr result and actual crust ending so replacing gpr val

    return phi_test, x_test, X_hat, n_ceft_end_hat, l_hat
