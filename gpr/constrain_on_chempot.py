"""
Module for generating GPR samples with constraints on the chemical potential integral of cs2/n,
as well as simple bounded constraints on cs2 between 0 and 1 (despite the name of the file).
Contains replacement functions for sampling.generate_sample() to handle constraints and integral observations.
"""

#############################
import numpy as np

import sampling as sam
import anal_helpers as anal
import prepare_ceft as pc
import prepare_pqcd as pp
from kernels import Kernel
from constrainedgp import CGP
from constants import ns
#############################



def C_trapezoid(Xc, n0, n_mu):
    """
    Construct the trapezoidal integration matrix C for integrating cs2/n from n0 to n_mu points.

    Parameters
    ----------
    Xc: sorted (N_c,) grid for cs2
    n0: scalar lower limit
    n_mu: (M,) points where chemical potential mu is known

    Returns 
    -------
    C: np.ndarray
        (M, N_c) such that (C @ y_c)[i] ≈ ∫_{n0}^{n_mu[i]} cs2(n)/n dn
    """
    X = np.asarray(Xc).ravel()
    M = len(n_mu)
    Nc = len(X)
    C = np.zeros((M, Nc))

    if not (X[0] <= n0 <= X[-1]):
        raise ValueError("n0 must lie within the constraint grid Xc range.")

    # indices to start integration
    j0 = np.searchsorted(X, n0) - 1
    j0 = np.clip(j0, 0, Nc - 2)

    dx = np.diff(X)
    invX = 1.0 / X

    for i, n_target in enumerate(n_mu):
        if n_target < n0:
            continue  # integral is zero if target before n0

        k = np.searchsorted(X, n_target) - 1
        k = np.clip(k, 0, Nc - 2)

        # full segments from max(j0,0) .. k-1
        j_start = j0

        # partial first cell [n0, X[j0+1]]
        if n0 > X[j0]:
            L = X[j0 + 1] - n0
            C[i, j0] += 0.5 * L * invX[j0]
            C[i, j0 + 1] += 0.5 * L * (1.0 / X[j0 + 1])
            j_start = j0 + 1

        # full cells
        for j in range(j_start, k):
            L = dx[j]
            C[i, j] += 0.5 * L * invX[j]
            C[i, j + 1] += 0.5 * L * (1.0 / X[j + 1])

        # partial last cell [X[k], n_target]
        if n_target > X[k]:
            L = n_target - X[k]
            C[i, k] += 0.5 * L * invX[k]
            C[i, k + 1] += 0.5 * L * (1.0 / X[k + 1])

    return C


def custom_build_w_vector(
    cgp: CGP, n_int: np.ndarray, C_int: np.ndarray, int_train: np.ndarray, mean_i:np.ndarray | None = None
):
    """
    As we are using integral observations as additional data, the kernels and prior means must be changed.
    CGP class (or GP) is not built to handle this (I am too lazy to modularise this).
    So, the easiest way is to just modify the kernels and stuff from outside

    Parameters
    ----------
    cgp: CGP
        The constrained GP object
    n_int: (N_int,) points where integral observations are made
    C_int: (N_int, N_c) trapezoidal integration matrix for integral observations
    int_train: (N_int,) integral observations to be added to training data
    mean_i: (N_int,) prior mean at integral observation points (optional)
    """

    if mean_i is None:
        (mean_train, _) = cgp._set_means(cgp.x_train, n_int)
    else:
        (mean_train, mean_i) = cgp._set_means(cgp.x_train, n_int)

    mean_train = np.ones_like(cgp.x_train) * mean_train
    mean_i = np.ones_like(n_int) * mean_i  # perhaps should be zero instead

    # here:
    # s signifies the cs2 training points
    # i signifies the integral constraint points
    # 1 signifies all observed points (training + integral constraints)
    # 2 signifies test points
    # c signifies constraining points

    K_ss = cgp._set_K_11(
        cgp.x_train, var_f=cgp.var_f, stabilise=True, jitter_value=cgp.jitter
    ) 
    K_1i = cgp.kernel.compute(cgp.x_train, n_int)
    K_ii_prior = cgp.kernel.compute(n_int)

    K_si = K_1i
    K_is = K_si.T
    K_ii = K_ii_prior

    K_11 = np.block([[K_ss, K_si @ C_int.T], 
                     [C_int @ K_is, C_int @ K_ii @ C_int.T]])
    K_11 += np.eye(K_11.shape[0]) * cgp.jitter

    K_22 = cgp.kernel.compute(cgp.x_test)

    # join training set
    y_joint = np.concatenate([cgp.y_train, int_train])
    cgp.y_train = y_joint

    # joint prior mean
    mean_train = np.concatenate([mean_train, C_int @ mean_i])

    # cross-cov between test f* and joint observations
    K_2s = cgp.kernel.compute(cgp.x_test, cgp.x_train)  # K(X_2, X_s)
    K_2i = cgp.kernel.compute(cgp.x_test, n_int)  # K(X_2, X_c)
    K_21 = np.hstack([K_2s, K_2i @ C_int.T])  # shape (n_test, n_obs)

    # kernels involving constraining points
    K_sc = cgp.kernel.compute(cgp.x_train, cgp.x_constr)
    K_ic = C_int @ (cgp.kernel.compute(n_int, cgp.x_constr))
    K_1c = np.vstack([K_sc, K_ic])
    K_cc = cgp.kernel.compute(cgp.x_constr, cgp.x_constr)
    K_2c = cgp.kernel.compute(cgp.x_test, cgp.x_constr)

    cgp.build_w_vector(
        K_11=K_11,
        K_1c=K_1c,
        K_cc=K_cc,
        K_12=K_21.T,
        K_2c=K_2c,
        K_22=K_22,
        mean_train=mean_train,
    )


def generate_constrained_sample(
    n_ceft,
    cs2_ceft_avg,
    cs2_l_ceft,
    cs2_u_ceft,
    cs2_crust,
    mu_ini,
    x_test_end=10,
    mu_low=2.2,
    mu_high=2.8,
    point_nums=200,
    ceft_end=0,
    kern="SE",
    burn_in=100
):
    """
    Generate a constrained sample of the speed of sound squared (0<=cs2<=1)  as a function of baryon density.
    Conditioned on log chemical potential for 2.6 GeV<mu<mu_high as well.


    Parameters
    ----------
    n_ceft: (N_c,) baryon density points from CEFT | in nsat (if used with make_conditioning_eos(), this is automatically the case)
    cs2_ceft_avg: (N_c,) average cs2 values from CEFT
    cs2_l_ceft: (N_c,) lower bound cs2 values from CEFT
    cs2_u_ceft: (N_c,) upper bound cs2 values from CEFT
    cs2_crust: (N_crust,) crust cs2 values
    mu_ini: float, chemical potential at n_ini | in MeV
    x_test_end: float, maximum baryon density for test points | in nsat
    mu_low: float, lower limit of chemical potential | in GeV
    mu_high: float, upper limit of chemical potential | in GeV
    point_nums: int, number of test points
    ceft_end: float, baryon density where CEFT ends | in nsat (set to 0 to treat as hyperparameter)
    kern: str, kernel type
    burn_in: int, number of burn-in samples for TMVN sampler

    Returns
    -------
    cs2_test: (point_nums,) sampled speed of sound squared values at test points
    x_test: (point_nums,) baryon density test points | in nsat
    X_hat: float, renormalisation scale used in pQCD | in GeV
    n_ceft_end_hat: float, baryon density where CEFT ends | in nsat
    l_hat: float, lengthscale hyperparameter
    """

    (cs2_hat, X_hat, nu_hat, l_hat, alpha_hat) = sam.get_hype_samples(kern)

    l_hat = l_hat*ns  # fm^-3

    match kern:
        case "SE":
            kernel = Kernel("SE", sigma=nu_hat**0.5, l=l_hat)
        case "RQ":
            kernel = Kernel("RQ", sigma=nu_hat**0.5, l=l_hat, alpha=alpha_hat)
        case "M32":
            kernel = Kernel("M32", sigma=nu_hat**0.5, l=l_hat)
        case "M52":
            kernel = Kernel("M52", sigma=nu_hat**0.5, l=l_hat)
        case "GE":
            kernel = Kernel("GE", sigma=nu_hat**0.5, l=l_hat, gamma=alpha_hat)
        case _:
            raise ValueError("Invalid kernel value")

    if ceft_end == 0:
        n_ceft_end_hat = sam.get_hype_n_ceft_end()
    else:
        n_ceft_end_hat = ceft_end
    # n_ceft_end a hyperparameter
    idx = np.searchsorted(n_ceft, n_ceft_end_hat)
    idx_or_before = np.argmin(
        [np.abs(n_ceft[idx - 1] - n_ceft_end_hat), np.abs(n_ceft[idx] - n_ceft_end_hat)]
    )
    if idx_or_before == 1:
        idx = idx + 1

    n_ceft = n_ceft[:idx]

    cs2_ceft_avg = cs2_ceft_avg[:idx]

    cs2_ceft_sigma = pc.CI_to_sigma(cs2_u_ceft - cs2_l_ceft, 95)
    cs2_ceft_sigma = cs2_ceft_sigma[:idx]

    (n_pqcd, cs2_pqcd) = pp.get_pqcd(X_hat, mu_low, mu_high, size=100)  # nsat, unitless
    
    int_size = 100
    lg_mu_pqcd = pp.get_chempot_train(mu_ini, mu_low=2.6, mu_high=mu_high, size=int_size)
    (n_int_obs, _) = pp.get_pqcd(X_hat, mu_low=2.6, mu_high=mu_high, size=int_size)
    n_int_obs = n_int_obs*ns  # fm^-3
    
    x_train = np.concatenate((n_ceft, n_pqcd))*ns  # fm^-3
    cs2_train = np.concatenate((cs2_ceft_avg, cs2_pqcd))
    cs2_pqcd_sigma = np.zeros_like(cs2_pqcd)

    cs2_sigma_train = np.concatenate((cs2_ceft_sigma, cs2_pqcd_sigma))

    train_noise = cs2_sigma_train**2

    x_test = np.linspace(n_ceft[0], x_test_end, point_nums) * ns  # fm^-3
    cgp = CGP(
        kernel=kernel,
        x_train=x_train,
        y_train=cs2_train,
        x_test=x_test,
        var_f=train_noise,
        prior_mean=cs2_hat,
        jitter=1e-8,
    )

    n_int = np.linspace(n_ceft[0], n_pqcd[-1], 400)*ns

    C = C_trapezoid(n_int, n_ceft[0]*ns, n_int_obs)
    
    x_c_gap = np.linspace(n_ceft[-1], n_pqcd[0], 200)*ns # the constraints must span throughout
    x_c_pqcd = np.linspace(n_pqcd[0], n_pqcd[-1], 30)*ns
    x_c = np.concatenate((x_c_gap, x_c_pqcd))
    
    a_c = np.zeros_like(x_c)
    b_c = np.ones_like(x_c)

    x_c_ceft = n_ceft[::10]*ns # in CEFT, the bounds are defined by the CI instead of 0,1
    a_c_ceft = cs2_l_ceft[::10]
    b_c_ceft = cs2_u_ceft[::10]

    x_c = np.concatenate((x_c_ceft, x_c))
    a_c = np.concatenate((a_c_ceft, a_c))
    b_c = np.concatenate((b_c_ceft, b_c))


    d = len(x_c)
    A_bounds = np.vstack([np.eye(d), -np.eye(d)])
    b_bounds = np.concatenate([-a_c, b_c])
    cgp.set_constraints(x_c, A_bounds, b_bounds)

    custom_build_w_vector(cgp, n_int, C, lg_mu_pqcd, mean_i=np.mean(cs2_pqcd))
    
    causality = False  # very rarely the noise from S* makes cs2 be below 0 or above 1, so just get another sample
    attempt = 0
    while not causality:
        cs2_test = cgp.posterior(n_samples=100, eta_init=200.0, update_eta=True, burn_in=burn_in)
        attempt += 1

        # check which samples are fully within [0,1]
        valid_mask = np.all((cs2_test >= 0) & (cs2_test <= 1), axis=1)
        valid_idxs = np.where(valid_mask)[0]

        if valid_idxs.size > 0:
            cs2_test = cs2_test[valid_idxs[0]] # pick the first valid sample
            causality = True
        else:
            attempt += 1
            if attempt > 10:
                raise RuntimeError("Failed to generate a valid constrained sample after 10 attempts.")
        

    cs2_test = cs2_test.flatten()
    cs2_test[0] = cs2_crust[
        -1
    ]  # minor difference in gpr result and actual crust ending so replacing gpr val

    x_test = x_test / ns  # back to nsat

    return cs2_test, x_test, X_hat, n_ceft_end_hat, l_hat


def generate_bounded_sample(
    n_ceft,
    cs2_ceft_avg,
    cs2_l_ceft,
    cs2_u_ceft,
    cs2_crust,
    x_test_end=10,
    mu_low=2.2,
    mu_high=2.8,
    point_nums=200,
    ceft_end=0,
    kern="SE",
    burn_in=100
):
    """
    Generate a constrained sample of the speed of sound squared (0<=cs2<=1)  as a function of baryon density.

    Parameters
    ----------
    n_ceft: (N_c,) baryon density points from CEFT | in nsat (if used with make_conditioning_eos(), this is automatically the case)
    cs2_ceft_avg: (N_c,) average cs2 values from CEFT
    cs2_l_ceft: (N_c,) lower bound cs2 values from CEFT
    cs2_u_ceft: (N_c,) upper bound cs2 values from CEFT
    cs2_crust: (N_crust,) crust cs2 values
    x_test_end: float, maximum baryon density for test points | in nsat
    mu_low: float, lower limit of chemical potential | in GeV
    mu_high: float, upper limit of chemical potential | in GeV
    point_nums: int, number of test points
    ceft_end: float, baryon density where CEFT ends | in nsat (set to 0 to treat as hyperparameter)
    kern: str, kernel type
    burn_in: int, number of burn-in samples for TMVN sampler

    Returns
    -------
    cs2_test: (point_nums,) sampled speed of sound squared values at test points
    x_test: (point_nums,) baryon density test points | in nsat
    X_hat: float, renormalisation scale used in pQCD | in GeV
    n_ceft_end_hat: float, baryon density where CEFT ends | in nsat
    l_hat: float, lengthscale hyperparameter
    """


    (cs2_hat, X_hat, nu_hat, l_hat, alpha_hat) = sam.get_hype_samples(kern)

    l_hat = l_hat  # nsat

    match kern:
        case "SE":
            kernel = Kernel("SE", sigma=nu_hat**0.5, l=l_hat)
        case "RQ":
            kernel = Kernel("RQ", sigma=nu_hat**0.5, l=l_hat, alpha=alpha_hat)
        case "M32":
            kernel = Kernel("M32", sigma=nu_hat**0.5, l=l_hat)
        case "M52":
            kernel = Kernel("M52", sigma=nu_hat**0.5, l=l_hat)
        case "GE":
            kernel = Kernel("GE", sigma=nu_hat**0.5, l=l_hat, gamma=alpha_hat)
        case _:
            raise ValueError("Invalid kernel value")

    if ceft_end == 0:
        n_ceft_end_hat = sam.get_hype_n_ceft_end()
    else:
        n_ceft_end_hat = ceft_end

    # n_ceft_end a hyperparameter
    idx = np.searchsorted(n_ceft, n_ceft_end_hat)
    idx_or_before = np.argmin(
        [np.abs(n_ceft[idx - 1] - n_ceft_end_hat), np.abs(n_ceft[idx] - n_ceft_end_hat)]
    )
    if idx_or_before == 1:
        idx = idx + 1
    n_ceft = n_ceft[:idx]
    cs2_ceft_avg = cs2_ceft_avg[:idx]

    (_, _, cs2_l_ceft, cs2_u_ceft) = anal.get_ceft_cs2()
    cs2_ceft_sigma = pc.CI_to_sigma(cs2_u_ceft - cs2_l_ceft, 95)
    cs2_ceft_sigma = cs2_ceft_sigma[:idx]

    (n_pqcd, cs2_pqcd) = pp.get_pqcd(X_hat, mu_low, mu_high, size=100)  # nsat, unitless

    x_train = np.concatenate((n_ceft, n_pqcd))  # nsat

    cs2_train = np.concatenate((cs2_ceft_avg, cs2_pqcd))
    cs2_pqcd_sigma = np.zeros_like(cs2_pqcd)

    cs2_sigma_train = np.concatenate((cs2_ceft_sigma, cs2_pqcd_sigma))

    train_noise = cs2_sigma_train**2

    x_test = np.linspace(n_ceft[0], x_test_end, point_nums)  # nsat

    cgp = CGP(
        kernel=kernel,
        x_train=x_train,
        y_train=cs2_train,
        x_test=x_test,
        var_f=train_noise,
        prior_mean=cs2_hat,
        jitter=1e-8,
    )

    x_c_gap = np.linspace(n_ceft[-1], n_pqcd[0], 200)*ns # the constraints must span throughout
    x_c_ceft = np.linspace(n_ceft[0], n_ceft[-1], 10)*ns
    x_c_pqcd = np.linspace(n_pqcd[0], n_pqcd[-1], 30)*ns
    x_c = np.concatenate((x_c_ceft, x_c_gap, x_c_pqcd))

    a_c = np.zeros_like(x_c)
    b_c = np.ones_like(x_c)

    d = len(x_c)
    A_bounds = np.vstack([np.eye(d), -np.eye(d)])
    b_bounds = np.concatenate([-a_c, b_c])
    cgp.set_constraints(x_c, A_bounds, b_bounds)

    cgp.build_w_vector()

    causality = False  # very rarely the noise from S* makes cs2 be below 0 or above 1, so just get another sample
    attempt = 0
    while not causality:
        cs2_test = cgp.posterior(n_samples=100, eta_init=200.0, update_eta=True, burn_in=burn_in)
        attempt += 1

        # check which samples are fully within [0,1]
        valid_mask = np.all((cs2_test >= 0) & (cs2_test <= 1), axis=1)
        valid_idxs = np.where(valid_mask)[0]

        if valid_idxs.size > 0:
            cs2_test[valid_idxs[0]] # select the first valid sample
            causality = True
        else:
            attempt += 1
            if attempt > 10:
                raise RuntimeError("Failed to generate a valid constrained sample after 10 attempts.")

    cs2_test = cs2_test.flatten()
    
    # minor difference in gpr result and actual crust ending so replacing gpr val:
    cs2_test[0] = cs2_crust[-1]  
    
    return cs2_test, x_test, X_hat, n_ceft_end_hat, l_hat
