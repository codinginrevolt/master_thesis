import numpy as np

import sampling as sam
import anal_helpers as anal
import prepare_ceft as pc
import prepare_pqcd as pp
from kernels import Kernel
from constrainedgp import CGP


# despite the name of this file, it also contains bounded 0<=cs2<=1 constraints only too


def C_trapezoid(Xc, n0, n_mu):
    """
    Xc: sorted (N_c,) grid for cs2
    n0: scalar lower limit
    n_mu: (M,) points where chemical potential mu is known

    Returns C: (M, N_c) such that (C @ y_c)[i] ≈ ∫_{n0}^{n_mu[i]} cs2(n')/n' dn'
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
    cgp: CGP, n_int: np.ndarray, C_int: np.ndarray, int_train: np.ndarray
):
    """
    As we are using integral observations as additional data, the kernels and prior means must be changed.
    CGP class (or GP) is not built to handle this (I am too lazy to figure out how i can make this modular).
    So, the easiest way is to just modify the kernels and stuff from outside
    """

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
    )  # training cov + noise if any
    K_1i = cgp.kernel.compute(cgp.x_train, n_int)
    K_ii_prior = cgp.kernel.compute(n_int)

    K_si = K_1i
    K_is = K_si.T
    K_ii = K_ii_prior

    # observation covariance K_11
    K_11 = np.block([[K_ss, K_si @ C_int.T], [C_int @ K_is, C_int @ K_ii @ C_int.T]])
    K_11 += np.eye(K_11.shape[0]) * cgp.jitter
    # test covariance K_22
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
):
    """
    input n must be in nsat, if used in conjunction with make_condition_eos() that is automatically the case
    out n in nsat
    """

    (cs2_hat, X_hat, nu_hat, l_hat, alpha_hat) = sam.get_hype_samples(kern)

    l_hat = l_hat  # fm^-3

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
    lg_mu_pqcd = pp.get_chempot_train(mu_ini, mu_low, mu_high, size=100)

    x_train = np.concatenate((n_ceft, n_pqcd))  # fm^-3

    cs2_train = np.concatenate((cs2_ceft_avg, cs2_pqcd))
    cs2_pqcd_sigma = np.zeros_like(cs2_pqcd)

    cs2_sigma_train = np.concatenate((cs2_ceft_sigma, cs2_pqcd_sigma))

    train_noise = cs2_sigma_train**2

    x_test = np.linspace(n_ceft[0], x_test_end, point_nums)  # fm^-3

    cgp = CGP(
        kernel=kernel,
        x_train=x_train,
        y_train=cs2_train,
        x_test=x_test,
        var_f=train_noise,
        prior_mean=cs2_hat,
        jitter=1e-8,
    )

    n_int = np.linspace(n_ceft[0], n_pqcd[-1], 400)
    C = C_trapezoid(n_int, n_ceft[0], n_pqcd)

    x_c = np.linspace(n_ceft[-1], n_pqcd[0], 100)
    a_c = np.zeros_like(x_c)
    b_c = np.ones_like(x_c)

    d = len(x_c)
    A_bounds = np.vstack([np.eye(d), -np.eye(d)])
    b_bounds = np.concatenate([-a_c, b_c])
    cgp.set_constraints(x_c, A_bounds, b_bounds)

    custom_build_w_vector(cgp, n_int, C, lg_mu_pqcd)

    causality = False  # very rarely the noise from S* makes cs2 be below 0 or above 1, so just get another sample
    tries = 0
    while not causality:
        cs2_test = cgp.posterior(n_samples=1, eta_init=200.0, update_eta=True)
        if np.all(cs2_test >= 0) and np.all(cs2_test <= 1):
            causality = True
        tries += 1
        if tries > 1000:
            print("WARNING: More than 1000 tries, possible hang!")
            break

    cs2_test = cs2_test.flatten()
    cs2_test[0] = cs2_crust[
        -1
    ]  # minor difference in gpr result and actual crust ending so replacing gpr val

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
):
    """
    input n must be in nsat, if used in conjunction with make_condition_eos() that is automatically the case
    out n in nsat
    """

    (cs2_hat, X_hat, nu_hat, l_hat, alpha_hat) = sam.get_hype_samples(kern)

    l_hat = l_hat  # fm^-3

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

    x_train = np.concatenate((n_ceft, n_pqcd))  # fm^-3

    cs2_train = np.concatenate((cs2_ceft_avg, cs2_pqcd))
    cs2_pqcd_sigma = np.zeros_like(cs2_pqcd)

    cs2_sigma_train = np.concatenate((cs2_ceft_sigma, cs2_pqcd_sigma))

    train_noise = cs2_sigma_train**2

    x_test = np.linspace(n_ceft[0], x_test_end, point_nums)  # fm^-3

    cgp = CGP(
        kernel=kernel,
        x_train=x_train,
        y_train=cs2_train,
        x_test=x_test,
        var_f=train_noise,
        prior_mean=cs2_hat,
        jitter=1e-8,
    )

    x_c = np.linspace(n_ceft[-1], n_pqcd[0], 100)
    a_c = np.zeros_like(x_c)
    b_c = np.ones_like(x_c)

    d = len(x_c)
    A_bounds = np.vstack([np.eye(d), -np.eye(d)])
    b_bounds = np.concatenate([-a_c, b_c])
    cgp.set_constraints(x_c, A_bounds, b_bounds)

    cgp.build_w_vector()

    causality = False  # very rarely the noise from S* makes cs2 be below 0 or above 1, so just get another sample
    tries = 0
    while not causality:
        cs2_test = cgp.posterior(n_samples=1, eta_init=200.0, update_eta=True)
        if np.all(cs2_test >= 0) and np.all(cs2_test <= 1):
            causality = True
        if tries > 1000:
            print("WARNING: More than 1000 tries, possible hang!")
            break

    cs2_test = cs2_test.flatten()
    
    # minor difference in gpr result and actual crust ending so replacing gpr val:
    cs2_test[0] = cs2_crust[-1]  
    
    return cs2_test, x_test, X_hat, n_ceft_end_hat, l_hat
