import numpy as np
import cvxpy as cp
import warnings


def softplus(x):
    """
    Approximation of f(x) = log(1 + exp(x)) that is numerically stable for large |x|
    ln(1+e^x) = 0 if x << 0
              = x if x >> 0
    """
    return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))


def log_likelihood_approx(zeta, A, b, eta):
    """
    Approximate log likelihood for linear constraints A zeta + b >= 0
    using a scaled sigmoid function.

    zeta: (d,) current sample
    A: (m, d) constraint matrix
    b: (m,) constraint offset
    eta: approximation strength

    Returns: scalar log likelihood
    """
    logits = A @ zeta + b
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            inside_bracket = np.log1p(np.exp(-eta * logits))
    except RuntimeWarning:
        inside_bracket = softplus(-eta * logits)

    return -np.sum(inside_bracket)


def sample_prior(cov):
    """
    Sample from N(0, cov)
    """
    return np.random.multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov)


def project_onto_constraints(A, b, z):
    """
    Project vector z onto the feasible set defined by A x + b >= 0
    using CVXPY with CLARABEL solver.
    Solves: min_x 0.5 * ||x - z||² subject to A x + b >= 0
    """
    d = z.shape[0]
    x = cp.Variable(d)

    objective = cp.Minimize(0.5 * cp.sum_squares(x - z))
    constraints = [A @ x + b >= 0]
    problem = cp.Problem(objective, constraints)

    problem.solve(solver=cp.CLARABEL, verbose=False)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Projection failed: {problem.status}")

    return x.value


def elliptical_slice_sampling(
    f, cov, A, b, eta, log_likelihood_fn, mu, max_attempts=100000, project_tol=1e-8
):
    nu = sample_prior(cov)

    # slice level evaluated at the absolute current state (z = f + mu)
    log_y = log_likelihood_fn(f + mu, A, b, eta) + np.log(np.random.uniform())

    theta = np.random.uniform(0, 2 * np.pi)
    (theta_min, theta_max) = theta - 2 * np.pi, theta

    for _ in range(max_attempts):
        f_prime = f * np.cos(theta) + nu * np.sin(theta)
        z_prime = f_prime + mu  # <-- ABSOLUTE variable

        ll = log_likelihood_fn(z_prime, A, b, eta)  # <-- evaluate on absolute z

        if ll > log_y:
            violation = A @ z_prime + b
            if np.any(violation < -project_tol):
                # project in absolute space, then recheck slice level
                z_proj = project_onto_constraints(A, b, z_prime)
                ll_proj = log_likelihood_fn(z_proj, A, b, eta)
                if ll_proj > log_y:
                    return z_proj - mu  # return centered state
            else:
                return f_prime  # feasible; return centered state

        # shrink bracket
        if theta < 0:
            theta_min = theta
        else:
            theta_max = theta
        theta = np.random.uniform(theta_min, theta_max)

    raise RuntimeError("ESS failed to find valid sample after max_attempts.")


def sample_tmvn_ess(
    mu, cov, A, b, X, y, n_samples, burn_in=100, eta_init=20.0, update_eta=False
):
    d = mu.shape[0]
    zeta = mu.copy()
    samples = []
    eta = eta_init
    count = 0

    while len(samples) < (n_samples + burn_in):
        zeta_centered = zeta - mu
        zeta_centered = elliptical_slice_sampling(
            f=zeta_centered,
            cov=cov,
            A=A,
            b=b,
            eta=eta,
            mu=mu,
            log_likelihood_fn=lambda z, *_: log_likelihood_approx(z, A, b, eta),
        )
        zeta = zeta_centered + mu
        if np.all(A @ zeta + b >= 0):
            samples.append(zeta.copy())

        if update_eta:
            eta *= 1.01

        if count > n_samples + 1000:
            raise ValueError("Max iterations reached")
        count += 1
    samples = np.array(samples)
    samples = samples[burn_in:]
    return samples
