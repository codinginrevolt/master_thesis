import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from collections.abc import Callable

from gaussianprocess import GP
from kernels import Kernel
import tmg_samplers as tmg


class CGP(GP):
    """
    Constrained Gaussian Process class inheriting from GP
    Based on Da Veiga  & Marcell (2012 & 2017)

    User must define matrix A and vector b representing the linear constraints

    Linear system: a<=Z<=b
    0 < [Z,-Z] + [-a, b]
    User must define and supply A = [Z, -Z] and b = [-a, b], that will encode the constraints
    """

    def __init__(
        self,
        kernel: Kernel,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        var_f: float | np.ndarray | None = None,
        prior_mean: (
            float | np.ndarray | Callable[[np.ndarray], np.ndarray] | None
        ) = None,
        x_constr: np.ndarray = None,
        A=None,
        b=None,
        jitter=1e-10,
    ) -> None:
        super().__init__(kernel, prior_mean)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.var_f = var_f
        self.jitter = jitter
        self.x_constr = x_constr
        self.A = A
        self.b = b
        self.Z_samples = None
        self.mu_Z = None
        self.Sigma_Z = None
        self.Sigma_YZ = None
        self.mean_train = None
        self.mean_test = None
        self.mean_star = None
        self.mean_constr_prior = None
        self.mu_constr = None
        self.cov_constr = None

        self.K_11 = None
        self.K_1c = None
        self.K_cc = None
        self.K_12 = None
        self.K_1c = None
        self.K_2c = None
        self.K_22 = None

    def set_constraints(self, x_constr, A, b):
        """
        Set the constraint matrix A and vector b
        """
        self.x_constr = x_constr
        self.A = A
        self.b = b

    def build_w_vector(
        self,
        K_11=None,
        K_1c=None,
        K_cc=None,
        K_12=None,
        K_2c=None,
        K_22=None,
        mean_train=None,
    ):

        if K_11 is None or K_1c is None or K_cc is None or K_12 is None or K_22 is None:
            (
                self.K_11,
                self.K_1c,
                self.K_cc,
                self.K_12,
                self.K_1c,
                self.K_2c,
                self.K_22,
            ) = self._set_constraining_kernels()

        if K_11 is not None:
            self.K_11 = K_11
        if K_1c is not None:
            self.K_1c = K_1c
        if K_cc is not None:
            self.K_cc = K_cc
        if K_12 is not None:
            self.K_12 = K_12
        if K_1c is not None:
            self.K_1c = K_1c
        if K_2c is not None:
            self.K_2c = K_2c
        if K_22 is not None:
            self.K_22 = K_22

        if mean_train is None:
            (self.mean_train, self.mean_constr_prior) = self._set_means(
                self.x_train, self.x_constr
            )
        else:
            (_, self.mean_constr_prior) = self._set_means(self.x_train, self.x_constr)
            self.mean_train = mean_train

        f_tilde = self.y_train - self.mean_train
        (L, alpha) = self._compute_cholesky(self.K_11, f_tilde)

        self.mu_Z = self.mean_constr_prior + (self.K_1c.T @ alpha)  # E[Y(X_c)]
        v_c = solve_triangular(
            L, self.K_1c, lower=True, check_finite=False
        )  # (L^{-1} k(X_1, X_c)); usual GP eqs
        self.Sigma_Z = self.K_cc - (v_c.T @ v_c)  # Cov[Y(X_c)]

        (_, self.mean_test) = self._set_means(self.x_train, self.x_test)

        # conditional cross-covariance: k(x_2, X_c) - k(X_1, x_2)^T Sigma_s^{-1} k(X_1, X_c)
        # k(X_1, x_2)^T Sigma_s^{-1} k(X_1, X_c) = k(X_1, x_2)^T (L L^T)^{-1} k(X_1, X_c) = (L^{-1} k(X_1, x_2))^T (L^{-1} k(X_1, X_c))
        v = solve_triangular(
            L, self.K_12, lower=True, check_finite=False
        )  # (L^{-1} k(X_1, x_2))^T

        self.Sigma_YZ = self.K_2c - v.T @ v_c  # Cov(Y(x_2), Y(X_c))

        self.cov_star = self.K_22 - (v.T @ v)

        self.mean_star = self.mean_test + (self.K_12.T @ alpha)

    def approximate_moments(self, n_samples=50, eta_init=20.0, update_eta=True):
        # needs some more work, iffy on the centering
        self.Z_samples = tmg.sample_tmvn_ess(
            self.mu_Z,
            self.Sigma_Z,
            self.A,
            self.b,
            X=None,
            y=None,
            n_samples=n_samples,
            eta_init=eta_init,
            update_eta=update_eta,
        )
        nu_Z = np.mean(self.Z_samples, axis=0)
        Gamma_Z = np.cov(self.Z_samples, rowvar=False)

        Lz = cholesky(
            self.Sigma_Z + self.jitter * np.eye(self.Sigma_Z.shape[0]),
            lower=True,
            check_finite=False,
        )
        nu_mu = nu_Z - self.mu_Z
        temp1 = solve_triangular(Lz, nu_mu, lower=True, check_finite=False)
        a = solve_triangular(
            Lz.T, temp1, lower=False, check_finite=False
        )  # Sigma_Z^{-1} (nu_Z - mu_Z)
        self.mu_constr = (
            self.mean_test + self.Sigma_YZ @ a
        )  # eq 15 of Da Veiga & Marrel 2017

        temp2 = solve_triangular(Lz, self.Sigma_YZ.T, lower=True, check_finite=False)
        term1 = self.Sigma_YZ @ solve_triangular(
            Lz.T, temp2, lower=False, check_finite=False
        )  # Sigma_YZ (Sigma_Z^{-1} Sigma_YZ^T)

        Sigma_Z_inv = cho_solve(
            (Lz, True), np.eye(self.Sigma_Z.shape[0]), check_finite=False
        )
        temp3 = Sigma_Z_inv @ Gamma_Z @ Sigma_Z_inv
        term2 = (
            self.Sigma_YZ @ temp3 @ self.Sigma_YZ.T
        )  # Sigma_YZ (Sigma_Z^{-1} Gamma_Z Sigma_Z^{-1} Sigma_YZ^T)

        self.cov_constr = (
            self.cov_star - term1 + term2
        )  # eq 16 of Da Veiga & Marrel 2017

        return self.mu_constr, self.cov_constr

    def posterior(self, n_samples=50, eta_init=20.0, update_eta=True):

        z0 = tmg.project_onto_constraints(self.A, self.b, self.mu_Z)

        self.Z_samples = tmg.sample_tmvn_ess(
            z0,
            self.Sigma_Z,
            self.A,
            self.b,
            X=None,
            y=None,
            n_samples=n_samples,
            eta_init=eta_init,
            update_eta=update_eta,
        )

        Lz = cholesky(
            self.Sigma_Z + self.jitter * np.eye(self.Sigma_Z.shape[0]), lower=True
        )
        inv_Sz = cho_solve((Lz, True), np.eye(self.Sigma_Z.shape[0]))
        S_star = self.cov_star - self.Sigma_YZ @ inv_Sz @ self.Sigma_YZ.T
        Z_centered = self.Z_samples - self.mu_Z
        m_stars = self.mean_star + (self.Sigma_YZ @ inv_Sz @ Z_centered.T).T
        rng = np.random.default_rng()

        n_t = S_star.shape[0]
        n_samples = m_stars.shape[0]

        eps = rng.multivariate_normal(
            mean=np.zeros(n_t),
            cov=S_star,
            size=n_samples,
            check_valid="ignore",
            tol=1e-6,
        )
        Ystar_samples = m_stars + eps

        return Ystar_samples

    def _set_constraining_kernels(self):
        # for covariances, 1 or s signifies  training points, c constraint points, t or * test points
        (K_11, K_1c, K_cc) = self._set_kernels(
            self.x_train,
            self.x_constr,
            var_f=self.var_f,
            stabilise=True,
            jitter_value=self.jitter,
        )

        K_12 = self.kernel.compute(self.x_train, self.x_test)
        K_2c = self.kernel.compute(
            self.x_test, self.x_constr
        )  # cross cov b/w test and constraint points

        K_22 = self.kernel.compute(self.x_test)

        return K_11, K_1c, K_cc, K_12, K_1c, K_2c, K_22
