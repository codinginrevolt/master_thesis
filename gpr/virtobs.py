import sys
import os

current_dir = os.path.dirname(os.path.abspath(''))
others_path = os.path.join(current_dir, '..', 'gpr')

others_path = os.path.abspath(others_path)
if others_path not in sys.path:
    sys.path.append(others_path)

import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable
import cvxpy as cp

import eos
from kernels import Kernel
import gaussianprocess
import finitedimensionalgp
import sampling as sam
import prepare_ceft as pc
import prepare_pqcd as pp
import anal_helpers as anal
from pqcd.pQCD import pQCD
from constants import get_phi, ns


from pathlib import Path
notebook_dir = Path.cwd()

from scipy.linalg import cholesky, solve_triangular, cho_solve
from scipy.stats import norm
import scipy as sp
import pandas as pd

def norm_cdf_int_approx(mu, std, LB, UB):
    """ 
    Return P(LB < X < UB) for X Normal(mu, std) using approximation of Normal CDF 
    
    Input: All inputs as 1-D arrays
    """
    l = normal_cdf_approx((LB - mu)/std)
    u = normal_cdf_approx((UB - mu)/std)
    return u - l

def normal_cdf_approx(x):
    """ 
    Approximation of standard normal CDF
    
    Input: x = array
    
    Polynomial approximation from Abramowitz and Stegun p. 932
    http://people.math.sfu.ca/~cbm/aands/frameindex.htm
    
    Absolute error < 7.5*10^-8
    """
    p = 0.2316419
    b = [0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429]
    
    xx = abs(x) # Approximation only works for x > 0, return 1 - p otherwise
    
    t = 1/(1 + p*xx)
    Z = (1/(np.sqrt(2*np.pi)))*np.exp(-(x*x)/2)
    pol = b[0]*t + b[1]*(t**2) + b[2]*(t**3) + b[3]*(t**4) + b[4]*(t**5)
    
    prob = 1 - Z*pol # For x > 0
    prob[x < 0] = 1 - prob[x < 0] # Change when x < 0
    
    return prob

class LinearOperator:
    """
    Abstract base for any linear operator acting on a GP.
    You must implement three covariance helpers so the
    back-end never has to know the maths.
    """
    def apply_mean(self, mu: np.ndarray) -> np.ndarray:
        """Return L mu  (vectorised over inputs)"""

    def K_with_f(self, ker: Kernel, X_op: np.ndarray, X_f: np.ndarray) -> np.ndarray:
        """Cov[ L f(X_op), f(X_f) ]  i.e. one side transformed"""

    def K_with_L(self, ker: Kernel,
                 X_op1: np.ndarray,
                 op2: 'LinearOperator',
                 X_op2: np.ndarray) -> np.ndarray:
        """Cov[ L1 f(X_op1), L2 f(X_op2) ]"""

class Constraint:
    def __init__(self,
                 op     : LinearOperator,
                 LB     : Callable[[np.ndarray], np.ndarray],
                 UB     : Callable[[np.ndarray], np.ndarray],
                 bounds : list[tuple[float, float]] | None = None
                ):
        self.op  = op
        self.LB  = LB
        self.UB  = UB
        self.bounds = bounds
        self.Xv  = None

    def add(self, x):
        x = np.atleast_2d(x)
        self.Xv = x if self.Xv is None else np.vstack([self.Xv, x])

    def LBXV(self, x=None):
        """Evaluate lower bound at all virtual points"""
        if self.Xv is None:
            raise ValueError("No virtual points added to constraint.")
        if x is None:
            x = self.Xv
        return self.LB(x)

    def UBXV(self, x=None):
        """Evaluate upper bound at all virtual points"""
        if self.Xv is None:
            raise ValueError("No virtual points added to constraint.")
        if x is None:
            x = self.Xv
        return self.UB(x)
    
class VOGP:
    """
    Small subset of Agrell 2019
    """

    def __init__(self, 
                kernel: Kernel,
                x_train : np.ndarray,
                y_train: np.ndarray,
                x_test: np.ndarray,
                var_y: float|np.ndarray|None = None, 
                prior_mean: float|np.ndarray|None = None,
                constraints: list[Constraint] = None,
                var_constr: float = 1e-6,
                stabilise: bool = False, 
                jitter_value: float = 1e-10,
                sampler=None):
        
        self.kern = kernel
        self.X = x_train.reshape(-1,1)
        self.Y = y_train.reshape(-1,1)
        self.Xs = x_test.reshape(-1,1)
        self.var_y = var_y if var_y is not None else 0.0
        self.var_constr = var_constr # noise in constraint (helps w stabilisation)

        if prior_mean is not None: 
            self.prior_mean = prior_mean
        else:
            self.prior_mean = 0

        self.stabilise = stabilise
        self.jitter = jitter_value
        self.constraints = constraints
        self.sampler = sampler
        self.fitted = False

        # caches (set after fit)
        self.cached = False
        self.L = None            # chol of K(X,X)+σ²I
        self.alpha = None          # (K+σ²I)⁻¹ (Y-mu)
        self.K_x_x = None
        self.K_x_xs = None
        self.K_xs_xs = None
        self.mean_train = None
        self.mean_test = None

        self.mean_star = None
        self.cov_star = None

    def fit(self) -> None:
            """
            fits the GP with the training data
            input of training data x1 and f with noise var_f, test data x2, whether to stabilise with jitter and jitter value
            variance in training data can be supplied using var_f, leave empty if noise free data

            uses algorithm 2.1 from gp book
            returns the mean and covariance metric
            """
            self._prepare()
            f_tilde = self.f_tilde
            K_11 = self.K_x_x
            K_12 = self.K_x_xs
            K_22 = self.K_xs_xs
            mean_test = self.mean_test
            L = self.L
            alpha = self.alpha

            self.mean_star = mean_test + (K_12.T @ alpha) # 2.1 Line 4

            v = solve_triangular(L, K_12, lower=True, check_finite=False) # 2.1 Line 5
            self.cov_star = K_22 - (v.T @ v) # 2.1 Line 6

            self.fitted = True
    
    def fit_constrained(self, n_samples=100, eta=20, burn_in=100, update_eta=True):

        if self.fitted == False:
            self.fit()

        L = self.L
        K_x_xs = self.K_x_xs
        K_xs_xs = self.K_xs_xs
        mean_test = self.mean_test
        f_tilde = self.f_tilde



        L2T_K_x_xv = self._calc_L2T(self.X_training)
        L1L2T_K_xv_xv = self._calc_L1L2()

        v1 = cho_solve((L, True), L2T_K_x_xv, check_finite=False)
        A1 = cho_solve((L.T, False), v1).T

        n = L1L2T_K_xv_xv.shape[0]
        B1 = L1L2T_K_xv_xv + (self.var_constr*np.identity(n)) - v1.T@v1


        v2 = cho_solve((L,True), K_x_xs)
        B2 = K_xs_xs - v2.T@v2
        A2 = cho_solve((L.T,False), v2, check_finite=False).T

        L2T_K_xs_xv = self._calc_L2T(self.Xs)
        L1 = self._compute_cholesky(B1, f=None, only_L=True)
        B3 = L2T_K_xs_xv - v2.T@v1
        v3 = cho_solve((L1,True), B3.T)
        
        A = cho_solve((L1.T, False), v3).T
        B = A2 - A@A1
        Sigma = (B2 - v3.T@v3) + (self.jitter * np.eye(B2.shape[0]))

        Lmu, constr_mean = self._calc_constr_mean()

        LB, UB = self._calc_constr_bounds()

        C_sim = self.sample_posterior(Lmu, constr_mean, LB, UB, B1, n_samples, eta=20, burn_in=100, update_eta=True)

        dim = self.Xs.shape[0]
        rng = np.random.default_rng()
        U_sim = rng.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=n_samples).T

        Q = self._compute_cholesky(Sigma, f=None, only_L=True)

        fs_sim = mean_test + (B@f_tilde) + (A@(C_sim - Lmu)) + Q@U_sim
        
        return fs_sim

    def sample_posterior(self, Lmu, constr_mean, LB, UB, B1, n_samples, burn_in=100, eta=20, update_eta=True):
        """
        Samples from the posterior distribution over the constraint GP C~|C,Y
        using elliptical slice sampling.
        """

        n = Lmu.shape[0]
        M = np.vstack([np.eye(n), -np.eye(n)])
        g = np.hstack([-LB, UB])

        samples = sample_tmvn_ess(
            mu = constr_mean.flatten(),
            cov = B1,
            A = M,
            b = g,
            X = self.X_training,
            y = self.Y,
            n_samples = n_samples,
            burn_in=burn_in,
            eta_init = eta,
            update_eta = update_eta
        )

        return samples

        return samples[burn_in:]

    def find_XV_subop(self, p_target, Omega=None, bounds=None, i_range=None,
                    nu=None, max_iterations=200, num_samples=1000,
                    batch_size=512, sampling_alg='ess'):

        """
        Find each constraint.Xv so that min_x P_c(x) >= p_target.

        Omega = Finite set of candidate points. If Omega = None then
        global optimization is performed in the region defined by 'bounds'
        
        bounds = bounds on input space
        p_target = target constraint probability
        """

        if (Omega is None) ^ (bounds is None):
            raise ValueError("Either supply Omega (finite set of points) or bounds (on input space)")

        # Determine which sub-operators to include
        if i_range is None:
            i_range = list(range(len(self.constraints)))

        # Set nu for widened constraint margins
        if nu is None:
            nu = max(self.constr_likelihood * sp.stats.norm.ppf(p_target), 0)

        # Reset all previous virtual points
        self.reset_XV()

        rows = []
        i_add_pts = 0

        for j in range(max_iterations):

            pc_list, x_list = [], []

            # Loop over each sub-operator and find the most violated point
            for i in i_range:
                if Omega is None:
                    # Global optimization
                    success, x_min, pc_min = self._argmin_pc_subop(
                        i, nu, bounds, sampling_alg=sampling_alg,
                        num_samples=num_samples
                    )
                else:
                    # Finite candidate set
                    pc_min, x_min = self._argmin_pc_subop_finite(
                        i, nu, Omega, batch_size=batch_size,
                        sampling_alg=sampling_alg, num_samples=num_samples,
                    )
                    success = True

                if not success: return None
                pc_list.append(pc_min)
                x_list.append(x_min)

            # Choose the operator with the worst violation (lowest p_c)
            pc_min = min(pc_list)
            i_min = pc_list.index(pc_min)
            x_min = x_list[i_min]

            # Store current progress
            rows.append([j, i_min] + list(x_min) + pc_list)

            # Check if all constraints are now satisfied
            if pc_min >= p_target:
                break

            # Add new virtual point to the appropriate constraint
            i_add_pts += 1
            self.constraints[i_min].add(x_min)

            # Invalidate any cached matrices dependent on Xv
            self.reset_XV() # need to make this

        # Format and return result table
        df_out = pd.DataFrame(rows)
        df_out.columns = ['num_Xv', 'update_constr'] + [f'Xv[{i+1}]' for i in range(len(x_min))] + [f'pc_{i+1}' for i in i_range]
        return df_out, i_add_pts, pc_min


    # utils
    def _prepare(self):
        self.cached = True
        x1 = self.X
        x2 = self.Xs
        f = self.Y
        var_f = self.var_y
        stabilise = self.stabilise
        jitter_value = self.jitter 
        mean_train, mean_test = self._set_means(x1, x2)
        K_11, K_12, K_22 = self._set_kernels(x1, x2, var_f, stabilise, jitter_value)

        L, alpha = self._compute_cholesky(K_11, f_tilde)

        f_tilde = f - mean_train
        
        self.f_tilde = f_tilde
        self.K_x_x = K_11
        self.K_x_xs = K_12
        self.K_xs_xs = K_22
        self.mean_train = mean_train
        self.mean_test = mean_test
        self.L = L
        self.alpha = alpha

    def _set_kernels(self, x1: np.ndarray, x2: np.ndarray, var_f: float|np.ndarray|None = None, stabilise: bool = False, jitter_value: float = 1e-10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        K_11 = self._set_K_11(x1, var_f, stabilise, jitter_value)
        K_12 = self.kernel.compute(x1, x2)
        K_22 = self.kernel.compute(x2)


        return K_11, K_12, K_22
    
    def _set_K_11(self, x1: np.ndarray, var_f: None|np.ndarray|float= None, stabilise: bool = False, jitter_value: float = 1e-10) -> np.ndarray:

        K_11 = self.kernel.compute(x1)

        if var_f is not None:
            var_f = np.atleast_1d(var_f)
            K_11 += (np.eye(len(var_f))*(var_f**2))

        if stabilise:
            # add jitter to the diagonals, helps with numerical stability
            K_11[np.diag_indices_from(K_11)] += jitter_value

        return K_11

    
    def _compute_cholesky(self, K_11: np.ndarray, f: np.ndarray|None, only_L=False) -> np.ndarray|tuple:
        try:
            # perform cholesky decomposition
            L = cholesky(K_11, lower=True, check_finite=False) # algo 2.1: Line 2
        except Exception as e:
            raise ValueError("The matrix K_11 probably is not semi-positive definite. Using a jitter value (set stabilise=True), or increasing it (default jitter_value=1e-10) can help with numerical stability.") from e
        if only_L:
            return L
        # algo 2.1 Line 3
        alpha = cho_solve((L,True), f, check_finite=False) # cho_solve does A\x (where A = LL.T) as opposed to using solve_triangular to find (L\x) and then (L.T \ (L\x)). L\x is Lx=F

        return L, alpha
    
    def _calc_L2T(self, XX: np.ndarray) -> np.ndarray:
        """
        Returns matrix: stacked [Cov[f(XX), L_i f(Xv_i)]] for all constraints i 
        where XX = X or XX = Xs
        """
        blocks = []

        for constraint in self.constraints:  # assumes self.constraints is a list of Constraint objects
            if constraint.Xv is not None:
                K_block = constraint.op.K_with_f(self.kernel, constraint.Xv, XX).T
                blocks.append(K_block)

        return np.block(blocks) if blocks else np.zeros((0, XX.shape[0]))
    
    def _calc_L1L2(self) -> np.ndarray:
        """
        Generalized version: compute the constraint covariance matrix:
        Cov[ L_i f(Xv_i), L_j f(Xv_j) ] for all constraint pairs (i, j)
        Uses each constraint's LinearOperator instance to do the math.
        """
        constraint_blocks = []
        row_offset = 0  # For alignment in asymmetric block filling

        for i, ci in enumerate(self.constraints):
            if ci.Xv is None:
                continue
            row_blocks = []
            col_offset = 0

            for j, cj in enumerate(self.constraints):
                if cj.Xv is None:
                    continue

                # Compute only upper-triangle (i ≤ j)
                if j < i:
                    # Leave blank; will fill in by symmetry later
                    row_blocks.append(None)
                    col_offset += cj.Xv.shape[0]
                    continue

                # Call operator's K_with_L to compute Cov[L_i f, L_j f]
                K_block = ci.op.K_with_L(self.kernel, ci.Xv, cj.op, cj.Xv)
                row_blocks.append(K_block)

                col_offset += cj.Xv.shape[0]

            constraint_blocks.append(row_blocks)
            row_offset += ci.Xv.shape[0]

        # Fill symmetric matrix from blocks
        full_rows = []
        for i, row in enumerate(constraint_blocks):
            full_row = []
            for j, block in enumerate(row):
                if block is not None:
                    full_row.append(block)
                else:
                    # Mirror from upper triangle
                    full_row.append(constraint_blocks[j][i].T)
            full_rows.append(np.hstack(full_row))

        return np.vstack(full_rows)

    def _calc_constr_mean(self):
        """
        Compute:
        Lmu = stacked [L_i mu(Xv_i)]
        constr_mean = Lmu + A1 @ (Y - mu(X))
        """
        # Stack Lmu from all constraints
        Lmu_blocks = []
        for constraint in self.constraints:
            if constraint.Xv is not None:
                # Apply the operator's mean transformation
                mu_v = self._mean_function(constraint.Xv)  # mu(Xv)
                Lmu_i = constraint.op.apply_mean(mu_v)    # L_i mu(Xv)
                Lmu_blocks.append(Lmu_i)

        if Lmu_blocks:
            Lmu = np.concatenate(Lmu_blocks, axis=0).reshape(-1, 1)
        else:
            Lmu = np.zeros((0, 1))  # No constraints case

        constr_mean = Lmu + self.A1 @ self.f_tilde

        return Lmu, constr_mean

    def _mean_function(self, X):
        if callable(self.prior_mean):
            return self.prior_mean(X)
        else:
            return np.full((X.shape[0], 1), self.prior_mean)

    def _calc_constr_bounds(self):
        LB, UB = [], []

        for constraint in self.constraints:
            if constraint.Xv is not None:
                LB.append(constraint.LBXV())
                UB.append(constraint.UBXV())

        return np.concatenate(LB), np.concatenate(UB)
    
    def _argmin_pc_subop(self, i, nu, bounds,
                        sampling_alg='ess',
                        num_samples=1000):

        min_prob_log = 1E-10

        if not self.cached:
            self._prepare()

        if not self._has_virtual_obs():
            args = (i, nu)
            def optfun(x, *args):
                i, nu = args
                p_c = self._constrprob_xs_1(np.array(x).reshape(1, -1), i, nu)[0]
                return np.log(max(p_c, min_prob_log))
        else:
            args = (i, nu, num_samples, sampling_alg)
            def optfun(x, *args):
                i, nu, n, alg = args
                p_c = self._constrprob_xs_2(np.array(x).reshape(1, -1), i, nu, n, alg, False)[0]
                return np.log(max(p_c, min_prob_log))

        res = sp.optimize.differential_evolution(optfun, bounds=bounds, args=args)
        return res.success, res.x, np.exp(res.fun)

    def _has_virtual_obs(self) -> bool:
        """Check if any constraint has virtual observation points"""
        return any(c.Xv is not None for c in self.constraints)

    def _constrprob_xs_1(self, XS, i, nu):
        # done
        """
        Return the probability that the i-th constraint is satisfied at XS
        For use when no constraint point supplied by user
        C~(XS) | Y
        """
        
        # Get mean and cov
        mu, cov = self._constr_posterior_dist_1(XS, i)
        std = np.sqrt(np.diagonal(cov))

        # Get bound vectors for constraint distribution
        LB, UB = self.calc_constr_bounds_subop(XS, i)
        
        # Widen intervals with nu
        LB = LB - nu
        UB = UB + nu

        # Calculate probability that the constraint holds at each XV
        return norm_cdf_int_approx(np.array(mu)[:,0], std, LB, UB) # Aprroximation within E-7 error

    def _constr_posterior_dist_1(self, XS, i):
        # done
        """
        Return mean and covariance of the i-th constraint at XS
        
        C~(XS) | Y
        """
        
        if not self.cached:
            self._prepare()

        # c_v2, c_A2 and c_B2
        c_v2, c_A2, c_B2 = self._constr_prep_1(XS, i)
        
                            
        # Get mean vector
        Lmu_all, _ = self._calc_constr_mean()

        # Get i-th block of Lmu
        constraint = self.constraints[i]
        n_XS = XS.shape[0]

        # Slice out Lmu_i from stacked Lmu_all
        if constraint.Xv is not None:
            sizes = [c.Xv.shape[0] for c in self.constraints if c.Xv is not None]
            idx_start = sum(sizes[:i])
            idx_end = idx_start + sizes[i]
            Lmu_i = Lmu_all[idx_start:idx_end]
        else:
            Lmu_i = np.zeros((n_XS, 1))    

        # Posterior mean
        mu = Lmu_i + c_A2@self.f_tilde
        
        # Return posterior mean and covariance
        return mu, c_B2

    def _constr_prep_1(self, XS, i): # i think _calc_L1L2 and _calc_L2T can be generalised to do this or maybe it would be easier to just write this out
        """
        Return c_v2, c_A2 and c_B2 for constraint distribution
        """
        # done
        constraint = self.constraint[i]
        L2T_K_X_XS = constraint.op.K_with_f(self.kernel, self.X, XS)
        L1L2T_K_XS_XS = constraint.op.K_with_L(self.kernel, self.X, constraint.op, XS)

        c_v2 = cho_solve((self.L, True), L2T_K_X_XS) 
        c_A2 = cho_solve((self.L.T, False), c_v2).T 
        c_B2 = L1L2T_K_XS_XS - c_v2.T@c_v2
        
        return c_v2, c_A2, c_B2

    def calc_constr_bounds_subop(self, XS, i):
        # done
        """ Return lower/upper bounds for the i-th suboperator only at XS """
        
        constraint = self.constraints[i]
        LB = constraint.LBXV(XS)
        UB = constraint.UBXV(XS)
        return LB, UB


    def _constrprob_xs_2(self, XS, i, nu, num_samples, algorithm, verbatim = False):
        # done
        """
        Return the probability that the i-th constraint is satisfied at XS
        For use when constraint points supplied by user

        C~(XS) | Y, C
        """
        
        # Calculations only depending on (X, Y)
        if not self.cached:
            self._prepare()

        # Calculations only depending on (X, XV) - v1, A1 and B1
        L2T_K_x_xv = self._calc_L2T(self.X_training)
        L1L2T_K_xv_xv = self._calc_L1L2()
        v1 = cho_solve((self.L, True), L2T_K_x_xv, check_finite=False)
        A1 = cho_solve((self.L.T, False), v1).T

        n = L1L2T_K_xv_xv.shape[0]
        B1 = L1L2T_K_xv_xv + (self.var_constr*np.identity(n)) - v1.T@v1

        # Calculate mean of constraint distribution at XV (covariance is B1)
        Lmu_XV, constr_mean = self._calc_constr_mean()
        
        # Get bound vectors for constraint distribution
        LB, UB = self._calc_constr_bounds()
        
        # Sample from truncated constraint distribution 
        self._sample_constr_XV(m = num_samples, mu = constr_mean, sigma = B1, LB = LB, UB = UB, algorithm = algorithm, resample = False, verbatim = verbatim)

        # c_v2, c_A2 and c_B2
        # Only compute diagonal elements of constraint covariance

        constraint = self.constraints[i]

        # Compute covariance blocks
        L2T_K_X_XS = constraint.op.K_with_f(self.kernel, constraint.Xv, self.X_training).T
        K_XS_XS = constraint.op.K_with_L(self.kernel, XS, constraint.op, XS)

        # Take diagonal
        L1L2T_K_XS_XS_diag = np.diag(K_XS_XS).reshape(-1, 1)


        c_v2 = cho_solve((self.L, True), L2T_K_X_XS) 
        c_A2 = cho_solve((self.L.T, False), c_v2).T

        # c_A, c_B and c_Sigma
        L1L2T_XS_XV = self._calc_FiL2T(XS, i)
        
        c_B3 = L1L2T_XS_XV - c_v2.T@v1
        
        self._prep_L1() # Compute L_1
        L_1 = self._compute_cholesky(B1, only_L = True)

        c_v3 = cho_solve((L_1, True), c_B3.T)

        c_A = cho_solve((L_1.T, False), c_v3).T
        
        c_B = c_A2 - c_A@A1
        
        c_Sigma_diag = (
                        L1L2T_K_XS_XS_diag.flatten() 
                        - np.sum(np.square(c_v2), axis=0) 
                        - np.sum(np.square(c_v3), axis=0)
                        ).reshape(-1, 1)

        # Get bound vectors for constraint distribution
        LB, UB = self.calc_constr_bounds_subop(XS, i)

        # Widen intervals with nu
        LB = LB - nu
        UB = UB + nu
        
        # Prior mean
        constraint = self.constraints[i]
        mu_xs = self._mean_function(XS)              # shape (n_XS, 1)
        Lmu = constraint.op.apply_mean(mu_xs)
        
        # Posterior mean
        mu = Lmu + c_A@(self.C_sim - Lmu_XV) + c_B@self.f_tilde
        
        # Posterior standard deviation
        std = np.array(np.sqrt(c_Sigma_diag)).flatten()

        # Calculate probability that the constraint holds at each XS individually 
        # for each sample C_j and take the average over C_j
        if XS.shape[0] == 1:
            
            # Faster for single input
            probs = norm_cdf_int_approx(np.array(mu)[0], std, LB, UB)
            probs = np.array([probs.mean()])
            
        else:
            probs = np.apply_along_axis(norm_cdf_int_approx, axis = 0, arr = np.array(mu), std = std, LB = LB, UB = UB)
            probs = probs.mean(axis = 1)
            
        # Return probability
        return probs

    def _calc_FiL2T(self, XS: np.ndarray, i: int) -> np.ndarray:
        # done
        """
        Return block-row i of the constraint covariance cross-term:
        Cov[L_i f(XS), L_j f(Xv_j)] for all j (i.e. Fi L2^T K_XS_XV)
        """
        constraint_i = self.constraints[i]
        blocks = []

        for constraint_j in self.constraints:
            if constraint_j.Xv is not None:
                # Cov[L_i f(XS), L_j f(Xv_j)]
                K_block = constraint_i.op.K_with_L(self.kernel, XS, constraint_j.op, constraint_j.Xv)
                blocks.append(K_block)

        return np.block(blocks) if blocks else np.zeros((XS.shape[0], 0))


    def _argmin_pc_subop_finite(self, 
                                i, 
                                nu, 
                                Omega, 
                                batch_size = None, 
                                sampling_alg = 'ess', 
                                num_samples = 1000):
        # unworked
        """
        Same as _armin_pc_subup but over a finite domain Omega
        """

        # Calculations only depending on (X, Y)
        if not self.cached:
            self._prepare()

        if batch_size is None: batch_size = Omega.shape[0]

        # Split Omega in batches
        assert batch_size <= Omega.shape[0], 'batch_size must be less than number of elements in Omega'

        num_intervals, rem = np.divmod(Omega.shape[0], batch_size)

        # Compute constraint probability for each element in Omega
        if not self._has_virtual_obs():
            
            p_c = []
            for j in range(num_intervals): 
                p_c += list(self._constrprob_xs_1(Omega[j*batch_size:(j+1)*batch_size], i, nu))

            if rem != 0:
                p_c += list(self._constrprob_xs_1(Omega[-rem:], i, nu))
        else:
            p_c = []
            for j in range(num_intervals): 
                    p_c += list(self._constrprob_xs_2(Omega[j*batch_size:(j+1)*batch_size], i, nu, num_samples, sampling_alg))

            if rem != 0:
                    p_c += list(self._constrprob_xs_2(Omega[-rem:], i, nu, num_samples, sampling_alg))

        # Find smallest element
        p_c = np.array(p_c)
        idx = p_c.argmin()
        prob = p_c[idx]
        argmin = Omega[idx]

        return prob, argmin

    def _sample_constr_XV(self, m, mu, sigma, LB, UB, algorithm, resample = False):
        """ 
        Generate m samples from the constraint distribution
        
        Input: 
        m -- number of samples
        mu, sigma, LB, UB -- distribution parameters of truncated Gaussian
        algorithm -- name of sampling algorithm ('rejection', 'gibbs' or 'minimax_tilting')
        resample -- resample = False -> Use old samples if they exist
        
        """
        n = mu.size
        M = np.vstack([np.eye(n), -np.eye(n)])
        g = np.hstack([-LB, UB])

        # Check if we should just use the old samples
        if self.C_sim is None: 
            generate_samples = True
        else:
            if m == self.C_sim.shape[1]:
                generate_samples = resample
            else:
                generate_samples = True
                    

        if generate_samples:
            # Generate samples        
            #self.C_sim = rtmvnorm(n = m, mu = mu, sigma = sigma, a = LB, b = UB, algorithm = algorithm).T
        
            self.C_sim = sample_tmvn_ess(
                mu=mu,
                cov=sigma,
                A=M,
                b=g,
                X=self.X_training,
                y=self.Y,
                n_samples=m,
                burn_in=100,
                eta_init=20.0,
                update_eta=False
            )
