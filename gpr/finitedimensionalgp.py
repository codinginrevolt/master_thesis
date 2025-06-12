import numpy as np
from scipy.linalg import cho_solve, solve_triangular, cholesky
from scipy.optimize import minimize

import tmg_hmc
from kernels import Kernel

import time

class FDGP:
    """
    Following notation mostly from DOI:10.1137/17M1153157
    """
    def __init__(
            self, 
            kernel: Kernel,
            x_train : np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            var_y: float|np.ndarray|None = None, 
            prior_mean: float|np.ndarray|None = None,
            stabilise: bool = False, 
            jitter_value: float = 1e-10,
            m: int| None = None, # number of knots
            knots: np.ndarray| None = None,
            sampling: bool = True,
            ) -> None:
        
        self.kernel = kernel
        
        self.x_min = np.min(x_train)
        self.x_max = np.max(x_train)
 
        self.x_train = self._normalise(x_train)

        self.y_train = y_train

        self.x_test = self._normalise(x_test)
 
        self.stabilise = stabilise
        self.jitter_value = jitter_value

        self.sampling = sampling

        if 'l' in self.kernel.params:
            self.kernel.params['l'] = self._normalise(self.kernel.params['l'])
        else:
            raise ValueError("'l' parameter not found in kernel parameters")


        if knots is not None:
            self.knots = self._normalise(knots)
            self.m = len(knots)
        else:
            if m is None:
                raise ValueError("Either supply predefined knots or specify number of knots for uniform placement.")
            self.m = m
            self.knots = np.linspace(self.x_train[0], self.x_train[-1], m) 
 
        self.delta_m = 1/(self.m-1)

        self.js = np.arange(0, m, 1)

        self.Gamma = self.kernel.compute(self.knots)
        self.Phi_matrix = self._phi_j(self.js[None,:], self.x_train[:,None])

        self.PhiGammaPhiT = self._set_kernel(var_y)

        if prior_mean is not None: 
            self.prior_mean = prior_mean
        else:
            self.prior_mean = 0

        self.mean_star = None
        self.cov_star = None
        self.constraints = None
        self.map = None
        self.cov_star_inverse = None


    def fit(self):
        """
        uses algorithm 2.1 from the gp book
        returns the mean and covariance metric
        """

        f_tilde = self.y_train - self.prior_mean # setting f_tilde to have mean of zero

        L, alpha = self._compute_cholesky(self.PhiGammaPhiT, f_tilde, self.sampling)

        PhiGamma = self.Phi_matrix @ self.Gamma

        v = solve_triangular(L, PhiGamma, lower=True, check_finite=False) # 2.1 Line 5

        self.mean_star = self.prior_mean + (PhiGamma.T @ alpha) # 2.1 Line 4
        self.cov_star = self.Gamma - (v.T @ v) # 2.1 Line 6

    def _normalise(self, x):
        """ 
        normalised to 0 and 1
        """

        return (x - self.x_min)/(self.x_max-self.x_min)

    def _phi_j(self, j, x_i):
        t_j = j * self.delta_m
        abs_val = np.abs((x_i - t_j)/self.delta_m)

        return np.where(abs_val <= 1, 1 - abs_val, 0)
        

    def _set_kernel(self, var_f: None|np.ndarray|float= None) -> np.ndarray:

        kern = self.Phi_matrix @ self.Gamma @ (self.Phi_matrix).T

        if var_f is not None:
            var_f = np.atleast_1d(var_f)
            kern += (np.eye(len(var_f))*(var_f**2))

        if self.stabilise:
            # add jitter to the diagonals, helps with numerical stability
            kern[np.diag_indices_from(kern)] += self.jitter_value

        return kern
    
    def _compute_cholesky(self, kern: np.ndarray, f: np.ndarray, sampling: bool = True) -> np.ndarray|tuple:
        if sampling: o_a = True
        else: o_a = False
        try:
            # Perform Cholesky decomposition
            L = cholesky(kern, lower=True, overwrite_a=o_a, check_finite=False) # delete this comment: 2.1: Line 2
        except Exception as e:
            raise ValueError("The covariance matrix probably is not semi-positive definite. Using a jitter value (set stabilise=True), or increasing it (default jitter_value=1e-10) can help with numerical stability.") from e
        
        # algo 2.1 Line 3
        alpha = cho_solve((L,True), f, overwrite_b=o_a, check_finite=False) # cho_solve does A\x (where A = LL.T) as opposed to using solve_triangular to find (L\x) and then (L.T \ (L\x)). L\x is Lx=F

        return L, alpha

    def compute_MAP(self):
        """
        Finds the maximum a posterior (MAP) which is the posterior mode.
        Can be thought of as the unconstrained posterior mean shifted to respect the constraints
        Uses equation 8 in 10.1080/03610926.2022.2055768

        Parameters
        ----------
        lower_bound: float
        upper_bound: float 
        """
        if self.mean_star is None:
            raise ValueError("Fit the unconstrained FDGP first")

        Gamma_cond = self.cov_star + 1e-8*np.eye(self.cov_star.shape[0])
        L = cholesky(Gamma_cond, lower=True, check_finite=False)
        self.cov_star_inverse = cho_solve((L,True), np.eye(L.shape[0]), check_finite=False, overwrite_b=True)

        def objective(x):
            Gamma_inv_x = cho_solve((L, True), x, check_finite=False)
            return 0.5 * (x.T @ Gamma_inv_x) - (self.mean_star.T @ Gamma_inv_x)

        def jacobian(x):
            Gamma_inv_x = cho_solve((L, True), x, check_finite=False)
            Gamma_inv_mean = cho_solve((L, True), self.mean_star, check_finite=False)
            return Gamma_inv_x - Gamma_inv_mean
        
        M = self.constraints["M"]
        g = self.constraints["g"]

        constraints = [{'type': 'ineq', 'fun': lambda x: M @ x + g}]

        show_display = not self.sampling
        res = minimize(
            fun=objective,
            x0=self.mean_star, 
            method='SLSQP',
            jac=jacobian,
            constraints=constraints,
            options={'disp': show_display}
        )

        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        self.map = res.x
        return res.x 
        
    def _bounds_lineq_sys(self, m: int|None = None, A:np.ndarray|None = None, l: np.ndarray|float = 0, u:np.ndarray|float = 1, rm_inf: bool = True):
        if A is None:
            A = np.eye(m)
        else:
            m = A.shape[0]
        l = np.full(m, l) if np.isscalar(l) else np.array(l)
        u = np.full(m, u) if np.isscalar(u) else np.array(u)

        M = np.vstack((A,-A))
        g = np.concatenate((-l,u))

        if rm_inf and np.any((g == np.inf) | (g == -np.inf)):
            mask = ~np.isinf(g)
            M = M[mask]
            g = g[mask]
        
        return {"M":M, "g": g}

    def set_constraints(self, m: int|None = None, A: np.ndarray|None = None, constraint_type: str = 'boundedness', l:float|np.ndarray = -np.inf, u:float|np.ndarray=np.inf, rm_inf: bool = True):
        """
        Constructs a one-sided linear inequality constraint system:
        M x + g ≥ 0

        Note that you should specify the arguments names.

        Parameters
        ----------
        m : int or None
            The number of variables. Required for built-in constraints like 'boundedness', 'monotonicity',
            and 'convexity'. Not needed for 'linear' if `A` is provided.

        A : np.ndarray or None
            The constraint matrix. Required only for constraint_type='linear'. Ignored for other constraint types.

        constraint_type : str
            Type of constraint to apply. Must be one of:
                - 'boundedness': simple lower and upper bounds on each variable
                - 'monotonicity': enforces x₁ ≤ x₂ ≤ ... ≤ xₘ
                - 'convexity': enforces convexity based on second-order finite differences
                - 'linear': allows custom linear inequality constraints (provide `A`, `l`, and `u`)

        l : float or np.ndarray, optional
            Lower bound(s) for the inequality system. Can be a scalar or a 1D array.
            Default is `-np.inf`, indicating no lower bound.

        u : float or np.ndarray, optional
            Upper bound(s) for the inequality system. Can be a scalar or a 1D array.
            Default is `np.inf`, indicating no upper bound.

        rm_inf : bool, optional
            Whether to remove constraints that are unbounded on both sides (i.e., -inf < x < inf).
            Default is True.

        Returns
        -------
        dict
            containing:
                - 'M': the matrix M in M x + g ≥ 0
                - 'g': bounds vector g

        """
        if A is None and m is None:
            raise ValueError("Either A or m must be supplied")
        
        match constraint_type:
            case "boundedness":
                lin_sys = self._bounds_lineq_sys(m=m, l=l, u=u, rm_inf=rm_inf)
            
            case "monotonicity":
                if np.isscalar(l):
                    l = np.concatenate(([-np.inf], np.full(m-1, l)))
                if np.isscalar(u):
                    u = np.concatenate(([-np.inf], np.full(m-1, u)))
                
                A = np.eye(m)
                for i in range(1,m):
                    A[i, i-1] = -1
                lin_sys = self._bounds_lineq_sys(A=A, l=l, u=u, rm_inf=rm_inf)

            case "convexity":
                if np.isscalar(l):
                    l = np.concatenate(([-np.inf], np.full(m-2, l)))
                if np.isscalar(u):
                    u = np.concatenate(([-np.inf], np.full(m-2, u)))
                A = np.eye(m)
                for i in range(2, m):
                    A[i, i-2] = 1
                    A[i, i-1] = -2
                
                lin_sys = self._bounds_lineq_sys(A=A, l=l, u=u, rm_inf=rm_inf)

            case "linear":
                lin_sys = self._bounds_lineq_sys(A=A, l=l, u=u, rm_inf=rm_inf)

            case _:
                raise NotImplementedError("Constraint is not implemented. Define a constraint matrix and bounds and use option 'linear'. ")

        self.constraints = lin_sys
        return lin_sys

    def combine_constraints(self, *constraint_dicts):
            """
            Combine multiple constraint dictionaries (from `set_constraints`) into a single
            constraint system of the form Mx + g ≥  0.

            Parameters
            ----------
            *constraint_dicts : dict
                Any number of constraint dicts (with keys "M" and "g") to be stacked.

            Outcome
            -------
            Sets `self.constraints` to the the new combined dictionary.

            Example
            -------
            >>> fdgp = finitedimensionalgp.FDGP(kernel, x_train, y_train, x_test, y_noise, y_mean, stabilise = True, m=m, sampling=False)
            >>> constr1 = fdgp.set_constraints(m, constraint_type='boundedness', l = l, u=u, rm_inf=True)
            >>> constr2 = fdgp.set_constraints(A, constraint_type='linear', l = l, u=u, rm_inf=True)
            >>> fdgp.combine_constraints(const1, constr2)

            """
            M_all = [c["M"] for c in constraint_dicts if c is not None]
            g_all = [c["g"] for c in constraint_dicts if c is not None]

            M_combined = np.vstack(M_all)
            g_combined = np.concatenate(g_all)

            self.constraints = {"M": M_combined, "g": g_combined}

    def sample_posterior(self, n_samples, burn_in=100, return_trace=False, seed = None):
        """
        Sample from the constrained posterior.
        """
        start_time = time.time()
        Lam = self.constraints["M"]
        g = self.constraints["g"]

        init_vec = self.map

        Gamma_inv = self.cov_star_inverse

        r = Gamma_inv @ self.mean_star

        mid_time = time.time()

        print(f"Time elapsed before sampling: {mid_time - start_time:.4f} seconds")
        if return_trace:
            samples, traces = tmg(n_samples, M = Gamma_inv, r = r, initial = init_vec, f = Lam, g = g, burn_in = burn_in, checks = False, return_trace = return_trace, seed = seed)
            return samples, traces
        else:
            samples = tmg(n_samples, M = Gamma_inv, r = r, initial = init_vec, f = Lam, g = g, burn_in = burn_in, checks = True, return_trace = return_trace, seed = seed)
            end_time = time.time()
            print(f"Time elapsed in total: {end_time - start_time:.4f} seconds")
            return samples

    def evaluate(self, x, xi_sample):
        """Evaluate the GP approximation at x for a given sample of xi."""
        return sum(xi_sample[j] * self.phi_funcs[j](x) for j in range(self.m))



def tmg(n: int, M: np.ndarray, r: np.ndarray, initial, f: np.ndarray , g: np.ndarray , burn_in:int = 30, checks:bool = True, return_trace:bool = False, seed:int = None):
    start_time = time.time()

    M = (M + M.T)/2
    d = M.shape[0]

    if checks:
        if M.shape[1] != d:
            raise ValueError("M must be a square matrix")
        if len(initial) != d:
            raise ValueError("Wrong Length for initial vector")

        eigs = np.linalg.eigvalsh(M)
        if np.any(eigs<=0):
            raise ValueError("M must be positive definite")            
        # add check to see if f and g are not None
    R = cholesky(M, lower=True, check_finite=False)
    Minv_r = cho_solve((R, True), r)
    R_inv = solve_triangular(R, np.eye(R.shape[0]), lower=True)
    initial2 = (R@initial) - (r@R_inv)

    if isinstance(f, np.ndarray) and isinstance(g, np.ndarray):
        numlin = f.shape[0]
        if len(g) != numlin or f.shape[1] != d:
            raise ValueError("f must be an m-by-d matrix and g an m-dimensional vector.")
        if np.any(f@initial + g < 0):
            raise ValueError("Initial point must respect constraints")
        
        f2 = f @ R_inv
        g2 = f @ Minv_r + g
    else:
        raise ValueError("For linear constraints, f must be a matrix and g a vector.")

    mid_time = time.time()
    print(f"Time elapsed for checks: {mid_time - start_time:.4f} seconds")

    if return_trace:
        samples, traces = tmg_hmc.sample(n+burn_in, initial2, numlin, seed, f2, g2, return_trace)
        samples = samples[burn_in:, :]
        samples = samples @ R_inv.T + np.tile(Minv_r, (n,1))
        traces = traces[burn_in:]
        return(samples,traces)

    else:
        samples = tmg_hmc.sample(n+burn_in, initial2, numlin, seed, f2, g2, return_trace)
        end_time = time.time()
        print(f"Time elapsed for calling rust: {end_time - mid_time:.4f} seconds")
        samples = samples[burn_in:, :]
        samples = samples @ R_inv.T + np.tile(Minv_r, (n,1))
        return samples