import numpy as np
from scipy.linalg import cho_solve, solve_triangular, cholesky
from collections.abc import Callable
from kernels import Kernel

class GP:
    """
    Inputs: 
    - kernel
    - prior mean: mean function to be evaluated at x_train and x_test or a global mean
    """
    def __init__(self, kernel: Kernel, prior_mean: float|np.ndarray|Callable[[np.ndarray],np.ndarray]|None = None) -> None:
        self.kernel = kernel
        
        if prior_mean is not None: 
            self.prior_mean = prior_mean
        else:
            self.prior_mean = 0

        self.mean_star = None
        self.cov_star = None

    
    def fit(self, x1: np.ndarray, x2: np.ndarray, f: np.ndarray, var_f: float|np.ndarray|None = None, stabilise: bool = False, jitter_value: float = 1e-10) -> None:
        """
        fits the GP with the training data
        input of training data x1 and f with noise var_f, test data x2, whether to stabilise with jitter and jitter value
        variance in training data can be supplied using var_f, leave empty if noise free data

        uses algorithm 2.1 from gp book
        returns the mean and covariance metric
        """

        mean_train, mean_test = self._set_means(x1, x2)
        K_11, K_12, K_22 = self._set_kernels(x1, x2, var_f, stabilise, jitter_value)

        f_tilde = f - mean_train # setting f_tilde to have mean of zero

        L, alpha = self._compute_cholesky(K_11, f_tilde)

        v = solve_triangular(L, K_12, lower=True, check_finite=False) # 2.1 Line 5


        self.mean_star = mean_test + (K_12.T @ alpha) # 2.1 Line 4
        self.cov_star = K_22 - (v.T @ v) # 2.1 Line 6



    def posterior(self, n: int = 1, mu: np.ndarray|float = None, cov: np.ndarray = None, sampling :bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        uses covariance metric and mean to produce the mean fuction with uncertainty or n function realisations from the GP
        input: n samples, optionally mean and cov if available from elsewhere 
        """

        if mu is None:
            mu = self.mean_star
        if cov is None:
            cov = self.cov_star

        if cov is None or mu is None:
            print("Use the fit function first before producing the GP or provide mean and kernel")


        if sampling:
            rng = np.random.default_rng()
            y = rng.multivariate_normal(mean=mu, cov=cov, size=n)
            return y

        y = mu.reshape(1,len(mu))
        sig = np.sqrt(np.diag(cov))

        return y, sig
    
    def get_log_likelihood(self, x: np.ndarray, f: np.ndarray, var_f: float|np.ndarray|None = None, stabilise: bool = False, jitter_value: float = 1e-10) -> float:
        """eq 2.30 and line 7 in algo 2.1"""

        K11 = self._set_K_11(x, var_f, stabilise, jitter_value)
        L, alpha = self._compute_cholesky(K11, f)
        n = len(alpha)

        term1 = -(1/2)*(f@alpha) # numpy way for f.T alpha
        term2 = - np.sum(np.log(np.diag(L)))
        term3 = - (1/2)*n*np.log(2*np.pi)    

        log_marginal_likelihood = term1 + term2 + term3
        return log_marginal_likelihood
    
    # utils
    def _set_means(self, x1: np.ndarray, x2: np.ndarray) -> tuple[np.ndarray|float, np.ndarray|float]:
        if callable(self.prior_mean):
            mean_train = self.prior_mean(x1)
            mean_test = self.prior_mean(x2)
        else:
            mean_train = self.prior_mean
            mean_test = self.prior_mean
        return mean_train, mean_test

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

    
    def _compute_cholesky(self, K_11: np.ndarray, f: np.ndarray, sampling: bool = True) -> np.ndarray|tuple:
        if sampling: o_a = True
        else: o_a = False
        try:
            # Perform Cholesky decomposition
            L = cholesky(K_11, lower=True, overwrite_a=o_a, check_finite=False) # delete this comment: 2.1: Line 2
        except Exception as e:
            raise ValueError("The matrix K_11 probably is not semi-positive definite. Using a jitter value (set stabilise=True), or increasing it (default jitter_value=1e-10) can help with numerical stability.") from e
        
        # algo 2.1 Line 3
        alpha = cho_solve((L,True), f, overwrite_b=o_a, check_finite=False) # cho_solve does A\x (where A = LL.T) as opposed to using solve_triangular to find (L\x) and then (L.T \ (L\x)). L\x is Lx=F

        return L, alpha
    