import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import cdist
from scipy.linalg import cho_solve, solve_triangular, cholesky
from scipy.integrate import cumulative_simpson as cumsimp

from collections.abc import Callable




class Kernel:
    def __init__(self, kernel_type: str ="SE", **kwargs: float) -> None: 
        """
        Initialize the kernel with a specific type and its hyperparameters.

        Parameters:
        - kernel_type: Type of kernel ('SE' for sqaured exponential, more later).
        - kwargs: Hyperparameters for the kernel (e.g. sigma, l). The naming is very specific
            -- For SE kernel, the arguments must be named 'sigma' and 'l'
        """

        self.kernel_type: str = kernel_type
        self.params: dict[str, float]  = kwargs

    def compute(self, x1: np.ndarray, x2: np.ndarray|None = None) -> np.ndarray:
        """
        Compute the covariance matrix for the given inputs.

        Parameters:
        - x1: First input array.
        - x2: Second input array (optional, defaults to x1 for self-covariance).
        arrays can either be 1D or in shape (n,1)

        Returns:
        - Covariance matrix (NxN or NxM).
        """
        
        if x2 is None:
            x2: np.ndarray = x1
        
        if self.kernel_type == "SE":
            covariance_matrix: np.ndarray = self._SE(x1, x2)
            return covariance_matrix
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
    @staticmethod
    def visualise_kernel(covmat: np.ndarray, title: str|None = None) -> None:
        """Visualise the covariance matrix as a heatmap"""
        if title is None:
            Title = 'Covariance Matrix'
        else:
            Title = title

        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = sns.heatmap(covmat, annot=True, fmt=".2g", ax=ax, cmap='mako')
        ax.set_title(Title)
        plt.show()



        
    # defining kernels below
    def _SE(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """The square exponential covariance function. 
        Only works for 1D input. And with itself.
        """

        # ensure arrays are of shape (n,1)
        x1: np.ndarray = self._ensure_shape(x1)
        x2: np.ndarray = self._ensure_shape(x2)
        
        sigma: float = self.params.get("sigma", 1)
        l: float = self.params.get("l", 1)

        r2: np.ndarray = cdist(x1, x2, metric='sqeuclidean')
        K: np.ndarray = sigma ** 2 * np.exp(-0.5 * r2 / ( l ** 2))

        self.name = "Square Exponential"

        return K
    
    # utilities
    @staticmethod
    def _ensure_shape(x: np.ndarray) -> np.ndarray:
        """Ensure that the input array is of shape (n,1)."""
        if x.ndim == 1:
            return x.reshape(-1,1)
        return x
    
    def __str__(self) -> str:
        param_str = ""
        for key, value in self.params.items():
            if param_str:
                # Add a comma if param_str is not empty
                param_str += f", {key} = {value}"
            else:
                # Add the first key-value pair without a comma
                param_str += f"{key} = {value}"

        return f"Kernel is {self.kernel_type} with hyperparameters: {param_str})"



class GP:
    # very basic, no noise for now
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
        #self.log_marginal_likelihood = None #is this needed?

    
    def fit(self, x1: np.ndarray, x2: np.ndarray, f: np.ndarray, var_f: np.ndarray|None = None, stabilise: bool = False, jitter_value: float = 1e-10, ) -> None:
        """
        fits the GP with the training data
        input of training data x1 and f with noise var_f, test data x2, whether to stabilise with jitter and jitter value
        variance in training data can be supplied using var_f, leave empty if noise free data

        uses algorithm 2.1 from gp book
        returns the mean and covariance metric

        commented argument:  return_marginal_likelihood: bool = False,
        """
        # DOING: add functionality for observations with errors
        mean_train, mean_test = self._set_means(x1, x2)
        K_11, K_12, K_22 = self._set_kernels(x1, x2, var_f, stabilise, jitter_value)

        f_tilde = f - mean_train # setting f_tilde to have mean of zero

        L, alpha = self._compute_cholesky(K_11, f_tilde)

        v = solve_triangular(L, K_12, lower=True) # delete this comment: 2.1 Line 5

        # if return_marginal_likelihood:
        #   self._get_log_likelihood(f_tilde, alpha, L, len(alpha))
        # no point in getting likelihood here yet

        self.mean_star = mean_test + (K_12.T @ alpha) # delete this comment: 2.1 Line 4
        self.cov_star = K_22 - (v.T @ v) # delete this comment: 2.1 Line 6



    def posterior(self, n: int = 1, mu: np.ndarray|float = None, cov: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        uses covariance metric and mean to produce 1 function from the GP
        input: n samples, optionally mean and cov
        """

        if mu is None:
            mu = self.mean_star
        if cov is None:
            cov = self.cov_star

        if self.mean_star is None or self.cov_star is None:
            print("Use the fit function first before producing the GP or provide mean and kernel")

        rng = np.random.default_rng()
        y = rng.multivariate_normal(mean=mu, cov=cov, size=n)
        sig = np.sqrt(np.diag(self.cov_star))

        return y, sig
    
    def get_log_likelihood(self, x: np.ndarray, f: np.ndarray, var_f: np.ndarray|None, stabilise: bool = True, jitter_value: float = 10e-10) -> float:
        """eq 2.30 and line 7 in algo 2.1"""

        K11 = self._set_K_11(x, var_f, stabilise, jitter_value)
        L, alpha = self._compute_cholesky(K11, f)
        n = len(alpha)

        term1 = -(1/2)*(f@alpha) #np way for f.T alpha
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

    def _set_kernels(self, x1: np.ndarray, x2: np.ndarray, var_f: None|np.ndarray|float, stabilise: bool, jitter_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        K_11 = self._set_K_11(x1, var_f, stabilise, jitter_value)
        K_12 = self.kernel.compute(x1, x2)
        K_22 = self.kernel.compute(x2)


        return K_11, K_12, K_22
    
    def _set_K_11(self, x1: np.ndarray, var_f: None|np.ndarray|float, stabilise: bool, jitter_value: float) -> np.ndarray:

        K_11 = self.kernel.compute(x1)

        if var_f is not None:
            var_f = np.atleast_1d(var_f)
            K_11 += (np.eye(len(var_f))*(var_f**2))

        if stabilise:
            # add jitter to the diagonals, helps with numerical stability
            K_11[np.diag_indices_from(K_11)] += jitter_value

        return K_11

    
    def _compute_cholesky(self, K_11: np.ndarray, f: np.ndarray) -> np.ndarray|tuple:
        try:
            # Perform Cholesky decomposition
            L = cholesky(K_11, lower=True) # delete this comment: 2.1: Line 2
        except Exception as e:
            raise ValueError("The matrix K_11 probably is not semi-positive definite. Using a jitter value (set stabilise=True), or increasing it (default jitter_value=1e-10) can help with numerical stability.") from e
        
        # delete this comment: below is 2.1 Line 3
        alpha = cho_solve((L,True), f) # cho_solve does A\x (where A = LL.T) as opposed to using solve_triangular to find (L\x) and then (L.T \ (L\x)). L\x is Lx=F

        return L, alpha
    

class EosProperties:
    """
    Example usage:
    given initial values of epsilon, p, and mu, and n:
    n = np.array([...])  # Example input for n
    phi = np.array([...])  # Example input for phi
    eos = EosProperties(mu_0, epsi_0, p_0, n, phi)
    results = eos.get_all()
    print(results)

    TODO: make the functions be able to work outside of class, i.e take arguments other than self
    """
    def __init__(self,  n: np.ndarray, phi: np.ndarray, epsi_0: float, p_0: float, mu_0: float) -> None:
        self.mu_0 = mu_0
        self.epsi_0 = epsi_0
        self.p_0 = p_0
        self.n = n
        self.phi = phi
        self.cs2 = None
        self.mu = None
        self.epsilon = None
        self.pressure1 = None
        self.pressure2 = None

    def get_cs2(self):
        """ab initio qcd paper, eq 7
        unitless
        """
        self.cs2 = 1 / (np.exp(-self.phi) + 1)
        return self.cs2

    def get_mu(self):
        """eq 8
        mu_0 is in Mev usually so then is mu
        """
        integrand = self.cs2 / self.n
        integral = cumsimp(y=integrand, x=self.n, initial=np.log(self.mu_0))
        log_mu = np.log(self.mu_0) + integral

        try:
            # Try to compute mu
            self.mu = np.exp(log_mu)
        except Exception as e:
            # Return log_mu if it doesn't work
            print("Error in exponentiating, returning log_mu instead")
            self.mu = log_mu
        
        #TODO: make sure log_mu return is handled when get_all() is called

        return self.mu

    def get_epsilon(self):
        """eq 9
        MeV * fm^-3
        """
        self.epsilon = self.epsi_0 + cumsimp(y=self.mu, x=self.n, initial=self.epsi_0)
        return self.epsilon

    def get_pressure1(self):
        """below eq 9"""
        self.pressure1 = -self.epsilon + self.mu * self.n
        return self.pressure1

    def get_pressure2(self):
        """Hauke's eq"""
        integrand = self.cs2 * self.mu
        integral = cumsimp(y=integrand, x=self.n, initial=self.p_0)
        self.pressure2 = self.p_0 + integral
        return self.pressure2

    def get_all(self):
        """
        get all properties in a dictionary at once
        """
        self.get_cs2()
        self.get_mu()
        self.get_epsilon()
        self.get_pressure1()
        self.get_pressure2()
        # Return all computed values as a dictionary for easy access
        return {
            "cs2": self.cs2,
            "mu": self.mu,
            "epsilon": self.epsilon,
            "pressure1": self.pressure1,
            "pressure2": self.pressure2
        }
    