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
        - kwargs: Hyperparameters for the kernel (e.g. sigma, l).
        """

        self.kernel_type: str = kernel_type
        self.params: dict[str, float]  = kwargs

    def compute(self, x1: np.ndarray, x2: np.ndarray|None =None) -> np.ndarray:
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
            x2 = x1
        
        if self.kernel_type == "SE":
            covariance_matrix = self._SE(x1, x2)
            return covariance_matrix
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
    # defining kernels below
    def _SE(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """The square exponential covariance function. 
        Only works for 1D input. And with itself.
        """

        # ensure arrays are of shape (n,1)
        x1 = self._ensure_shape(x1)
        x2 = self._ensure_shape(x2)
        
        sigma: float = self.params.get("sigma", 1)
        l: float = self.params.get("l", 1)

        r2: np.ndarray = cdist(x1, x2, metric='sqeuclidean')
        K: np.ndarray = sigma ** 2 * np.exp(-0.5 * r2 / ( l ** 2))

        self.name = "Square Exponential"

        return K
    
    # utilities
    @staticmethod
    def _ensure_shape(self, x: np.ndarray) -> np.ndarray:
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


    
    def fit(self, x1: np.ndarray, x2: np.ndarray, f: np.ndarray, stabilise: bool = False, jitter_value: float = 1e-10) -> None:
        """
        fits the GP with the training data
        input of training data x1 and f, test data x2, whether to stabilise with jitter and jitter value
        increase jitter value if error like 
        uses algorithm 2.1 from gp book
        returns the mean and covariance metric
        """
        # TODO: add functionality for observations with errors
        mean_train, mean_test = self._set_means(x1, x2)
        K_11, K_12, K_22 = self._set_kernels(x1, x2, stabilise, jitter_value)

        try:
            # Perform Cholesky decomposition
            L = cholesky(K_11, lower=True)
        except Exception as e:
            raise ValueError("The matrix K_11 probably is not semi-positive definite. Using a jitter value, or increasing it can help with numerical stability.") from e

        f_tilde = f - mean_train # setting f_tilde to have mean of zero

        alpha = cho_solve((L,True), f_tilde) # cho_solve does A\x (where A = LL.T) as opposed to using solve_triangular to find (L\x) and then (L.T \ (L\x)). L\x is Lx=F

        v = solve_triangular(L, K_12, lower=True)
        
        self.mean_star = mean_test + (K_12.T @ alpha)
        self.cov_star = K_22 - (v.T @ v)



    def posterior(self, n: int = 1, mu: np.ndarray|float = None, cov: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        uses covariance metric and mean to produce 1 function from the GP
        input: n samples
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
    
    # utils
    def _set_means(self, x1: np.ndarray, x2: np.ndarray) -> tuple[np.ndarray|float, np.ndarray|float]:
        if callable(self.prior_mean):
            mean_train = self.prior_mean(x1)
            mean_test = self.prior_mean(x2)
        else:
            mean_train = self.prior_mean
            mean_test = self.prior_mean
        return mean_train, mean_test

    def _set_kernels(self, x1: np.ndarray, x2: np.ndarray, stabilise: bool, jitter_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        K_11 = self.kernel.compute(x1)
        K_12 = self.kernel.compute(x1, x2)
        K_22 = self.kernel.compute(x2)

        if stabilise:
            # add jitter to the diagonals, helps with numerical stability
            K_11[np.diag_indices_from(K_11)] += jitter_value
        return K_11, K_12, K_22
    

class EosProperties:
    """
    Example usage:
    given initial values of epsilon, p, and mu, and n:
    n = np.array([...])  # Example input for n
    phi = np.array([...])  # Example input for phi
    eos = EosProperties(mu_0, epsi_0, p_0, n, phi)
    results = eos.get_all()
    print(results)
    """
    def __init__(self,  n, phi, epsi_0, p_0, mu_0):
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
    