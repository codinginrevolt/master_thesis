import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sp


class Kernel:
    def __init__(self, kernel_type="SE", **kwargs):
        """
        Initialize the kernel with a specific type and hyperparameters.

        Parameters:
        - kernel_type: Type of kernel ('SE' for sqaured exponential, more later).
        - kwargs: Hyperparameters for the kernel (e.g., sigma, l).
        """
        self.kernel_type = kernel_type
        self.params = kwargs

    def compute(self, x1, x2=None):
        """
        Compute the covariance matrix for the given inputs.

        Parameters:
        - x1: First input array.
        - x2: Second input array (optional, defaults to x1 for self-covariance).

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
    def _SE(self, x1, x2):
        """The square exponential covariance function. 
        Only works for 1D input. And with itself."""
        
        sigma = self.params.get("sigma", 1)
        l = self.params.get("l", 1)
        r = np.subtract.outer(x1, x2)
        K = sigma ** 2 * np.exp(-r / (2 * l ** 2))

        #if x1.shape == x2.shape:
        #    jitter = 1e-10
        #    K += np.diag(np.full(x1.shape[0], jitter))
        
        return K


class GP:
    # very basic, no noise for now
    def __init__(self):
        pass

    def fit(x1, x2, f, kernel):

        pass