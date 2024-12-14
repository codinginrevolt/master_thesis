import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sp
from scipy.integrate import cumulative_simpson as cumsimp


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
        self.covariance_matrix = None 

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
            self.covariance_matrix = self._rbf_kernel(x1, x2)
            return self.covariance_matrix
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
    def visualise(self, x1, x2=None):
        """
        Visualise the covariance matrix for the given inputs.

        Parameters:
        - x1: First input array.
        - x2: Second input array (optional, defaults to x1 for self-covariance).
        """
        if self.covariance_matrix is None: 
            self.compute(x1, x2)  # Compute and store the result
        sns.heatmap(self.covariance_matrix, cmap="magma")
        plt.show() 

    # defining kernels below
    def _SE(self, x1, x2):
        """The square exponential covariance function. 
        Only works for 1D input. And with itself."""
        
        sigma = self.params.get("sigma", 1)
        l = self.params.get("l", 1)
        r = np.subtract.outer(x1, x2)
        K = sigma ** 2 * np.exp(-r / (2 * l ** 2))
        
        return K

