import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist


class Kernel:
    def __init__(self, kernel_type: str ="SE", **kwargs: float) -> None: 
        """
        Initialize the kernel with a specific type and its hyperparameters.

        Parameters:
        - kernel_type: Type of kernel
            -- 'SE' for sqaured exponential
            -- 'RQ' for rational quadratic
            -- 'Matern32' for Matern 3/2
            -- 'Matern52' for Matern 5/2
            -- 'GE' for gamma exponetial
        - kwargs: Hyperparameters for the kernel (e.g. sigma, l). The naming is very specific
            -- For SE or Matern kernel, the arguments must be named 'sigma' and 'l'
            -- For RQ kernel, the arguments must be named 'sigma', 'l', and 'alpha'
            -- For GE kernel, the arguments must be named 'sigma', 'l', and 'gamma'
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


        match self.kernel_type:
            case "SE":
                covariance_matrix: np.ndarray = self._SE(x1, x2)
            case "RQ":
                covariance_matrix: np.ndarray = self._RQ(x1, x2)
            case "M32":
                covariance_matrix: np.ndarray = self._M32(x1, x2)
            case "M52":
                covariance_matrix: np.ndarray = self._M52(x1, x2)
            case "GE":
                covariance_matrix: np.ndarray = self._GE(x1, x2)
            case _:
                raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        return covariance_matrix

        
    @staticmethod
    def visualise_kernel(covmat: np.ndarray, title: str|None = None, annotation: bool = True) -> None:
        """Visualise the covariance matrix as a heatmap"""
        if title is None:
            Title = 'Covariance Matrix'
        else:
            Title = title

        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = sns.heatmap(covmat, annot=annotation, fmt=".2g", ax=ax, cmap='mako')
        ax.set_title(Title)
        plt.show()



        
    # defining kernels below
    def _SEr(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """The square exponential covariance function. 
        Only works for 1D input.
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

    def _RQ(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """The rational quadratic covariance function. 
        Only works for 1D input.
        """

        # ensure arrays are of shape (n,1)
        x1: np.ndarray = self._ensure_shape(x1)
        x2: np.ndarray = self._ensure_shape(x2)
        
        sigma: float = self.params.get("sigma", 1)
        l: float = self.params.get("l", 1)
        alpha: float = self.params.get("alpha", 1)

        r2: np.ndarray = cdist(x1, x2, metric='sqeuclidean')
        K: np.ndarray = sigma ** 2 * (1 + (-0.5 * r2 / ( alpha * (l ** 2))))**(-alpha)

        self.name = "Rational Quadratic"

        return K
    

    def _M32(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """The Matern 3/2 covariance function. 
        Only works for 1D input.
        """

        # ensure arrays are of shape (n,1)
        x1: np.ndarray = self._ensure_shape(x1)
        x2: np.ndarray = self._ensure_shape(x2)
        
        sigma: float = self.params.get("sigma", 1)
        l: float = self.params.get("l", 1)

        r: np.ndarray = np.sqrt(cdist(x1, x2, metric='sqeuclidean'))
        K: np.ndarray = sigma ** 2 * (1 + (np.sqrt(3) * r)/l) * np.exp(- ((np.sqrt(3) * r) /l ))

        self.name = "Matern 3/2"

        return K
    

    def _M52(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """The Matern 5/2 covariance function. 
        Only works for 1D input.
        """

        # ensure arrays are of shape (n,1)
        x1: np.ndarray = self._ensure_shape(x1)
        x2: np.ndarray = self._ensure_shape(x2)
        
        sigma: float = self.params.get("sigma", 1)
        l: float = self.params.get("l", 1)

        r: np.ndarray = np.sqrt(cdist(x1, x2, metric='sqeuclidean'))
        K: np.ndarray = sigma ** 2 * (1 + (np.sqrt(5) * r)/l) * np.exp(- ((np.sqrt(5) * r) /l ))

        self.name = "Matern 5/2"

        return K
    
    def _GE(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """The gamma exponential covariance function. 
        Only works for 1D input.
        """

        # ensure arrays are of shape (n,1)
        x1: np.ndarray = self._ensure_shape(x1)
        x2: np.ndarray = self._ensure_shape(x2)
        
        sigma: float = self.params.get("sigma", 1)
        l: float = self.params.get("l", 1)
        gamma: float = self.params.get("gamma", 1)

        r: np.ndarray = np.sqrt(cdist(x1, x2, metric='sqeuclidean'))
        K: np.ndarray = sigma ** 2 * np.exp(-0.5 * ( r/l)** gamma)

        self.name = "Gamma-Exponential"

        return K
    def _SE(self, x1, x2):
        """
        Square-exponential using broadcasting
        """
        # (n,1)   (m,1)
        x1 = self._ensure_shape(x1)
        x2 = self._ensure_shape(x2)

        sigma = self.params.get("sigma", 1.0)
        l     = self.params.get("l",     1.0)

        # pairwise squared distance  (n,m) via broadcasting
        r2 = (x1 - x2.T)**2          # no cdist

        K  = sigma**2 * np.exp(-0.5 * r2 / l**2)
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


