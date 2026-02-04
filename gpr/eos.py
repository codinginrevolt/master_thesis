import numpy as np
from scipy.integrate import cumulative_simpson as cumsimp
from constants import ns

class EosProperties:
    """
    Class to compute EOS properties given n and either phi or cs2, along with initial conditions.
    Parameters
    ----------
     - n (np.ndarray): Array of densities (should be given in units of nuclear saturation density, n_sat). Internally converted to units of fm^-3.
     - phi (np.ndarray | None): Auxiliary variable used to compute cs2. If provided, cs2 will be calculated from phi. Set phi to None to use cs2 directly.
     - epsi_0 (float): Initial energy density (in MeV fm⁻³).
     - p_0 (float): Initial pressure (in MeV fm⁻³).
     - mu_0 (float): Initial chemical potential (in MeV).
     - cs2 (np.ndarray | None, default None): Optional. Speed of sound squared. Must be provided if phi is set to None.

    Example usage:
    given initial values of epsilon, p, and mu, and n:
    ```
    n = np.array([...])  # Example input for n | must be in nsat
    phi = np.array([...])  # Example input for phi
    eos = EosProperties(n, phi, epsi_0, p_0, mu_0)
    # Access results
    cs2 = results["cs2"]
    mu = results["mu"]
    epsilon = results["epsilon"]
    pressure = results["pressure"]
    ```
    ------------------------------------------------------------
    Note:
    If phi is not provided, cs2 must be provided:
    ```
    cs2 = np.array([...])
    eos = EosProperties(n, phi = None, epsi_0, p_0, mu_0, cs2=cs2)
    ```
    """

    def __init__(
        self,
        n: np.ndarray,
        phi: np.ndarray | None,
        epsi_0: float,
        p_0: float,
        mu_0: float,
        cs2: np.ndarray | None = None,
    ) -> None:
        self.mu_0 = mu_0
        self.epsi_0 = epsi_0
        self.p_0 = p_0
        self.n = n * ns  # fm^-3

        if phi is not None:
            self.phi = phi.flatten()
            self.cs2 = None

        if phi is None:
            if cs2 is None:
                raise ValueError("If phi is set to None, cs2 must be given")
            self.cs2 = cs2
            self.phi = None

        self.mu = None
        self.epsilon = None
        self.pressure = None

    def get_cs2(self):
        self.cs2 = 1 / (np.exp(-self.phi) + 1)
        return self.cs2

    def get_mu(self):
        """
        Return chemical potential mu in MeV
        """
        integrand = self.cs2 / self.n
        integral = cumsimp(y=integrand, x=self.n, initial=0)

        self.mu = self.mu_0 * np.exp(integral) 

        return self.mu

    def get_epsilon(self):
        """
        Return energy density epsilon in MeV * fm⁻³
        """
        self.epsilon = self.epsi_0 + cumsimp(y=self.mu, x=self.n, initial=0)
        return self.epsilon

    def get_pressure(self):
        """
        Return pressure in MeV * fm⁻³
        """
        integrand = self.cs2 * self.mu
        integral = cumsimp(y=integrand, x=self.n, initial=0)
        self.pressure = self.p_0 + integral
        return self.pressure

    def get_all(self):
        """
        Get all properties in a dictionary at once
        
        Returns
        -------
        dict
            Containing cs2, mu (MeV), epsilon (MeV fm⁻³), and pressure (MeV fm⁻³)
        """
        if self.phi is not None:
            self.get_cs2()
        self.get_mu()
        self.get_epsilon()
        self.get_pressure()

        # return all computed values as a dictionary
        return {
            "cs2": self.cs2,
            "mu": self.mu,
            "epsilon": self.epsilon,
            "pressure": self.pressure,
        }
