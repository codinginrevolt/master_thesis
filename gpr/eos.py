import numpy as np
from scipy.integrate import cumulative_simpson as cumsimp


class EosProperties:
    """
    Example usage:
    given initial values of epsilon, p, and mu, and n:
    n = np.array([...])  # Example input for n | must be in nsat
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
        self.n = n * 0.16 # fm^-3
        self.phi = phi.flatten()
        self.cs2 = None
        self.mu = None
        self.epsilon = None
        self.pressure = None

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
        integral = cumsimp(y=integrand, x=self.n, initial=0)

        self.mu = self.mu_0 * np.exp(integral) #self.mu_0 * np.exp(integral)


        return self.mu

    def get_epsilon(self):
        """eq 9
        MeV * fm^-3
        """
        self.epsilon = self.epsi_0 + cumsimp(y=self.mu, x=self.n, initial=0)
        return self.epsilon

    def get_pressure(self):
        """Hauke's eq"""
        integrand = self.cs2 * self.mu
        integral = cumsimp(y=integrand, x=self.n, initial=0)
        self.pressure = self.p_0 + integral
        return self.pressure

    def get_all(self):
        """
        get all properties in a dictionary at once
        """
        self.get_cs2()
        self.get_mu()
        self.get_epsilon()
        self.get_pressure()

        # Return all computed values as a dictionary for easy access
        return {
            "cs2": self.cs2,
            "mu": self.mu,
            "epsilon": self.epsilon,
            "pressure": self.pressure
        }