import numpy as np

from constants import ns, crust_end, get_phi
from prepare_ceft import smooth_cs2, get_old_ini_vacuum
import eos
import kernels
import gaussianprocess

from pathlib import Path

base_dir = Path(__file__).resolve().parent


def _load_ceft():
    ####### loading the new ceft band ############
    ceft_95 = np.loadtxt(
        base_dir / "EOS/chiEFT/chiEFT_band_95_percent_credibility.txt"
    ).T
    (n, cs2_l, cs2_u) = ceft_95

    n_ceft = n[crust_end:]  # fm^-3
    cs2_l_ceft = cs2_l[crust_end:]
    cs2_u_ceft = cs2_u[crust_end:]

    ###### using GP to interpolate and make datapoints equidistant ######
    n_uniform = np.linspace(n_ceft[0], n_ceft[-1], 100)
    kern = kernels.Kernel("SE", sigma=0.01, l=0.5)
    gp_l = gaussianprocess.GP(kern)
    gp_l.fit(n_ceft, n_uniform, cs2_l_ceft, stabilise=True)
    cs2_l_uniform = gp_l.posterior(sampling=True)

    gp_u = gaussianprocess.GP(kern)
    gp_u.fit(n_ceft, n_uniform, cs2_u_ceft, stabilise=True)
    cs2_u_uniform = gp_u.posterior(sampling=True)

    n_ceft = n_uniform
    cs2_l_ceft = cs2_l_uniform.flatten()
    cs2_u_ceft = cs2_u_uniform.flatten()

    return n_ceft / ns, cs2_l_ceft, cs2_u_ceft


def _load_crust():
    ####### loading the new crust ############
    crust = np.loadtxt(base_dir / "EOS/chiEFT/crust.dat").T
    n_crust = crust[0]
    e_crust = crust[1]
    p_crust = crust[2]
    cs2_crust = crust[3]

    return n_crust / ns, e_crust, p_crust, cs2_crust


def get_ceft_cs2():
    """
    Returns:
    n_ceft in nsat,
    cs2_ceft_avg,
    cs2_ceft_lower,
    cs2_ceft_upper,
    """

    (n_ceft, cs2_l_ceft, cs2_u_ceft) = _load_ceft()

    cs2_avg_ceft = (cs2_l_ceft + cs2_u_ceft) / 2

    return n_ceft, cs2_avg_ceft, cs2_l_ceft, cs2_u_ceft


def get_ceft_phi():
    """
    Returns:
    n_ceft in nsat,
    phi_ceft_avg,
    phi_ceft_lower,
    phi_ceft_upper,
    """
    (n_ceft, cs2_ceft_avg, cs2_ceft_lower, cs2_ceft_upper) = get_ceft_cs2()

    # phi
    phi_ceft_lower = get_phi(cs2_ceft_lower)
    phi_ceft_upper = get_phi(cs2_ceft_upper)
    phi_ceft_avg = get_phi(cs2_ceft_avg)

    return n_ceft, phi_ceft_avg, phi_ceft_lower, phi_ceft_upper


def get_ceft_ini():
    """
    Output
    ------
    - e_ini
    - p_ini
    - n_ini in fm^-3
    - mu_ini
    """

    ####### crust eos #######
    (n_crust, e_crust, p_crust, _) = _load_crust()

    e_ini = e_crust[-1]
    p_ini = p_crust[-1]
    n_ini = n_crust[-1] * ns  # in fm^3
    mu_ini = (p_ini + e_ini) / n_ini

    return e_ini, p_ini, n_ini, mu_ini


def get_crust(value: str):
    """
    Retrieve various physical properties of the neutron star crust, such as sound speed squared (cs2), chemical potential (mu),
    pressure, or energy density, depending on the input argument.

    Note that output is the same array 3 times as avg, lower and upper bound to support legacy code. New crust is a single curve, not a band.

    Input
    ----------
    value : str
        Specifies which property to compute and return. Must be one of the following:
        - "cs2": Returns the squared speed of sound (dp/de) in the crust.
        - "mu": Returns the chemical potential (mu = (epsilon + pressure) / density) in the crust, in (MeV)
        - "pressure": Returns the pressure in the crust, in (MeV/fm^3)
        - "epsilon": Returns the energy density in the crust, in (MeV/fm^3)

    Output
    -------
    tuple of numpy.ndarray
        All outputs are tuple of four numpy arrays:
        - n_crust_scaled : numpy.ndarray
            Baryon density in the crust, in saturation density (n/n_sat, where n_sat=0.16 fm^-3).
        - avg : numpy.ndarray
            The average value (upper + lower)/2 of the requested property at each density.
        - lower : numpy.ndarray
            The lower bound of the requested property at each density.
        - upper : numpy.ndarray
            The upper bound of the requested property at each density.
    """

    (n_crust, e_crust, p_crust, cs2_crust) = _load_crust()

    ######### crust eos ##############
    if value == "cs2":
        return n_crust, cs2_crust, cs2_crust, cs2_crust

    if value == "mu":
        mu_crust = (e_crust + p_crust) / (n_crust * ns)
        return n_crust, mu_crust, mu_crust, mu_crust

    if value == "pressure":
        return n_crust, p_crust, p_crust, p_crust

    if value == "epsilon":
        return n_crust, e_crust, e_crust, e_crust


def get_ceft(value: str):
    """
    Retrieve various physical properties of the neutron star in 0.5-2nsat, such as chemical potential (mu),
    pressure, or energy density, depending on the input argument.

    Input
    ----------
    value : str
        Specifies which property to compute and return. Must be one of the following:
        - "mu": Returns the chemical potential (mu = (epsilon + pressure) / density) in the crust, in (MeV)
        - "pressure": Returns the pressure in the crust, in (MeV/fm^3)
        - "epsilon": Returns the energy density in the crust, in (MeV/fm^3)

    Output
    -------
    tuple of numpy.ndarray
        All outputs are tuple of four numpy arrays:
        - n_crust_scaled : numpy.ndarray
            Baryon density in the crust, in saturation density (n/n_sat, where n_sat=0.16 fm^-3).
        - avg : numpy.ndarray
            The average value (upper + lower)/2 of the requested property at each density.
        - lower : numpy.ndarray
            The lower bound of the requested property at each density.
        - upper : numpy.ndarray
            The upper bound of the requested property at each density.
    """

    (n_ceft, _, cs2_l_ceft, cs2_u_ceft) = get_ceft_cs2()

    (e_ini_ceft, p_ini_ceft, _, mu_ini_ceft) = get_ceft_ini()

    eos_l_ceft = eos.EosProperties(
        n_ceft,
        phi=None,
        epsi_0=e_ini_ceft,
        p_0=p_ini_ceft,
        mu_0=mu_ini_ceft,
        cs2=cs2_l_ceft,
    )
    eos_u_ceft = eos.EosProperties(
        n_ceft,
        phi=None,
        epsi_0=e_ini_ceft,
        p_0=p_ini_ceft,
        mu_0=mu_ini_ceft,
        cs2=cs2_u_ceft,
    )

    properties_u_ceft = eos_u_ceft.get_all()
    properties_l_ceft = eos_l_ceft.get_all()

    if value == "mu":
        mu_ceft_lower = properties_l_ceft["mu"]  # MeV
        mu_ceft_upper = properties_u_ceft["mu"]  # MeV
        mu_ceft_avg = (mu_ceft_upper + mu_ceft_lower) / 2
        return n_ceft, mu_ceft_avg, mu_ceft_lower, mu_ceft_upper

    if value == "pressure":
        p_ceft_lower = properties_l_ceft["pressure"]
        p_ceft_upper = properties_u_ceft["pressure"]
        p_ceft_avg = (p_ceft_upper + p_ceft_lower) / 2
        return n_ceft, p_ceft_avg, p_ceft_lower, p_ceft_upper

    if value == "epsilon":
        e_ceft_lower = properties_l_ceft["epsilon"]
        e_ceft_upper = properties_u_ceft["epsilon"]
        e_ceft_avg = (e_ceft_upper + e_ceft_lower) / 2
        return n_ceft, e_ceft_avg, e_ceft_lower, e_ceft_upper


def get_n_test(n_end, numpoints):
    """
    Input:
    n_end: in nsat, corrresponding to sample's termination point,
    numpoints: number of datapoints,

    Returns:
    n: in nsat, test array including crust number density and gpr test number density
    """

    (n_ceft, _, _) = _load_ceft()

    (n_crust, _, _, _) = _load_crust()

    n_test = np.linspace(n_ceft[0], n_end, numpoints)
    n = np.concatenate((n_crust[:-1], n_test))
    return n


###########################
#     OLD CEFT BAND       #
###########################


def _load_ceft_old():
    ceft_lower = np.loadtxt(base_dir / "EOS/ceft/eos_ceft_lower.dat")
    (n_ceft_lower, p_ceft_lower, e_ceft_lower) = ceft_lower.T

    ceft_upper = np.loadtxt(base_dir / "EOS/ceft/eos_ceft_upper.dat")
    (_, p_ceft_upper, e_ceft_upper) = ceft_upper.T

    return n_ceft_lower, p_ceft_lower, e_ceft_lower, p_ceft_upper, e_ceft_upper


def get_ceft_cs2_old():
    """
    Returns:
    n_ceft in nsat,
    cs2_ceft_avg,
    cs2_ceft_lower,
    cs2_ceft_upper,
    """
    (n_ceft_lower, p_ceft_lower, e_ceft_lower, p_ceft_upper, e_ceft_upper) = (
        _load_ceft_old()
    )

    n_ceft = n_ceft_lower  # fm^-3

    cs2_lower = np.gradient(p_ceft_lower, e_ceft_lower)  # dp/de
    cs2_upper = np.gradient(p_ceft_upper, e_ceft_upper)

    e_ceft = (e_ceft_lower + e_ceft_upper) / 2  # MeVfm^-3
    p_ceft = (p_ceft_lower + p_ceft_upper) / 2  # MeVfm^-3

    # seperating ceft proper
    p_ceft_upper = p_ceft_upper[crust_end:]
    p_ceft_lower = p_ceft_lower[crust_end:]

    e_ceft_upper = e_ceft_upper[crust_end:]
    e_ceft_lower = e_ceft_lower[crust_end:]

    n_ceft = n_ceft[crust_end:]
    e_ceft = e_ceft[crust_end:]
    p_ceft = p_ceft[crust_end:]

    # sound speed
    cs2_ceft_lower = cs2_lower[crust_end:]
    cs2_ceft_lower = smooth_cs2(n_ceft, cs2_ceft_lower, 6, 34, 101, 134)
    cs2_ceft_upper = cs2_upper[crust_end:]
    cs2_ceft_upper = smooth_cs2(n_ceft, cs2_ceft_upper, 35, 85, 95, 140)
    cs2_ceft_avg = (cs2_ceft_upper + cs2_ceft_lower) / 2

    return n_ceft / ns, cs2_ceft_avg, cs2_ceft_lower, cs2_ceft_upper


def get_ceft_phi_old():
    """
    Returns:
    n_ceft in nsat,
    phi_ceft_avg,
    phi_ceft_lower,
    phi_ceft_upper,
    """
    (n_ceft, cs2_ceft_avg, cs2_ceft_lower, cs2_ceft_upper) = get_ceft_cs2_old()

    # phi
    phi_ceft_lower = get_phi(cs2_ceft_lower)
    phi_ceft_upper = get_phi(cs2_ceft_upper)
    phi_ceft_avg = get_phi(cs2_ceft_avg)

    return n_ceft, phi_ceft_avg, phi_ceft_lower, phi_ceft_upper


def get_crust_old(value: str):
    """
    Retrieve various physical properties of the neutron star crust, such as sound speed squared (cs2), chemical potential (mu),
    pressure, or energy density, depending on the input argument.

    Input
    ----------
    value : str
        Specifies which property to compute and return. Must be one of the following:
        - "cs2": Returns the squared speed of sound (dp/de) in the crust.
        - "mu": Returns the chemical potential (mu = (epsilon + pressure) / density) in the crust, in (MeV)
        - "pressure": Returns the pressure in the crust, in (MeV/fm^3)
        - "epsilon": Returns the energy density in the crust, in (MeV/fm^3)

    Output
    -------
    tuple of numpy.ndarray
        All outputs are tuple of four numpy arrays:
        - n_crust_scaled : numpy.ndarray
            Baryon density in the crust, in saturation density (n/n_sat, where n_sat=0.16 fm^-3).
        - avg : numpy.ndarray
            The average value (upper + lower)/2 of the requested property at each density.
        - lower : numpy.ndarray
            The lower bound of the requested property at each density.
        - upper : numpy.ndarray
            The upper bound of the requested property at each density.
    """

    (n_crust, p_crust_lower, e_crust_lower, p_crust_upper, e_crust_upper) = (
        _load_ceft_old()
    )

    n_crust = n_crust[: crust_end + 1]

    p_crust_upper = p_crust_upper[: crust_end + 1]
    p_crust_lower = p_crust_lower[: crust_end + 1]

    e_crust_upper = e_crust_upper[: crust_end + 1]
    e_crust_lower = e_crust_lower[: crust_end + 1]

    if value == "cs2":
        cs2_crust_lower = np.gradient(p_crust_lower, e_crust_lower)  # dp/de
        cs2_crust_upper = np.gradient(p_crust_upper, e_crust_upper)
        cs2_crust_avg = (cs2_crust_upper + cs2_crust_lower) / 2
        return n_crust / 0.16, cs2_crust_avg, cs2_crust_lower, cs2_crust_upper

    if value == "mu":
        mu_crust_lower = (e_crust_lower + p_crust_lower) / n_crust  # MeV
        mu_crust_upper = (e_crust_upper + p_crust_upper) / n_crust  # MeV
        mu_crust_avg = (mu_crust_upper + mu_crust_lower) / 2
        return n_crust / 0.16, mu_crust_avg, mu_crust_lower, mu_crust_upper

    if value == "pressure":
        p_crust_avg = (p_crust_upper + p_crust_lower) / 2
        return n_crust / 0.16, p_crust_avg, p_crust_lower, p_crust_upper

    if value == "epsilon":
        e_crust_avg = (e_crust_upper + e_crust_lower) / 2
        return n_crust / 0.16, e_crust_avg, e_crust_lower, e_crust_upper


def get_ceft_old(value: str):
    """
    Retrieve various physical properties of the neutron star in 0.5-2nsat, such as chemical potential (mu),
    pressure, or energy density, depending on the input argument.

    Input
    ----------
    value : str
        Specifies which property to compute and return. Must be one of the following:
        - "mu": Returns the chemical potential (mu = (epsilon + pressure) / density) in the crust, in (MeV)
        - "pressure": Returns the pressure in the crust, in (MeV/fm^3)
        - "epsilon": Returns the energy density in the crust, in (MeV/fm^3)

    Output
    -------
    tuple of numpy.ndarray
        All outputs are tuple of four numpy arrays:
        - n_crust_scaled : numpy.ndarray
            Baryon density in the crust, in saturation density (n/n_sat, where n_sat=0.16 fm^-3).
        - avg : numpy.ndarray
            The average value (upper + lower)/2 of the requested property at each density.
        - lower : numpy.ndarray
            The lower bound of the requested property at each density.
        - upper : numpy.ndarray
            The upper bound of the requested property at each density.
    """

    (n_ceft, p_ceft_lower, e_ceft_lower, p_ceft_upper, e_ceft_upper) = _load_ceft_old()

    n_ceft = n_ceft[crust_end:]

    p_ceft_upper = p_ceft_upper[crust_end:]
    p_ceft_lower = p_ceft_lower[crust_end:]

    e_ceft_upper = e_ceft_upper[crust_end:]
    e_ceft_lower = e_ceft_lower[crust_end:]

    if value == "mu":
        mu_ceft_lower = (e_ceft_lower + p_ceft_lower) / n_ceft  # MeV
        mu_ceft_upper = (e_ceft_upper + p_ceft_upper) / n_ceft  # MeV
        mu_ceft_avg = (mu_ceft_upper + mu_ceft_lower) / 2
        return n_ceft / 0.16, mu_ceft_avg, mu_ceft_lower, mu_ceft_upper

    if value == "pressure":
        p_ceft_avg = (p_ceft_upper + p_ceft_lower) / 2
        return n_ceft / 0.16, p_ceft_avg, p_ceft_lower, p_ceft_upper

    if value == "epsilon":
        e_ceft_avg = (e_ceft_upper + e_ceft_lower) / 2
        return n_ceft / 0.16, e_ceft_avg, e_ceft_lower, e_ceft_upper


def get_n_test_old(n_end, numpoints):
    """
    Input:
    n_end: in nsat, corrresponding to sample's termination point,
    numpoints: number of datapoints,

    Returns:
    n: in nsat, test array including crust number density and gpr test number density
    """

    ceft_lower = np.loadtxt(base_dir / "EOS/ceft/eos_ceft_lower.dat")
    (n_ceft_lower, _, _) = ceft_lower.T

    n_ceft = n_ceft_lower  # fm^-3
    n_crust = n_ceft[:crust_end] / ns  # nsat

    n_test = np.linspace(n_ceft[crust_end] / ns, n_end, numpoints)

    n = np.concatenate((n_crust, n_test))

    return n
