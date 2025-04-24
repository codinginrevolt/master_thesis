import numpy as np
from scipy.stats import loguniform, norm
from pqcd.pQCD import pQCD
import kernels
import gaussianprocess

ns = 0.16 # fm^-3 nsat
def get_hype_samples():
    """
    Hyperparams for GP mean and kernel, as well as pqcd renormalisation scale
    """

    rng = np.random.default_rng()
    nu = rng.normal(1.25, 0.2)
    l = rng.normal(1.0, 0.25) # note that std here is different than the ones in the qcd papers

    while l<0:
        l = rng.normal(0.2, 0.2) # reject negative correlation length, does it matter if l**2 in kernel?


    cs2_hat = rng.normal(0.5,0.25)
    while cs2_hat<0 or cs2_hat>1: # sound speed shouldnt be negative or more than c
        cs2_hat = rng.normal(0.5,0.25)        

    X = loguniform.rvs(0.5, 2)


    return cs2_hat, nu, l, X

def get_pqcd(X: float, mu_low: float = 2.2, mu_high:float = 2.8, size: int = 100):
    """
    Retrieve pQCD calculations of number density (in n_sat) and sound speed squared using an X value (renormalisation scale)
    Default mu from 2.2 GeV to 3 GeV
    Returns n_pqcd in nsat units, cs2 unitless
    """

    mu_grid = np.linspace(mu_low, mu_high, size)
        
    pQCD_temp = pQCD(X)
    n_pqcd  = np.vectorize(pQCD_temp.number_density)(mu_grid) /ns # in nsat
    cs2_pqcd = np.vectorize(pQCD_temp.speed2)(mu_grid)
    
    return n_pqcd, cs2_pqcd

def get_phi(cs2):
    return -np.log(1/cs2 - 1)

def CI_to_sigma(width, CI):
    """
    TODO: enable the same from any given CI to a sigma value.
    CI: confidence interval percentage (e.g., 75 for 75%)
    """
    z = norm.ppf(1 - (1 - 75/100) / 2)
    sig = width/(2*z)
    return(sig)

def check_pqcd_connection(X_hat, e_end, p_end, n_end):
    pqcd_temp = pQCD(X_hat)
    weight = (int(pqcd_temp.constraints(e0=e_end/1000, p0=p_end/1000, n0=n_end*ns)))
    boolean = weight == 1  
    return boolean

def generate_sample(n_ceft, cs2_ceft_avg, phi_ceft_sigma, n_crust, cs2_crust, x_test_end = 10, mu_low = 2.2, mu_high = 2.8, point_nums=200):
    """
    input n must be in nsat, if used in conjunction with make_condition_eos() that is automatically the case
    out n in nsat
    """

    cs2_hat, nu_hat, l_hat, X_hat = get_hype_samples()

    kernel = kernels.Kernel('SE', sigma=nu_hat, l=l_hat)


    n_pqcd, cs2_pqcd = get_pqcd(X_hat, mu_low, mu_high, size=100) # nsat, unitless

    x_train =  np.concatenate((n_crust[-10:-1],n_ceft, n_pqcd)) #nsat
    cs2_train =  np.concatenate((cs2_crust[-10:-1],cs2_ceft_avg, cs2_pqcd)) # crust excluding last element because it is already in ceft

    phi_pqcd_sigma = np.zeros_like(cs2_pqcd)
    phi_crust_sigma = np.zeros_like(cs2_crust[-10:-1])

    phi_sigma_train = np.concatenate((phi_crust_sigma, phi_ceft_sigma, phi_pqcd_sigma))
    phi_train = get_phi(cs2_train)
    train_noise = phi_sigma_train**2

    x_test = np.linspace(n_ceft[0], x_test_end, point_nums) # number density in nsat, starting val is ending val of n crust, ending val is default 10 nsat


    gp = gaussianprocess.GP(kernel, get_phi(cs2_hat))
    gp.fit(x_train, x_test, phi_train, var_f = train_noise, stabilise=True)

    phi_test = gp.posterior(sampling=True)

    phi_test = phi_test.flatten()
    phi_test[0] = get_phi(cs2_crust[-1]) # minor difference in gpr result and actual crust ending so replacing gpr val

    return phi_test, x_test, X_hat

def smooth_cs2(n_ceft, cs2_ceft, a, b, c, d):
    
    end = 200
    cs2_ceft_train = np.concatenate((cs2_ceft[0:a], cs2_ceft[b:c], cs2_ceft[d:end]))
    n_ceft_train = np.concatenate((n_ceft[0:a], n_ceft[b:c], n_ceft[d:end]))
    n_ceft_test = np.concatenate((n_ceft[a:b], n_ceft[c:d]))

    kern = kernels.Kernel('SE', sigma = 0.01, l = 0.01)
    gp = gaussianprocess.GP(kern)
    gp.fit(n_ceft_train, n_ceft_test, cs2_ceft_train, stabilise=True)

    cs2_ceft_test, _sig = gp.posterior()
    cs2_ceft_test = cs2_ceft_test.flatten()

    new_cs2_ceft = cs2_ceft.copy()
    new_cs2_ceft[a:b] = cs2_ceft_test[:b-a]
    new_cs2_ceft[c:d] = cs2_ceft_test[b-a:]

    return new_cs2_ceft

def make_conditioning_eos():
    "returns n in nsat"

    ceft_lower= np.loadtxt('EOS/ceft/eos_ceft_lower.dat')
    n_ceft_lower, p_ceft_lower, e_ceft_lower = ceft_lower.T

    ceft_upper= np.loadtxt('EOS/ceft/eos_ceft_upper.dat')
    _, p_ceft_upper, e_ceft_upper = ceft_upper.T


    n_ceft = n_ceft_lower #fm^-3

    cs2_lower = np.gradient(p_ceft_lower, e_ceft_lower) #dp/de
    cs2_lower = smooth_cs2(n_ceft, cs2_lower, 6,34,101,134) #getting rid of that dips at beginning of ceft

    cs2_upper = np.gradient(p_ceft_upper, e_ceft_upper)
    cs2_upper = smooth_cs2(n_ceft, cs2_upper, 35,85,95,140)

    cs2_avg = (cs2_upper+cs2_lower)/2

    # average CEFT EOS
    e_ceft = (e_ceft_lower+e_ceft_upper)/2 # MeVfm^-3 
    p_ceft = (p_ceft_lower+p_ceft_upper)/2 # MeVfm^-3

    # seperating crust: n_crust = n_ceft[:428] last element overlap
    crust_end = 428
    n_crust = n_ceft[:crust_end+1]
    e_crust = e_ceft[:crust_end+1]
    p_crust = p_ceft[:crust_end+1]
    cs2_crust = cs2_avg[:crust_end+1]

    # seperating ceft proper
    p_ceft_upper = p_ceft_upper[crust_end:]
    p_ceft_lower = p_ceft_lower[crust_end:]

    e_ceft_upper = e_ceft_upper[crust_end:]
    e_ceft_lower = e_ceft_lower[crust_end:]

    n_ceft = n_ceft[crust_end:]
    e_ceft = e_ceft[crust_end:]
    p_ceft = p_ceft[crust_end:]

    # chemical potential
    e_ini = e_ceft[0] 
    p_ini = p_ceft[0] 
    n_ini = n_ceft[0]
    mu_ini = (e_ini + p_ini) / n_ini # MeV

    # sound speed
    cs2_ceft_lower = cs2_lower[crust_end:]
    cs2_ceft_upper = cs2_upper[crust_end:] 
    cs2_ceft_avg = cs2_avg[crust_end:]  

    # phi
    phi_ceft_lower = get_phi(cs2_ceft_lower)
    phi_ceft_upper = get_phi(cs2_ceft_upper)
    phi_ceft_width = phi_ceft_upper-phi_ceft_lower
    phi_ceft_sigma = CI_to_sigma(phi_ceft_width, 75)

    return n_ceft/ns, cs2_ceft_avg, phi_ceft_sigma, e_ini, p_ini, mu_ini, n_crust/ns, e_crust, p_crust, cs2_crust

def append_eos_results(results_e, results_p, results_cs2, epsilon, pressure, cs2, e_crust, p_crust, cs2_crust):
    e_array = np.concatenate((e_crust[:-1], epsilon)) # last element of crust is first element of gpr result
    p_array = np.concatenate((p_crust[:-1], pressure))
    cs2_array = np.concatenate((cs2_crust[:-1], cs2))
    
    results_e.append(e_array)
    results_p.append(p_array)
    results_cs2.append(cs2_array)

def append_n_phi(results_n, results_phi, n_array, phi_array, n_crust, cs2_crust):
    n_array = np.concatenate((n_crust[:-1], n_array))
    phi_array = np.concatenate((get_phi(cs2_crust[:-1]), phi_array))

    results_n.append(n_array)
    results_phi.append(phi_array)



