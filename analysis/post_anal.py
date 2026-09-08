import numpy as np
import pandas as pd
import corner
import matplotlib.pyplot as plt
import json
import seaborn as sb

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_hex
from matplotlib.colors import to_rgba
from scipy.spatial.distance import jensenshannon
from typing import List

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../gpr')))
import anal_helpers as anal


def load_eos_data(folder_name):
    eos_file = f"/home/sam/thesis/code/results/eos_samples/{folder_name}/{folder_name}_eos.npy"

    try:
        eos = np.load(eos_file)

    except FileNotFoundError:
        try:
            mr_file = f"/home/sam/thesis/code/results/tov_res/{folder_name}_tidal.npy"
            mr = np.load(mr_file)
            print(f"EoS file not found: {eos_file}")
            return None, mr
        except FileNotFoundError:
            print(f"EoS or MRL file not found: {mr_file}")
            return None, None

    mr_file = f"/home/sam/thesis/code/results/tov_res/{folder_name}_tidal.npy"
    try:
        mr = np.load(mr_file)
    except FileNotFoundError:
        print(f"MRL file not found: {mr_file}")
        return eos, None

    return eos, mr

def load_results(folder_name):
    eos, mr = load_eos_data(folder_name)

    filepath = f"/home/sam/thesis/code/results/nmma/{folder_name}/GW170817_result.json"

    try:
        
        with open(filepath, "r") as f:
            temp_json = json.load(f)

        content = temp_json.get("posterior", {}).get("content")

        df = pd.DataFrame(content)

        df["EOS"] = np.int64(np.floor(df["EOS"]))
        df["lambda_tilde"] = lambda_tilde(df["mass_1_source"], df["mass_2_source"], df["lambda_1"], df["lambda_2"])

        _, lam_14 = get_14(mr)
        df["lambda_14"] = lam_14[df["EOS"]]
        
        p_3ns_arr = p_3ns(eos)
        df["p_3ns"] = p_3ns_arr[df["EOS"]]

        try:
            PSR07, PSR16 = load_psr(folder_name)
            likelihood_columns(df, PSR07, PSR16)

        except FileNotFoundError:
            print("PSR data not found, skipping likelihood columns.")
        
        return df
        
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    
def load_psr(folder_name):
    PSR07 = np.loadtxt(f"/home/sam/thesis/code/results/pulsars/{folder_name}/eos_likelihood_PSRJ0740")
    PSR16 = np.loadtxt(f"/home/sam/thesis/code/results/pulsars/{folder_name}/eos_likelihood_PSRJ1614")

    return PSR07, PSR16


def likelihood_columns(dataframe, PSR07, PSR16):
    PSR07_posterior = PSR07[dataframe["EOS"] - 1]
    PSR16_posterior = PSR16[dataframe["EOS"] - 1]

    dataframe["PSRJ0740_likelihood"] = PSR07_posterior
    dataframe["PSRJ1614_likelihood"] = PSR16_posterior

    dataframe["GW_likelihood"] = dataframe["EOS"].map(dataframe["EOS"].value_counts() / len(dataframe))
    dataframe["Likelihood"] = dataframe["GW_likelihood"] * dataframe["PSRJ0740_likelihood"] * dataframe["PSRJ1614_likelihood"]


def get_tov(r, m):
    r_tov = []
    m_tov = []
    for i in range(len(r)):
        m_tov.append(np.nanmax(m[i])) # unstable branch has NaN mass
        r_tov.append(r[i][np.nanargmax(m[i])])
    
    return np.asarray(r_tov), np.asarray(m_tov)
    
def r_14(r, m, m_target=1.4):
    """
    gets radius at 1.4 solar masses for each MR curve in r and m arrays
    """
    r_14 = []

    for i in range(len(r)):
        m_val = m[i][~np.isnan(m[i])]
        r_val = r[i][~np.isnan(m[i])]
        r_14.append(get_obs_14(r_val, m_val, m_target=m_target))
    r_14 = np.asarray(r_14)

    return r_14

def lam_14(lam, m, m_target=1.4):
    """
    gets tidal deformability at 1.4 solar masses for each MR curve in lam and m arrays
    """
    lam_14 = []

    for i in range(len(lam)):
        m_val = m[i][~np.isnan(m[i])]
        lam_val = lam[i][~np.isnan(m[i])]
        lam_14.append(get_obs_14(lam_val, m_val, m_target=m_target))
    lam_14 = np.asarray(lam_14)

    return lam_14

def lam_tilde(m1, m2, mrl):
    """
    Gets the combined tidal deformability, given two masses and an MRL array
    """

    masses = mrl[1]
    lambdas = mrl[2]

    val_masses = masses[~np.isnan(masses)]
    lambdas = lambdas[~np.isnan(masses)]
    
    if np.min(val_masses) > min(m1, m2) or np.max(val_masses) < max(m1, m2):
        return np.nan

    lam1 = np.interp(m1, val_masses, lambdas)
    lam2 = np.interp(m2, val_masses, lambdas)

    const = 16/13
    term1 = (m1 + 12*m2) * m1**4 * lam1
    term2 = (m2 + 12*m1) * m2**4 * lam2
    denom = (m1 + m2)**5
    lambda_tilde = const * (term1 + term2) / denom
    return lambda_tilde

def get_obs_14(r, m, m_target=1.4):
    """r can be radius or tidal deformability, the observable at 1.4 solar mass
    gets scalars for singular MRT curve
    """

    if np.nanmax(m) < m_target:
        r_14 = np.nan
    else:
        r_interp = np.interp(m_target, m, r)
        r_14 = r_interp

    return r_14

def get_14(mrl):
    r = mrl[0]
    m = mrl[1]
    lam = mrl[2]

    r14 = r_14(r, m)
    lam14 = lam_14(lam, m)
    return r14, lam14

def component_masses(M_c, q):
    m1 = []
    m2 = []
    for i in range(len(M_c)):
        m2_i = M_c[i] * ((q[i]+1)**(1/5)) / q[i]**(3/5)
        m1_i = q[i] * m2_i
        m1.append(m1_i)
        m2.append(m2_i)
 
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    return m1, m2

def component_tidal_deformability(lam, m, m1, m2):
    lam1 = []
    lam2 = []
    for i in range(len(m1)):
        valid_mask = ~np.isnan(m[i]) # unstable branch has NaN mass
        m_valid = m[i][valid_mask]
        lam_valid = lam[i][valid_mask]
        
        lam1_interp = np.interp(m1[i], m_valid, lam_valid)
        lam1.append(lam1_interp)
        lam2_interp = np.interp(m2[i], m_valid, lam_valid)
        lam2.append(lam2_interp)
            
    return lam1, lam2 


def mrt_bounds(mr, interested_axis="radius", ci=95, weights=None):
    """
    Given an MRT array, compute the 95% CI radius (or tidal deformability) 
    for each mass across all EOS models.
    
    Parameters:
    - mr: The mass-radius (or mass-tidal deformability) array of shape (4, num_eos, num_points)
    - interested_axis: A string, either "radius" or "lambda"
    - ci: The confidence interval percentage (e.g., 95)
    - weights: An array of weights for each EOS model
    """
    masses = mr[1]
    masses = masses[~np.isnan(masses)]
    max_mass = masses.max()
    min_mass = masses.min()
    masses_arr = np.linspace(min_mass, max_mass, 200)

    if interested_axis == "radius":
        axis_index = 0
    elif interested_axis == "lambda":
        axis_index = 2
    else:
        raise ValueError("interested_axis must be 'radius' or 'lambda'")

    num_eos = mr.shape[1]
    all_interpolated_vals = np.zeros((num_eos, len(masses_arr)))

    for j in range(num_eos):
        rads = mr[axis_index, j, :]                
        ms   = mr[1, j, :]                

        mask = ~(np.isnan(rads))
        rads = rads[mask]
        ms  = ms[mask]
        r_vals = np.interp(masses_arr, ms, rads, left=np.nan, right=np.nan)
        all_interpolated_vals[j, :] = r_vals

    lower_p = ((100 - ci) / 2)/100
    upper_p = (100 - lower_p)/100

    r_lows = np.full(len(masses_arr), np.nan)
    r_highs = np.full(len(masses_arr), np.nan)

    for k in range(len(masses_arr)):
        vals = all_interpolated_vals[:, k]
        mask = np.isfinite(vals)
        sample_vals = vals[mask]
        sample_weights = weights[mask] if weights is not None else None

        if sample_vals.size == 0:
            continue
        if sample_vals.size == 1:
            r_lows[k] = sample_vals[0]
            r_highs[k] = sample_vals[0]
            continue

        q_low, q_high = corner.quantile(sample_vals, [lower_p, upper_p], weights=sample_weights)
        r_lows[k] = q_low
        r_highs[k] = q_high
                
    #r_lows = np.nanpercentile(all_interpolated_vals, lower_p, axis=0)
    #r_highs = np.nanpercentile(all_interpolated_vals, upper_p, axis=0)

    return masses_arr, r_lows, r_highs

def p_3ns(eos: np.array, 
          n: np.array= None):
    """
    Parameters
    ----------
    eos : np.array
        Array of EoS samples, shape (N_samples, 3, N_points), where the second dimension corresponds to (energy density, pressure, sound speed squared).
    n : np.array, optional
        Number density array corresponding to the EoS samples. If not provided, defaults to a range of [0,10] n_sat with 275 points.

    Returns 
    ----------
    np.array
        pressures [MeV/fm^3] at 3 times nuclear saturation density for each EoS sample. 
    """
    if n is None:
        n = anal.get_n_test(10, 200)

    pressures = eos[1]


    p_3 = np.array([np.interp(3, n, pressure) for pressure in pressures])

    return p_3

def cs2_6ns(eos: np.array, 
          n: np.array= None, cs2_target: float=6.0) -> np.array:
    """
    Parameters
    ----------
    eos : np.array
        Array of EoS samples, shape (N_samples, 3, N_points), where the second dimension corresponds to (energy density, pressure, sound speed squared).
    n : np.array, optional
        Number density array corresponding to the EoS samples. If not provided, defaults to a range of [0,10] n_sat with 275 points.

    Returns 
    ----------
    np.array
        pressures [MeV/fm^3] at 3 times nuclear saturation density for each EoS sample. 
    """
    if n is None:
        n = anal.get_n_test(10, 200)

    cs2 = eos[2]


    cs2_6 = np.array([np.interp(cs2_target, n, cs) for cs in cs2])

    return cs2_6

def lambda_tilde(m1, m2, lam1, lam2):
    const = 16/13
    numer1 = (m1 + 12*m2) * m1**4 * lam1
    numer2 = (m2 + 12*m1) * m2**4 * lam2
    denom = (m1 + m2)**5
    lam_tilde = const * (numer1 + numer2) / denom
    return lam_tilde

def get_valid_eos(df):
    valid_eos = np.unique(df['EOS']) - 1
    
    return np.asarray(valid_eos)

def get_posterior_eos(df, eos, mr):

    valid_eos = np.array(get_valid_eos(df), dtype=int)
    eos_posterior = np.array([np.asarray(arr)[valid_eos] for arr in eos])
    mr_posterior = np.array([np.asarray(arr)[valid_eos] for arr in mr])

    return eos_posterior, mr_posterior

def get_posterior_weights(df):

    if "Likelihood" not in df.columns:
        weights = df.groupby('EOS')['GW_likelihood'].first().values

    else:
        weights = df.groupby('EOS')['Likelihood'].first().values

    return weights

def get_prior_weights(eos):
    n_sams = eos.shape[1]
    weights = np.ones(n_sams) / n_sams

    return weights

def load_inj_results(kernel_name, set):

    folder_name = f"{kernel_name}_10ns_{set}"
    eos, mr = load_eos_data(folder_name)


    filepath = f"/home/sam/thesis/code/results/nmma/injections/{folder_name}/ET_injection_{kernel_name}_10ns_{set}_result.json"
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        data2 = data.get("posterior")
        content = data2[0].get("content")

        df = pd.DataFrame(content)
        
        df["EOS"] = np.int64(np.floor(df["EOS"]))
        df["lambda_tilde"] = lambda_tilde(df["mass_1_source"], df["mass_2_source"], df["lambda_1"], df["lambda_2"])

        p_3ns_arr = p_3ns(eos)
        df["p_3ns"] = p_3ns_arr[df["EOS"]]

        _, lam_14 = get_14(mr)
        df["lambda_14"] = lam_14[df["EOS"]]

        try:
            PSR07, PSR16 = load_psr(folder_name)
            likelihood_columns(df, PSR07, PSR16)

        except FileNotFoundError:
            print("PSR data not found, skipping likelihood columns.")
            
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

def JSD(column, df1, df2, bins=40, weighted=False):

    x = df1[column].to_numpy()
    y = df2[column].to_numpy()
    x = x[x > 0]
    y = y[y > 0]

    if weighted:
        weights1 = df1["Likelihood"][df1[column] > 0].to_numpy()
        weights2 = df2["Likelihood"][df2[column] > 0].to_numpy()
        weights1 = weights1[x > 0]
        weights2 = weights2[y > 0]
        
    if not weighted:
        weights1 = None
        weights2 = None

    # common bin edges for both distributions
    edges = np.histogram_bin_edges(np.concatenate([x, y]), bins=bins)

    p, _ = np.histogram(x, bins=edges, weights=weights1)
    q, _ = np.histogram(y, bins=edges, weights=weights2)

    p = p.astype(float)
    q = q.astype(float)

    p /= p.sum()
    q /= q.sum()

    return jensenshannon(p, q, base=2) ** 2

def JSD_average(column, dfs, weighted=True, bins=40):
    """Plot the Jensen-Shannon Divergence between multiple pairs of dataframes for a given column."""
    n = len(dfs)
    jsd_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            jsd_matrix[i, j] = JSD(column, dfs[i], dfs[j], bins, weighted)
            jsd_matrix[j, i] = jsd_matrix[i, j]
            
    indices = np.triu_indices(len(jsd_matrix), k=1)
    unique_jsd_values = jsd_matrix[indices]
    avg_jsd = np.mean(unique_jsd_values)

    print(f"Average JSD: {avg_jsd}")

    return avg_jsd

def get_stats(arr, weights=None):
    mean = np.average(arr, axis=0, weights=weights)

    p16 = np.array([corner.quantile(arr[:, i], 0.16, weights=weights)
        for i in range(arr.shape[1])]).reshape(np.shape(mean))
    p84 = np.array([corner.quantile(arr[:, i], 0.84, weights=weights)
        for i in range(arr.shape[1])]).reshape(np.shape(mean))
    p025 = np.array([corner.quantile(arr[:, i], 0.025, weights=weights)
        for i in range(arr.shape[1])]).reshape(np.shape(mean))
    p975 = np.array([corner.quantile(arr[:, i], 0.975, weights=weights)
        for i in range(arr.shape[1])]).reshape(np.shape(mean))


    return mean, p16, p84, p025, p975

############## PLOTTING ########################


def plot_histogram(mrl: np.array = None,
                   eos: np.array = None, 
                   dfs: pd.DataFrame=None, 
                   param: str = "r_1.4", 
                   labels: list[str]= None, 
                   colours: list[str]= None,
                   only_posteriors: bool =True,
                   m1: float = 1.4013655921528583,
                   m2: float = 1.251350789219395,
                   bins:int= 40
                   ):
    
    """
    Plot histogram of given parameter from either MR samples, EoS samples, and/or posterior DataFrames.

    Parameters
    ----------
    mrl : np.array, optional
        Array of mass-radius-lambda samples for prior distribution, required for 'lambda_1.4' or prior for 'lambda_tilde' and 'r_1.4'.
    eos : np.array, optional
        Array of EoS samples for prior distribution, required if param is 'p_3ns'.
    dfs : pd.DataFrame or list of pd.DataFrame, optional
        DataFrame(s) containing posterior samples. Required for any posterior.
    param : str, optional
        Parameter to plot ('r_1.4', 'lambda_1.4', 'p_3ns', or any in dfs columns).
    labels : list of str, optional
        Labels for each dataset in the histogram.
    colours : list of str, optional
        Colors for each dataset in the histogram.
    only_posterior : bool, optional
        Defaults to only plotting posteriors, if False, will plot both priors and posteriors for parameters dependent on MRL or EoS.
    m1, m2 : floats, optional
        Component masses for prior lambda_tilde calculation. Required if param is 'lambda_tilde'.
    bins : int, optional
        Number of bins for the histogram.
    """

    if mrl is None and dfs is None and eos is None:
        print("No data provided for histogram.")
        return
    
    is_list = isinstance(dfs, list)
    is_mr_list = isinstance(mrl, list)
    is_eos_list = isinstance(eos, list)

    n = 1
    if is_list:
        n = len(dfs)
    elif is_mr_list:
        n = len(mrl)
    elif is_eos_list:
        n = len(eos)


    eos_dependent_param = ["lambda_1.4", "r_1.4", "p_3ns"]

    data_0 = None
    data_1 = None

    count = 0

    if labels is None:
        labels = [f"Set {i+1}" for i in range(n)]
    if colours is None:
        cmap = plt.get_cmap("tab10")
        colours = [to_hex(cmap(i % 10)) for i in range(n+1)]

    fig, ax = plt.subplots(figsize=(8,6))

   
    if not (is_mr_list or is_eos_list or is_list): # case for prior vs post, just prior, or just post
        if mrl is not None:
            if param == "r_1.4":
                r_14_prior, _ = get_14(mrl)
                data_0 = r_14_prior
            elif param == "lambda_1.4":
                _, lam_14_prior = get_14(mrl)
                data_0 = lam_14_prior
            elif param == "lambda_tilde":
                pass # not yet implemented
        if param == "p_3ns":
            if eos is None:
                print("eos argument must not be empty for p_3ns parameter.")
                return
            data_0 = p_3ns(eos)
        
        if data_0 is not None:
            count = 1
            ax.hist(data_0, bins=bins, density=True, alpha=0.5, label=labels[0], color=colours[0])

        if dfs is not None:
            if param == "r_1.4":
                data_1 = dfs["R_14"]

            elif param == "lambda_tilde":
                data_1 = dfs["lambda_tilde"]

            elif param == "lambda_1.4":
                valid_eos_index = np.array(get_valid_eos(dfs))
                valid_mrl = mrl[:, valid_eos_index,:]
                _, l_14_prior = get_14(valid_mrl)
                data_1 = l_14_prior

            elif param == "p_3ns":
                if eos is None:
                    print("eos argument must not be empty for p_3ns parameter.")
                    return
                
                valid_eos_index = np.array(get_valid_eos(dfs))
                valid_eos = eos[:, valid_eos_index,:]
                data_1 = p_3ns(valid_eos)

            else:
                if param in dfs.columns:
                    data_1 = dfs[param]
                else:
                    print(f"Parameter '{param}' not found in DataFrame columns.")
                    return

            ax.hist(data_1, bins=bins, density=True, alpha=0.5, label=labels[count], color=colours[count])

    if dfs is None and (is_mr_list or is_eos_list): # only priors, whole section is repeating the priors from the next case too, needs to be rewritten
        if param == "r_1.4" or param == "lambda_1.4": 
                if mrl is None:
                    print("mrl argument must not be empty for lambda_1.4 or r_1.4 parameter.")
                    return
                for i, (mrl_i) in enumerate(mrl):
                    if param == "r_1.4":
                        r_14_prior, _ = get_14(mrl_i)
                        data_0 = r_14_prior
                    elif param == "lambda_1.4":
                        _, lam_14_prior = get_14(mrl_i)
                        data_0 = lam_14_prior
                    elif param == "lambda_tilde":
                        pass # not yet implemented
                    ax.hist(data_0, bins=bins, density=True, alpha=0.5, label=labels[i], color=colours[i], histtype='step', linewidth=1.5) #  this code be written better

        if param == "p_3ns":
            if eos is None:
                print("eos argument must not be empty for p_3ns parameter.")
                return    
            for i, (eos_i) in enumerate(eos):
                data_0 = p_3ns(eos_i)

                ax.hist(data_0, bins=bins, density=True, alpha=0.5, label=labels[i], color=colours[i]) # the linestyle is the only element not repeated


    if is_list: # posteriors
        for i, df in enumerate(dfs[0:], start=0):
                    if param == "r_1.4":
                        data_i = df["R_14"].dropna()
                    elif param == "lambda_1.4":
                        if mrl is None:
                            print("mrl argument must not be empty for lambda_1.4 parameter.")
                            return
                        valid_eos_index = np.array(get_valid_eos(df))
                        valid_mrl = mrl[i][:, valid_eos_index,:]
                        _, l_14_post = get_14(valid_mrl)
                        data_i = l_14_post
                    elif param == "lambda_tilde":
                        data_i = df["lambda_tilde"].dropna()

                    elif param == "p_3ns":
                        if eos is None:
                            print("eos argument must not be empty for p_3ns parameter.")
                            return
                        
                        valid_eos_index = np.array(get_valid_eos(df))
                        valid_eos = eos[i][:, valid_eos_index,:]
                        data_i = p_3ns(valid_eos)
                    
                    else:
                        if param in df.columns:
                            data_i = df[param].dropna()


                    ax.hist(data_i, bins=bins, alpha=0.5, label=labels[i], color=colours[i],  density=True)

        if not only_posteriors and param in eos_dependent_param: # priors if asked for and exists
            ax_r = ax.twinx()

            if param == "r_1.4" or param == "lambda_1.4" or param == "lambda_tilde":
                if mrl is None:
                    print("mrl argument must not be empty for lambda_1.4 or r_1.4 parameter.")
                    return
                for i, (mrl_i) in enumerate(mrl):
                    if param == "r_1.4":
                        r_14_prior, _ = get_14(mrl_i)
                        data_0 = r_14_prior
                    elif param == "lambda_1.4":
                        _, lam_14_prior = get_14(mrl_i)
                        data_0 = lam_14_prior
                    elif param == "lambda_tilde":
                        pass # not yet implemented
    
                    ax_r.hist(data_0, bins=bins, alpha=0.5, label=labels[i], color=colours[i], histtype='step', density=True, fill=False) #  this code be written better

            if param == "p_3ns":
                if eos is None:
                    print("eos argument must not be empty for p_3ns parameter.")
                    return    
                for i, (eos_i) in enumerate(eos):
                    data_0 = p_3ns(eos_i)

                    ax_r.hist(data_0, bins=bins, alpha=0.5, label=labels[i], color=colours[i], histtype='step', density=True, fill=False)

    ax.set_xlabel(param)
    ax.set_ylabel("Density")
    ax.legend()


def plot_eos_comparisons(eos_list, ax1 = None, ax2=None, ax3 = None, ax4=None, mr_list = None,
                         labels=None, colours=None, n=None, plot_samples=False, plot_mean = True,
                         plot_alone=None, axis_fontsize=14, legend_fontsize=14, weights=None):
    
    """
    Plotting function to compare multiple EOS and MR sets. Plots 95% CI and mean.

    Function must include at least one of the axes arguments: "ax1 = ... ", "ax2 = ... ", "ax3 = ... ", or "ax4 = ... ". 

    Parameters
    ----------
    eos_list : list of tuples
        Each tuple contains three numpy arrays: (energy density, pressure, sound speed squared).
    ax1 : matplotlib.axes.Axes
        Axes for the pressure vs energy density plot.
    ax2 : matplotlib.axes.Axes, optiona
        Axes for the sound speed squared vs number density plot. 
    ax3 : matplotlib.axes.Axes, optional
        Axes for the mass-radius plot. If None, mass-radius plot is skipped.
    mr_list : list of tuples, optional
        Each tuple contains two numpy arrays: (radius, mass). If None, mass-radius plot is skipped.
    labels : list of str, optional
        Labels for each EOS/MR set. If None, default labels are used.
    colours : list of str, optional
        Colours for each EOS/MR set. If None, distinct colours are auto-generated.
    n : numpy.ndarray, optional
        Number density array for sound speed squared plot. If None, a default range of [0,10] n_sat is used.
    plot_samples : bool, optional
        If True, 5 random EOS samples are plotted in addition to the mean and CI.
    plot_mean : bool, optional
        If True, the mean of each EOS set is plotted. If False, only the CI is plotted.
    plot_alone : str, optional
        If provided, only the specified plot ('pressure', 'cs2', 'mr', 'tide') is generated.
    axis_fontsize : int, optional
    legend_fontsize : int, optional
    """

    if n is None:
        n = anal.get_n_test(10, 200)
    K = len(eos_list)

    if labels is None:
        labels = [f"Set {i+1}" for i in range(K)]

    if colours is None:
        # auto-generate distinct colours
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("tab10")
        colours = [cmap(i % 10) for i in range(K)]

    # For building the legend once
    if ax1 is not None:
        legend_handles, legend_labels = ax1.get_legend_handles_labels()
    if ax1 is None and ax2 is not None:
        legend_handles, legend_labels = ax2.get_legend_handles_labels()
    if ax1 is None and ax2 is None and ax3 is not None:
        legend_handles, legend_labels = ax3.get_legend_handles_labels()
    if ax1 is None and ax2 is None and ax3 is None and ax4 is not None:
        legend_handles, legend_labels = ax4.get_legend_handles_labels()


    if mr_list is None:
        mr_list = [None] * K


    # Loop over all EOS/MR inputs
    for i, (eos, mr) in enumerate(zip(eos_list, mr_list)):
        label = labels[i]
        colour = colours[i]
        face_color = to_rgba(colour, alpha=0.3)

        if weights is not None:
            w = weights[i]            
        else:
            w = None

        mean_p, p_16, p_84, p_025, p_975 = get_stats(eos[1], weights=w)
        mean_e, e_16, e_84, e_025, e_975 = get_stats(eos[0], weights=w)
        mean_cs2, cs2_16, cs2_84, cs2_025, cs2_975 = get_stats(eos[2], weights=w)

        if mr is not None:
            mean_m, r_min, r_max = mrt_bounds(mr, interested_axis = "radius", ci=95, weights=w)
            mean_m, lam_min, lam_max = mrt_bounds(mr, interested_axis = "lambda", ci=95, weights=w)

        if plot_alone == 'pressure':
            ax2 = None
            ax3 = None
            ax4 = None
        elif plot_alone == 'cs2':
            ax1 = None
            ax3 = None
            ax4 = None
        elif plot_alone == 'mr':
            ax1 = None
            ax2 = None
            ax4 = None
        elif plot_alone == 'tide':
            ax1 = None
            ax2 = None
            ax3 = None

        if ax1 is not None:
            if plot_mean:
                ax1.plot(n, mean_p, color=colour)
            ax1.fill_between(n, p_025, p_975, facecolor=face_color, edgecolor=colour, linestyle='dashed', linewidth=1.5)
  
        if ax2 is not None:
            if plot_mean:
                ax2.plot(n, mean_cs2, color=colour)
            ax2.fill_between(n, cs2_025, cs2_975, facecolor=face_color, edgecolor=colour, linestyle='dashed', linewidth=1.5)

        if (ax3 is not None) and (mr is not None):
            ax3.fill_betweenx(mean_m, r_min, r_max, facecolor=face_color, edgecolor=colour, linestyle='dashed', linewidth=1.5)

        if ax4 is not None and mr is not None:
            ax4.fill_between(mean_m, lam_min, lam_max, facecolor=face_color, edgecolor=colour, linestyle='dashed', linewidth=1.5)

        if legend_fontsize != 0:
            if plot_mean:
                legend_handles.extend([
                    Line2D([0], [0], color=colour, label=f'{label}'),
                ])
            else:
                legend_handles.extend([
                    Patch(facecolor=face_color, edgecolor=colour, label=f'{label}', linestyle='dashed'),
                ])

    if plot_samples:
        for i, (eos, mr) in enumerate(zip(eos_list, mr_list)):
            idxes = np.random.randint(0, eos[0].shape[0], size=5)
            colour = colours[i]
            for idx in idxes:
                if ax1 is not None:
                    ax1.plot(n, eos[1][idx], color=colour, alpha=0.3)
                if ax2 is not None:
                    ax2.plot(n, eos[2][idx], color=colour, alpha=0.3)
                if (ax3 is not None) and (mr is not None):
                    ax3.plot(mr[0][idx], mr[1][idx], color=colour, alpha=0.3)
                if ax4 is not None and mr is not None:
                    ax4.plot(mr[1][idx], mr[2][idx], color=colour, alpha=0.3)


    if ax1 is not None:
        ax1.set_yscale("log")
        ax1.set_xlabel(r"Number density $n$ in $n_{\mathrm{sat}}$", fontsize=axis_fontsize)
        ax1.set_ylabel(r"Pressure $p$ in MeV fm$^{-3}$", fontsize=axis_fontsize)
        ax1.set_ylim(1e-1,)
        if legend_fontsize != 0:
            ax1.legend(handles=legend_handles, loc='lower right', prop={'size': legend_fontsize})

    if ax2 is not None:
        ax2.set_xlabel(r"Number density $n$ in $n_{\mathrm{sat}}$", fontsize=axis_fontsize)
        ax2.set_ylabel(r"$c_{s}^{2}$ in $c^2$", fontsize=axis_fontsize)

        if plot_alone == 'cs2' or (ax1 is None and ax3 is None and ax4 is None) and legend_fontsize != 0:
            ax2.legend(handles=legend_handles, loc='center right', prop={'size': legend_fontsize})
        
    if ax3 is not None:
        ax3.set_xlabel(r"Radius $R$ in km", fontsize=axis_fontsize)
        ax3.set_ylabel(r"Mass $M$ in M$_\odot$", fontsize=axis_fontsize)
        if plot_alone == 'mr' or (ax1 is None and ax2 is None) and legend_fontsize != 0:
            ax3.legend(handles=legend_handles, loc='upper left', prop={'size': legend_fontsize})

    if ax4 is not None:
        ax4.set_yscale("log")
        ax4.set_xlabel(r"Mass $M$ in M$_\odot$", fontsize=axis_fontsize)
        ax4.set_ylabel(r"Tidal Deformability $\Lambda$", fontsize=axis_fontsize)
        if plot_alone == 'tide' or (ax1 is None and ax2 is None and ax3 is None) and legend_fontsize != 0:
            ax4.legend(handles=legend_handles, loc='upper right', prop={'size': legend_fontsize})

def plot_contour(df, title, axis_fontsize=14):

    params_to_plot = [
        "chirp_mass", "mass_ratio", "a_1", "a_2",
        "tilt_1", "tilt_2", "luminosity_distance",
    ]

    labels = [
        r"$\mathcal{M}$", r"$q$", r"$a_1$", r"$a_2$",
        r"$\theta_1$", r"$\theta_2$", r"$d_L$"
    ]

    figure = corner.corner(
        df[params_to_plot],
        bins=40,
        smooth=True,    
        color='salmon',
        labels=labels,
        label_kwargs=dict(fontsize=axis_fontsize),
        title_kwargs=dict(fontsize=14),
        quantiles=[0.05,0.95],
        levels=[0.68, 0.95],
        plot_density=False,
        plot_datapoints=False,
        fill_contours=True,
        max_n_ticks=3,
        min_n_ticks=3,
        save=False,
        truth_color="darkorange",
        labelpad=0.2
        )



    plt.suptitle(f"{title}", fontsize=16)
    plt.show()

    params_to_plot = [
    "R_14", "lambda_tilde", "TOV_mass", "mass_1_source", "mass_2_source", 
    ]

    labels = [
        r"$r_{1.4}$", r"$\tilde{\Lambda}$", r"$m_{TOV}$", r"$m_{1}$", r"$m_{2}$"]
    df_clean = df[params_to_plot].dropna()


    
    figure = corner.corner(
        df_clean[params_to_plot],
        bins=40,
        smooth=True,    
        color='salmon',
        labels=labels,
        label_kwargs=dict(fontsize=axis_fontsize),
        title_kwargs=dict(fontsize=14),
        quantiles=[0.05,0.95],
        levels=[0.68, 0.95],
        plot_density=False,
        plot_datapoints=False,
        fill_contours=True,
        max_n_ticks=3,
        min_n_ticks=3,
        save=False,
        truth_color="darkorange",
        labelpad=0.2
        )


    plt.suptitle(f"{title}", fontsize=16)

    plt.show()

def plot_parameters(dfs, injected_params=None, title=None, labels=None, colours=None, truth_colour="#DC267F", contour_fill=True, which=None, legend_fontsize=14, axis_fontsize=14, weighted=False):
    """
    Plot contours, either singular PE run or comparison between multiple PE runs.

    Parameters
    ----------
    dfs : list of pandas.DataFrame
        List of posterior dataframes to compare.
    injected_params : list of dict, optional
        List of injected parameter values for each dataframe (used for truth lines).
    title : str, optional
        Title for the comparison figures.
    labels : list of str, optional
        Labels for each dataframe (used in legend).
    colours : list of str, optional
        Colours for each dataset.
    truth_colour : str, optional
        Color for the truth lines.
    contour_fill : bool, optional
        Whether to fill the contours in the plots.
    which : int, optional
        If 1, only plot first parameter group; if 2, only plot second group; if None, plot both.
    legend_fontsize : int, optional
        Font size for the legend text.
    axis_fontsize : int, optional
        Font size for the axis labels.
    weighted : bool, optional
        Whether to weight the histograms by likelihood values.
    """
    if injected_params is None:
        is_injection = False

    is_injection = True if injected_params is not None else False
    if labels is None:
        labels = [f"Set {i+1}" for i in range(len(dfs))]
    if colours is None:
        cmap = plt.get_cmap("tab10")
        colours = [to_hex(cmap(i % 10)) for i in range(len(dfs))]

    is_1df = len(dfs) == 1

    # first parameter group (binary/source parameters)
    params1 = [
        "chirp_mass", "mass_ratio", "a_1", "a_2",
        "tilt_1", "tilt_2", "luminosity_distance", "geocent_time",
    ]
    lab1 = [
        r"$\mathcal{M}$ in [M$_\odot$]", r"$q$", r"$a_1$", r"$a_2$",
        r"$\theta_1$", r"$\theta_2$", r"$d_L$", "coalscence time"
    ]

    if not is_injection:
        truths = [None] * len(dfs)
    else:
        truths = [[injection_param[p] for p in params1] for injection_param in injected_params]

    if which is None or which == 1:
        # Build base figure from first dataframe
        df0 = dfs[0][params1].dropna()
        fig1 = corner.corner(
            df0,
            bins=40,
            smooth=True,
            color=colours[0],
            labels=lab1,
            label_kwargs=dict(fontsize=axis_fontsize),
            hist_kwargs=dict(density=True),
            levels=[0.68, 0.95],
            show_titles=is_1df,
            title_kwargs={"fontsize": 14},
            title_fmt=".3f",
            plot_density=False,
            plot_datapoints=False,
            fill_contours=contour_fill,
            contourf_kwargs=dict(zorder=2, alpha=[0.0, 0.4, 0.8]), #outside outer contour, between inner and outer, inside inner contour
            contour_kwargs=dict(zorder=3),
            max_n_ticks=3,
            min_n_ticks=3,
            truths=truths[0],
            truth_color=truth_colour,
            truth_kwargs=dict(zorder=10),
            fig=plt.figure(figsize=(20, 20))

        )
        

        # Overlay remaining dataframes
        for i, df in enumerate(dfs[1:], start=1):
            data = df[params1].dropna()
            corner.corner(
                data,
                fig=fig1,
                bins=40,
                smooth=True,
                color=colours[i],
                hist_kwargs=dict(density=True),
                levels=[0.68, 0.95],
                plot_density=False,
                plot_datapoints=False,
                fill_contours=contour_fill,
                contourf_kwargs=dict(zorder=2, alpha=[0.0, 0.4, 0.8]),
                contour_kwargs=dict(zorder=3),
            )

        # Legend
        handles = [Patch(color=colours[i], lw=2, label=labels[i], fill=False) for i in range(len(dfs))]
        fig1.axes[1].legend(handles=handles, loc='center right', bbox_to_anchor=(1.05, 0.5),
            ncol=1, fancybox=True, prop={'size': legend_fontsize})

        plt.suptitle(title or "Comparison (set 1)", fontsize=16)

        if which is None:
            plt.show()

    if which is None or which == 2:
        # second parameter group (EOS / radius / tidal)
        params2 = [
                "lambda_14", "R_14", "TOV_mass", "p_3ns"
        ]
        lab2 = [
                r"$\Lambda_{1.4}$", r"$R_{1.4}$ in [km]", r"$M_{\mathrm{TOV}}$ in [M$_\odot$]", r"$p_{3n_{\mathrm{sat}}}$ in [MeV/fm$^3$]"
        ]

        if not is_injection:
                truths = [None] * len(dfs)
        else:
                truths = [[injection_param[p] for p in params2] for injection_param in injected_params]


        df0b = dfs[0][params2].dropna()
        df0b = df0b[df0b["R_14"] != 0]

        if weighted:
            weights0 = dfs[0]["Likelihood"].dropna()[dfs[0]["R_14"] != 0]
        else:
            weights0 = None

        fig2 = corner.corner(
            df0b,
            bins=40,
            smooth=True,
            color=colours[0],
            labels=lab2,
            weights=weights0,
            hist_kwargs=dict(density=True),
            label_kwargs=dict(fontsize=axis_fontsize),
            show_titles=is_1df,
            title_kwargs={"fontsize": 20},
            title_fmt=".3f",
            axes_kwargs={"fontsize": 16},
            levels=[0.68, 0.95],
            plot_density=False,
            plot_datapoints=False,
            fill_contours=contour_fill,
            contourf_kwargs=dict(zorder=2, alpha=[0.0, 0.4, 0.8]),
            contour_kwargs=dict(zorder=3),
            max_n_ticks=3,
            min_n_ticks=3,
            truths=truths[0],
            truth_color=truth_colour,
            truth_kwargs=dict(zorder=10),
            fig=plt.figure(figsize=(20, 20)),
        )
        for i, df in enumerate(dfs[1:], start=1):
            data = df[params2].dropna()
            data = data[data["R_14"] != 0]
            if weighted:
                weight = df["Likelihood"].dropna()[df["R_14"] != 0]
            else:
                weight = None
            corner.corner(
                data,
                fig=fig2,
                weights=weight,
                hist_kwargs=dict(density=True),
                bins=40,
                smooth=True,
                color=colours[i],
                levels=[0.68, 0.95],
                plot_density=False,
                plot_datapoints=False,
                fill_contours=contour_fill,
                contourf_kwargs=dict(zorder=2, alpha=[0.0, 0.4, 0.8]),
                contour_kwargs=dict(zorder=3),
            )

        handles2 = [Patch(color=colours[i], lw=2, label=labels[i], fill=False) for i in range(len(dfs))]
        fig2.axes[0].legend(handles=handles2, loc='center right', bbox_to_anchor=(2.05, 0.5),
            ncol=1, fancybox=True, prop={'size': legend_fontsize})
        plt.suptitle(title or "Comparison (set 2)", fontsize=16)
        if which is None:
            plt.show()

def plot_JSD(column, dfs, labels, title=None, bins=40, weighted=False, size=(8, 6), box=True, annot_size=16):
    """Plot the Jensen-Shannon Divergence between multiple pairs of dataframes for a given column."""
    n = len(dfs)
    jsd_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            jsd_matrix[i, j] = JSD(column, dfs[i], dfs[j], bins, weighted)
            jsd_matrix[j, i] = jsd_matrix[i, j]

    plt.figure(figsize=size)
    sb.heatmap(jsd_matrix, xticklabels=labels, yticklabels=labels, 
               annot=True, cmap='plasma', center=0.5, vmin=0, vmax=1, annot_kws={"size": annot_size})

    if box:
        cbar = plt.gca().collections[0].colorbar
        for spine in cbar.ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color('black')
        for spine in plt.gca().spines.values():
            spine.set_visible(True)      
            spine.set_linewidth(1)       
            spine.set_color('black')  


    plt.title(f'Jensen-Shannon Divergence for {column}' if title is None else title)


