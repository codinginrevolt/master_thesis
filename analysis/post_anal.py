import numpy as np
import pandas as pd


def load_eos_data(folder_name):
    eos_file = f"/home/sam/thesis/code/results/eos_samples/{folder_name}/{folder_name}_eos.npy"

    try:
        eos = np.load(eos_file)

    except FileNotFoundError:
        try:
            mr_file = f"/home/sam/thesis/code/results/tov_res/{folder_name}_tidal.npy"
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

    filepath = f"/home/sam/thesis/code/results/nmma/{folder_name}/result/GW170817_samples.parquet"

    try:
        df = pd.read_parquet(filepath)
        df["EOS"] = np.int64(np.floor(df["EOS"]))
        additional_columns(df, eos, mr)
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    
def additional_columns(dataframe, eos, mr):

    r = mr[0]
    m = mr[1]
    tide = mr[2]
    press_cent = mr[3]

    press_cent = mr[3]
    edens = eos[0]
    press = eos[1]
    cs2 = eos[2]

    valid_eos = get_valid_eos(dataframe)
    m_from_valid_eos = m[valid_eos]
    tide_from_valid_eos = tide[valid_eos]

    r_14, pc_14, ec_14 = one_point_four(r, m, press_cent, edens, press)
    m1, m2 = component_masses(dataframe["chirp_mass"], dataframe["mass_ratio"])
    tide1, tide2 = component_tidal_deformability(tide_from_valid_eos, m_from_valid_eos, m1, m2)
    lam_tilde = lambda_tilde(m1, m2, tide1, tide2)

    dataframe["r_1.4"] = r_14[valid_eos]
    dataframe["pc_1.4"] = pc_14[valid_eos] # c for center
    dataframe["ec_1.4"] = ec_14[valid_eos]
    dataframe["m1"] = m1
    dataframe["m2"] = m2
    dataframe["tide1"] = tide1
    dataframe["tide2"] = tide2
    dataframe["lam_tilde"] = lam_tilde


def one_point_four(r, m, press_cent, edens, press):
    r_14 = []
    pc_14 = []
    ec_14 = []

    for i in range(len(r)):
        valid_mask = ~np.isnan(m[i]) # unstable branch has NaN mass
        m_valid = m[i][valid_mask]
        r_valid = r[i][valid_mask]
        pc_valid = press_cent[i][valid_mask]
        edens_valid = edens[i]
        press_valid = press[i]
        if len(m_valid) > 1 and (np.min(m_valid) <= 1.4 <= np.max(m_valid) or np.max(m_valid) <= 1.4 <= np.min(m_valid)): # if eos reaches 1.4
            r_interp = np.interp(1.4, m_valid, r_valid)
            r_14.append(r_interp)
            pc_interp = np.interp(1.4, m_valid, pc_valid)
            pc_14.append(pc_interp)
            ec_interp = np.interp(pc_interp, press_valid, edens_valid)
            ec_14.append(ec_interp)
        else:
            r_14.append(np.nan)
            pc_14.append(np.nan)
            ec_14.append(np.nan)
            
    r_14 = np.asarray(r_14)
    pc_14 = np.asarray(pc_14)
    ec_14 = np.asarray(ec_14)

    return r_14, pc_14, ec_14

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

def component_tidal_deformability(tide, m, m1, m2):
    tide1 = []
    tide2 = []
    for i in range(len(m1)):
        valid_mask = ~np.isnan(m[i]) # unstable branch has NaN mass
        m_valid = m[i][valid_mask]
        tide_valid = tide[i][valid_mask]
        
        if len(m_valid) > 1 and (np.min(m_valid) <= 1.4 <= np.max(m_valid) or np.max(m_valid) <= 1.4 <= np.min(m_valid)):
            tide1_interp = np.interp(m1[i], m_valid, tide_valid)
            tide1.append(tide1_interp)
            tide2_interp = np.interp(m2[i], m_valid, tide_valid)
            tide2.append(tide2_interp)
        else:
            tide1.append(np.nan)
            tide2.append(np.nan)
    return tide1, tide2 

def lambda_tilde(m1, m2, tide1, tide2):
    const = 16/13
    numer1 = (m1 + 12*m2) * m1**4 * tide1
    numer2 = (m2 + 12*m1) * m2**4 * tide2
    denom = (m1 + m2)**5
    lam_tilde = const * (numer1 + numer2) / denom
    return lam_tilde

def get_valid_eos(df):
    valid_eos = []
    for i in df["EOS"]:
        valid_eos.append(i)
    return np.asarray(valid_eos)