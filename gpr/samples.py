############################################
import numpy as np
import os
import toml
from tqdm import tqdm

import eos as EOS
import helpers as hel
############################################



if __name__ == "__main__":
    #input changes in toml file
    config_file = "samples.toml"
    config = toml.load(config_file)

    samples_n = config['samples']['samples_number']
    n_end = config['samples']['n_end']
    n_points = config['samples']['n_points']

    if config['samples']['mu_start'] == 0.0:
        mu_start = 2.2
    else: mu_start = config['samples']['mu_start']
    if config['samples']['mu_end'] == 0.0:
        mu_end = 2.8
    else: mu_end = config['samples']['mu_end']



    convert_eos = config['samples']['convert_eos']
    return_only_eos = config['samples']['return_only_eos']

    if convert_eos: 
        return_connecting = config['samples']['return_connecting']
        return_normscale = config['samples']['return_normscale']

    else: 
        return_connecting = False
        return_normscale = False
        return_only_eos = False

    if return_connecting: return_normscale = False

    print("EOS characteristics:")
    print(f" Generating {samples_n} EOS \n with kernel SE,\n ending at {n_end} nsat\n with {n_points} points")
    print("Output configuration:")
    print(f" EOS will be converted to edens and pressure: {convert_eos}.")
    if convert_eos:
        print(f" Only converted EOS will be returned: {return_only_eos}")
        print(f" Connection to pQCD will be valid for all returned EOS: {return_connecting}.")

    n_ceft, cs2_ceft_avg, phi_ceft_sigma, e_ini, p_ini, mu_ini, n_crust, e_crust, p_crust, cs2_crust = hel.make_conditioning_eos()

    results_phi = []
    results_n = []
    results_p = []
    results_e = []
    results_cs2 = []
    results_Xhat = []

    NEOS = 0
    with tqdm(total=samples_n) as pbar:
        while NEOS < samples_n:
            phi, n, X_hat = hel.generate_sample(n_ceft, cs2_ceft_avg, phi_ceft_sigma, n_end, mu_start, mu_end, n_points)

            if convert_eos:
                eos = EOS.EosProperties(n, phi, epsi_0=e_ini, p_0=p_ini, mu_0=mu_ini)
                eos_result = eos.get_all()

                if return_connecting and (hel.check_pqcd_connection(X_hat, eos_result["epsilon"][-1], eos_result["pressure"][-1], n_end)):
                    hel.append_eos_results(results_e, results_p, results_cs2, eos_result["epsilon"], eos_result["pressure"], eos_result["cs2"], e_crust, p_crust, cs2_crust)
                    NEOS += 1
                    pbar.update(1)

                elif not return_connecting:
                    hel.append_eos_results(results_e, results_p, results_cs2, eos_result["epsilon"], eos_result["pressure"], eos_result["cs2"], e_crust, p_crust, cs2_crust)
                    NEOS += 1
                    pbar.update(1)

                if return_normscale:
                    results_Xhat.append(X_hat) # crust not added back to when finding Xhat

                if not return_only_eos:
                    hel.append_n_phi(results_n, results_phi, n, phi, n_crust, cs2_crust)

            else:
                hel.append_n_phi(results_n, results_phi, n, phi, n_crust, cs2_crust)
                NEOS += 1
                pbar.update(1)



    # outputting
    save_dir = config['output']['directory']
    filename = config['output']['filename']
    os.makedirs(save_dir, exist_ok=True)



    if convert_eos:
            save_path_eos = os.path.join(save_dir, f"{filename}_eos.npy")
            final_array_eos = np.array([results_e, results_p, results_cs2])
            np.save(save_path_eos, final_array_eos)

            if return_normscale:
                save_path_Xhat = os.path.join(save_dir, f"{filename}_renormscale.npy")
                final_array_Xhat = np.array(results_Xhat)
                np.save(save_path_Xhat, final_array_Xhat)

            print(f"Saved {len(results_e)} EOSes in pressure-energydens results to {save_path_eos}")

    if not return_only_eos or not convert_eos:
        save_path_phi = os.path.join(save_dir, f"{filename}_phi.npy")
        final_array_phi = np.array([results_n, results_phi])
        np.save(save_path_phi, final_array_phi)
        print(f"Saved EOSes in phi-numberdens results to {save_path_phi}")




#TODO: make the script take arguments like the kernel and the num samples to generate
