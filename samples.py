############################################
import numpy as np
import argparse
import os
import toml
from tqdm import tqdm

import dickandballs as db
import helpers as hel
############################################



if __name__ == "__main__":
    #input changes in toml file
    config_file = "samples.toml"
    config = toml.load(config_file)

    samples_n = config['samples']['samples_number']
    n_end = config['samples']['n_end']
    n_points = config['samples']['n_points']

    convert_eos = config['samples']['convert_eos']
    return_only_eos = config['samples']['return_only_eos']

    n_ceft, cs2_ceft_avg, phi_ceft_sigma, e_ini, p_ini, mu_ini = hel.make_conditioning_eos()

    results_phi = []
    results_n = []
    results_p = []
    results_e = []

    for _ in tqdm(range(samples_n), desc="Generating samples"):
        phi, n = hel.generate_sample(n_ceft, cs2_ceft_avg, phi_ceft_sigma, n_end, n_points)

        if convert_eos:
            eos = db.EosProperties(n, phi, epsi_0=e_ini, p_0=p_ini, mu_0=mu_ini)
            eos_result = eos.get_all()

            results_e.append(eos_result["epsilon"])
            results_p.append(eos_result["pressure"])

            if not return_only_eos:
                results_phi.append(phi)
                results_n.append(n)
        else:
            results_phi.append(phi)
            results_n.append(n)

    # outputting
    save_dir = config['output']['directory']
    filename = config['output']['filename']
    os.makedirs(save_dir, exist_ok=True)

    if convert_eos:
            save_path_eos = os.path.join(save_dir, f"{filename}_eos.npy")
            final_array_eos = np.array([results_e, results_p])
            np.save(save_path_eos, final_array_eos)
            print(f"Saved EOS in pressure-energydens results to {save_path_eos}")

    if not return_only_eos or not convert_eos:
        save_path_phi = os.path.join(save_dir, f"{filename}_phi.npy")
        final_array_phi = np.array([results_n, results_phi])
        np.save(save_path_phi, final_array_phi)
        print(f"Saved EOS in phi-numberdens results to {save_path_phi}")

#TODO: make the script take arguments like the kernel and the num samples to generate
