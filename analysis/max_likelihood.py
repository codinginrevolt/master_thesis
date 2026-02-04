import os
import argparse
import pandas as pd
import numpy as np
import json

import heavy_pulsars.eos_inference_PSRJ1614_radio_timing as psr16
import heavy_pulsars.eos_inference_PSRJ0740_radio_timing as psr07
import post_anal as pa

def main():
    validate_psr_dir()

    with open(nmma_path, "r") as f:
        temp_json = json.load(f)

    content = temp_json.get("posterior", {}).get("content")

    df = pd.DataFrame(content)

    df["EOS"] = np.int64(np.floor(df["EOS"]))

    
    PSR07, PSR16 = pa.load_psr(set_name)
    pa.likelihood_columns(df, PSR07, PSR16)

    eos_max_likelihood = df.loc[df["Likelihood"].idxmax(), "EOS"]

    print(f"Max likelihood EoS index for {set_name}: {eos_max_likelihood}")


def validate_psr_dir():
    if not os.path.exists(PSRJ1614_file):
        print("PSR J1614 likelihood file not found. Creating it now.")
        psr16.eos_path_macro = eos_path_macro
        psr16.outfile = PSRJ1614_file

        psr16.main()
    if not os.path.exists(PSRJ0740_file):
        print("PSR J0740 likelihood file not found. Creating it now.")
        psr07.eos_path_macro = eos_path_macro
        psr07.outfile = PSRJ0740_file
        psr07.main()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Return index of max likelihood EoS from PSRs + GW")
    parser.add_argument("set_name", help="dataset name, e.g. SE_10ns_1")
    args = parser.parse_args()
    set_name = args.set_name
    
    # path to the eos files
    eos_path_macro = f"/home/sam/thesis/code/results/tov_res/{set_name}/"

    if not os.path.exists(eos_path_macro):
        raise FileNotFoundError(f"TOV not solved yet. No directory named {eos_path_macro}")
    
    # outfile
    PSRJ1614_file = f"/home/sam/thesis/code/results/pulsars/{set_name}/eos_likelihood_PSRJ1614"
    PSRJ0740_file = f"/home/sam/thesis/code/results/pulsars/{set_name}/eos_likelihood_PSRJ0740"

    nmma_path = f"/home/sam/thesis/code/results/nmma/{set_name}/GW170817_result.json"

    main()