############################################
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import toml
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import eos
import sampling as sam
import prepare_pqcd as pp
import prepare_ceft as pc
import append_results as ar
import anal_helpers as anal
from config import parse_config




############################################


def generate_and_check(
    config,
    n_ceft,
    cs2_ceft_avg,
    e_ini,
    p_ini,
    mu_ini,
    cs2_crust, 
    phi_ceft_sigma = None,
    n_crust = None,
    cs2_l_ceft = None,
    cs2_u_ceft = None

):

    import eos as EOS, sampling as sam, prepare_pqcd as pp, constrain_on_chempot as coc

    if config.use_cs2 and not config.use_mu_obs:
        (cs2_test, n, X_hat, n_ceft_end_hat, l_hat) = coc.generate_bounded_sample(        
        n_ceft,
        cs2_ceft_avg,
        cs2_l_ceft,
        cs2_u_ceft,
        cs2_crust,
        config.n_end,
        config.mu_start,
        config.mu_end,
        config.n_points,
        config.n_ceft_end,
        config.kernel,
        #burn_in=800 #for the ones in Jarvis
    )

    elif config.use_mu_obs:
        (cs2_test, n, X_hat, n_ceft_end_hat, l_hat) = coc.generate_constrained_sample(
            n_ceft,
            cs2_ceft_avg,
            cs2_l_ceft,
            cs2_u_ceft,
            cs2_crust,
            mu_ini,
            config.n_end,
            config.mu_start,
            config.mu_end,
            config.n_points,
            config.n_ceft_end,
            config.kernel,
            #burn_in=800 #for the ones in Jarvis
        )

    else:
        (phi, n, X_hat, n_ceft_end_hat, l_hat) = sam.generate_sample(
            n_ceft,
            cs2_ceft_avg,
            phi_ceft_sigma,
            n_crust,
            cs2_crust,
            config.n_end,
            config.mu_start,
            config.mu_end,
            config.n_points,
            config.n_ceft_end,
            config.kernel,
        )

    result = {
        "accepted": False,
        "e": None,
        "p": None,
        "cs2": None,
        "phi": None,
        "n": None,
        "Xhat": None,
        "l_hat": None,
    }

    if config.convert_eos:
        if config.use_cs2:
            EOS = eos.EosProperties(n, phi=None, epsi_0=e_ini, p_0=p_ini, mu_0=mu_ini, cs2=cs2_test)
        
        else:
            EOS = eos.EosProperties(n, phi, epsi_0=e_ini, p_0=p_ini, mu_0=mu_ini)

        eos_result = EOS.get_all()

        if not config.use_mu_obs:
            if config.return_connecting and not pp.check_pqcd_connection(
                X_hat, eos_result["epsilon"][-1], eos_result["pressure"][-1], config.n_end
            ):
                return result  # rejected sample, returns empty result
        

        (result["e"], result["p"], result["cs2"]) = (
            eos_result["epsilon"],
            eos_result["pressure"],
            eos_result["cs2"],
        )
        result["accepted"] = True
        if config.return_normscale:
            result["Xhat"] = X_hat
            result["n_ceft_end"] = n_ceft_end_hat
            result["l_hat"] = l_hat
        if not config.return_only_eos:
            result["phi"] = phi
            result["n"] = n
    else:
        result["phi"] = phi
        result["n"] = n
        result["accepted"] = True

    return result


def process_sample(args):
    return generate_and_check(*args)


if __name__ == "__main__":

    # input changes in toml file
    config_file = "samples.toml"
    raw_config = toml.load(config_file)
    config = parse_config(raw_config)

    print("EOS characteristics:")
    print(
        f"Generating {config.samples_n} EOS\n"
        f"with kernel {config.kernel},\n"
        f"ending at {config.n_end} nsat\n"
        f"with {config.n_points} points"
    )
    if config.use_cs2:
        print("Using constrained GP in cs2-n space.")
    if config.use_mu_obs:
        print("All samples connected to pQCD using integral mu observations.")
    if config.n_ceft_end == 0:
        print(f"Chiral EFT endpoint treated as a hyperparameter.\n")
    else:
        print(f"Chiral EFT trusted until {config.n_ceft_end} n_sat.\n")

    print("Output configuration:")
    print(f" EOS will be converted to edens and pressure: {config.convert_eos}.")
    if config.convert_eos:
        print(f" Only converted EOS will be returned: {config.return_only_eos}")
        print(
            f" Connection to pQCD will be valid for all returned EOS: {config.return_connecting}."
        )

    (
        n_ceft,
        cs2_ceft_avg,
        phi_ceft_sigma,
        e_ini,
        p_ini,
        mu_ini,
        n_crust,
        e_crust,
        p_crust,
        cs2_crust,
    ) = pc.make_conditioning_eos()

    if config.use_cs2:
        (_, _, cs2_l_ceft, cs2_u_ceft) = anal.get_ceft_cs2()

    results_phi = []
    results_n = []
    results_p = []
    results_e = []
    results_cs2 = []
    results_Xhat = []
    results_n_ceft_end = []

    results_l_hat = []

    NEOS = 0

    with Pool(processes=cpu_count()) as pool, tqdm(total=config.samples_n) as pbar:
        while NEOS < config.samples_n:

            batch_size = config.samples_n - NEOS
            if not config.use_cs2:
                args = [
                (
                    config,
                    n_ceft,
                    cs2_ceft_avg,
                    e_ini,
                    p_ini,
                    mu_ini,
                    cs2_crust,
                    phi_ceft_sigma,
                    n_crust,
                    None,
                    None
                )
                ] * batch_size
            else:
                args = [
                (
                    config,
                    n_ceft,
                    cs2_ceft_avg,
                    e_ini,
                    p_ini,
                    mu_ini,
                    cs2_crust,
                    None,
                    None,
                    cs2_l_ceft,
                    cs2_u_ceft
                )
                ] * batch_size


            for result in pool.imap_unordered(process_sample, args):
                if result["accepted"]:
                    if config.convert_eos:
                        ar.append_eos_results(
                            results_e,
                            results_p,
                            results_cs2,
                            result["e"],
                            result["p"],
                            result["cs2"],
                            e_crust,
                            p_crust,
                            cs2_crust,
                        )

                        if config.return_normscale:
                            results_Xhat.append(result["Xhat"])
                            results_n_ceft_end.append(result["n_ceft_end"])
                            results_l_hat.append(result["l_hat"])
                        if not config.return_only_eos:
                            ar.append_n_phi(
                                results_n,
                                results_phi,
                                result["n"],
                                result["phi"],
                                n_crust,
                                cs2_crust,
                            )
                    else:
                        ar.append_n_phi(
                            results_n,
                            results_phi,
                            result["n"],
                            result["phi"],
                            n_crust,
                            cs2_crust,
                        )

                    NEOS += 1
                    pbar.update(1)
                if NEOS >= config.samples_n:
                    break

    # outputting
    os.makedirs(config.save_dir, exist_ok=True)

    if config.convert_eos:
        save_path_eos = os.path.join(config.save_dir, f"{config.filename}_eos.npy")
        final_array_eos = np.array([results_e, results_p, results_cs2])
        np.save(save_path_eos, final_array_eos)

        if config.return_normscale:
            save_path_Xhat = os.path.join(
                config.save_dir, f"{config.filename}_renormscale.npy"
            )
            final_array_Xhat = np.array(results_Xhat)
            np.save(save_path_Xhat, final_array_Xhat)

            save_path_l_hat = os.path.join(
                config.save_dir, f"{config.filename}_lengthscale.npy"
            )
            final_array_l_hat = np.array(results_l_hat)
            np.save(save_path_l_hat, final_array_l_hat)

            if config.n_ceft_end == 0:
                save_path_n_ceft_end = os.path.join(
                    config.save_dir, f"{config.filename}_nceft_end.npy"
                )
                final_array_n_ceft_end = np.array(results_n_ceft_end)
                np.save(save_path_n_ceft_end, final_array_n_ceft_end)

        print(
            f"Saved {len(results_e)} EOSes in pressure-energy density space results to {save_path_eos}"
        )

    if not config.return_only_eos or not config.convert_eos:
        save_path_phi = os.path.join(config.save_dir, f"{config.filename}_phi.npy")
        final_array_phi = np.array([results_n, results_phi])
        np.save(save_path_phi, final_array_phi)
        print(f"Saved EOSes in phi-number density space results to {save_path_phi}")
