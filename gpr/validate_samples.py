import numpy as np
from scipy.interpolate import interp1d

import toml
from pathlib import Path

import anal_helpers as anal
import prepare_pqcd as pp
from config import parse_config


def get_paths():
    this_dir = Path(__file__).parent
    config_file = "samples.toml"
    raw_config = toml.load(config_file)
    config = parse_config(raw_config)

    eos_pathname = f"{config.save_dir}/{config.filename}_eos.npy"
    rescale_pathname = f"{config.save_dir}/{config.filename}_renormscale.npy"

    save_dir_eos = Path(f"{config.save_dir}/{config.filename}")

    tov_config = toml.load(this_dir.parent / "TOV/tov.toml")
    if tov_config["output"]["out_type"] == "npy":
        tov_pathname = f"{tov_config['output']['path']}/{tov_config['output']['filename']}_tidal.npy"
        save_dir_tov = Path(
            f"{tov_config['output']['path']}/{tov_config['output']['filename']}"
        )
    elif tov_config["output"]["out_type"] == "both":
        tov_pathname = f"{tov_config['output']['path']}/{tov_config['output']['filename']}/{tov_config['output']['filename']}_tidal.npy"
        save_dir_tov = Path(
            f"{tov_config['output']['path']}/{tov_config['output']['filename']}/valids"
        )

    else:
        raise ValueError("TOV solver's output must be in .npy")

    return eos_pathname, rescale_pathname, tov_pathname, save_dir_eos, save_dir_tov


def load_eos_mr(eos_pathname, rescale_pathname, tov_pathname):
    eos = np.load(eos_pathname)
    Xhats = np.load(rescale_pathname)
    tov = np.load(tov_pathname)

    e = eos[0]
    p = eos[1]

    p_c = tov[-1]

    return p_c, p, e, Xhats


def get_ntov(p_c, p, e, n):
    p_i = p_c

    valid_idx = np.where(~np.isnan(p_i))[0]

    last_valid_p = p_i[valid_idx[-1]]

    n_interp = interp1d(p, n, bounds_error=False, fill_value="extrapolate")
    e_interp = interp1d(p, e, bounds_error=False, fill_value="extrapolate")

    ntov = n_interp(last_valid_p)
    etov = e_interp(last_valid_p)
    ptov = last_valid_p

    return ntov, etov, ptov


def get_valid_sams(p_c, p, e, Xhats, n_end=40, n_points=200):

    n_end = 40
    n_points = 200

    n_test = anal.get_n_test(n_end, n_points)

    index = []
    last_index = []

    for i in range(len(e)):

        (ntov, etov, ptov) = get_ntov(p_c[i], p[i], e[i], n_test)

        idx = np.where(n_test <= ntov)[0][-1]

        if pp.check_pqcd_connection(Xhats[i], etov, ptov, ntov):
            last_index.append(idx)
            index.append(i)

    index = np.asarray(index)
    last_index = np.asarray(last_index)

    return index, last_index


def save_as_dat(save_dir, tov_pathname, index):

    save_dir.mkdir(parents=True, exist_ok=True)

    tov = np.load(tov_pathname)
    count = 1
    for i in index:
        slice = tov[:, i, :].T
        non_nan_indices = ~np.isnan(slice).all(axis=1)
        non_nan_rows = slice[non_nan_indices]

        head = f"# r[km]    m(r)[Msol]    y(r)    pressure(r)[MeV/fm³]\n"
        filename = save_dir / f"{count}.dat"

        np.savetxt(filename, non_nan_rows, delimiter=" ", header=head, comments="")

        count += 1


def main():

    (eos_pathname, rescale_pathname, tov_pathname, save_dir_eos, save_dir_tov) = (
        get_paths()
    )
    (p_c, p, e, Xhats) = load_eos_mr(eos_pathname, rescale_pathname, tov_pathname)
    (index, last_index) = get_valid_sams(p_c, p, e, Xhats)

    index_filename = str(save_dir_eos) + "_valid_indices.npy"
    last_index_filename = str(save_dir_eos) + "_last_indices.npy"

    save_as_dat(save_dir_tov, tov_pathname, index)

    np.save(index_filename, index)
    np.save(last_index_filename, last_index)


if __name__ == "__main__":
    main()
