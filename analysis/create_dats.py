import numpy as np
import os 
import argparse
from tqdm import tqdm

def npy_to_dat(npy_path, out_dir, start_index=1):
    data = np.load(npy_path, mmap_mode=None)
    if data.ndim != 3:
        raise ValueError(f"expected 3D array (m, n, p), got shape {data.shape}")
    m, n, p = data.shape
    os.makedirs(out_dir, exist_ok=True)

    for i in tqdm(range(n), desc="Writing .dat files", unit="file"):
        slice_mp = data[:, i, :]          # shape (m, p)
        arr = slice_mp.T                  # shape (p, m) -> original rows
        mask = ~np.all(np.isnan(arr), axis=1)  # keep rows not all-NaN
        rows = arr[mask]
        file_index = start_index + i
        file_path = os.path.join(out_dir, f"{file_index}.dat")

        header = "# r[km]    m[Msol]    lambda    core_pressure[MeV/fm³]"

        np.savetxt(file_path, rows, header=header, comments='')


    print(f"Saved {n} files to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MRT curves in .npy (m,n,p) format to .dat files (one per [:,i,:])")
    parser.add_argument("set_name", help="dataset name, e.g. SE_10ns_3")
    args = parser.parse_args()

    set_name = args.set_name
    npy_path = f"/home/sam/thesis/code/results/tov_res/{set_name}_tidal.npy"  # Path to .npy file
    out_dir = f"/home/sam/thesis/code/results/tov_res/{set_name}/"  # Directory to save .dat files
    npy_to_dat(npy_path, out_dir)