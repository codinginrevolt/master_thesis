This is a Rust/Python translation of the [tmg package for R](https://rdrr.io/cran/tmg/) based on the paper [Exact Hamiltonian Monte Carlo for Truncated Multivariate Gaussians](https://doi.org/10.1080/10618600.2013.788448).

Unlike the R package, this only focuses on linear constraints for multivariate Gaussians.

# How to use:

Make sure maturin (and Rust) is installed in your desired python environment: `pip install maturin`

Then run `maturin develop -r` and the package will be installed in the python environment.
