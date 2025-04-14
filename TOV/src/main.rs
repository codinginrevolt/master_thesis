mod dorpi5;
mod npyfiles;
mod tov;


extern crate ndarray;
extern crate ndarray_npy;

use std::error::Error;
use dorpi5::ArrayOrScalar;
use npyfiles::*;
use tov::*;

fn main() -> Result<(), Box<dyn Error>> {

    let filepath = "/home/sam/thesis/thesiscode/TOV/src/tov.toml";

    let config = load_config(filepath);
    
    let (edens, pressures, cs2s) = load_npyfile(&config)?;
    
    let rtol = ArrayOrScalar::Scalar(1e-6_f64);
    let atol = ArrayOrScalar::Scalar(1e-9_f64);

    let results = match config.settings.kind {
        TovType::MR => get_results_mr(edens, pressures, &config, &rtol, &atol)?,
        TovType::Tidal =>  get_results_tidal(edens, pressures, cs2s, &config, &rtol, &atol)?,
        TovType::Debug => get_debug_results(edens, pressures, cs2s, &config, &rtol, &atol)?,
    };

    let _ = write_npyfile(results, &config);

    Ok(()) 
}
