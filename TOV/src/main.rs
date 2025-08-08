mod dorpi5;
mod npyfiles;
mod tov;
mod sort;


extern crate ndarray;
extern crate ndarray_npy;

use std::error::Error;
use ndarray::{stack, Array3, Axis};

use dorpi5::ArrayOrScalar;
use npyfiles::*;
use sort::sort_indices;
use tov::*;

fn main() -> Result<(), Box<dyn Error>> {

    let filepath = "./tov.toml"; // make sure tov.toml path is correct

    let config = load_config(filepath);
    
    let (edens, pressures, cs2s) = load_npyfile(&config)?;
    
    let mut eos_set = match (&config.settings.kind, &config.output.sort_nmma){
        (TovType::Tidal, true) => stack(
                    Axis(0),
             &[edens.view(), pressures.view(), cs2s.view()])
             .expect("Axis for stacking out of bound when puttng EOS back together for sorting"),
        _ =>{
            let shape = edens.shape();
            Array3::from_elem((3, shape[0], shape[1]), 0.0)},
    };
     
    let rtol = ArrayOrScalar::Scalar(1e-8_f64);
    let atol = ArrayOrScalar::Scalar(1e-10_f64);

    let mut results = match config.settings.kind {
        TovType::MR => get_results_mr(edens, pressures, &config, &rtol, &atol)?,
        TovType::Tidal =>  get_results_tidal(edens, pressures, cs2s, &config, &rtol, &atol)?,
        TovType::Debug => get_debug_results(edens, pressures, cs2s, &config, &rtol, &atol)?,
    };

    match config.settings.kind{
        TovType::Tidal => {
            if config.output.sort_nmma == true{
                let sorted_indices = sort_indices(&results);
                eos_set = eos_set.select(Axis(1), &sorted_indices);
                let _ = rewrite_npyfile(eos_set, &config, &sorted_indices);
                print!("Sorted ");
                results = results.select(Axis(1), &sorted_indices);
            
            }
    }
        _ => ()
    }

    let _ = match config.output.out_type {
        OutType::npy => write_npyfile(results, &config),
        OutType::dat => write_datfile(results, &config),
        OutType::both => {let _ = write_npyfile(results.clone(), &config);
                        write_datfile(results, &config)}, 
    };
    
    Ok(()) 
}
