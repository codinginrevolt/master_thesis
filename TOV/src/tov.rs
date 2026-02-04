mod tov_helpers;
mod tov_setup;
mod tov_solvers;

use tov_solvers::*;

use crate::{dorpi5::*, npyfiles::Config};
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use ndarray::{s, Array1, Array2, Array3, Axis};
use std::error::Error;

use rayon::prelude::*;

pub fn get_results_mr(
    edens: Array2<f64>,
    pressures: Array2<f64>,
    config: &Config,
    rtol: &ArrayOrScalar,
    atol: &ArrayOrScalar,
) -> Result<Array3<f64>, Box<dyn Error>> {
    let tov_iters = config.settings.tov_iters;

    let eos_num = edens.nrows();

    let mut results = Array3::<f64>::zeros((3, eos_num, tov_iters));

    let pb = ProgressBar::new(eos_num as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner} [{elapsed_precise}] [{bar}] {pos}/{len} ({per_sec}) ({eta})")
            .unwrap(),
    );

    (0..eos_num)
        .into_par_iter()
        .zip(results.axis_iter_mut(Axis(1)))
        .for_each(|(i, mut result_slice)| {
            let mut e_array = edens.row(i).to_owned();
            let mut p_array = pressures.row(i).to_owned();

            let (radii, masses, core_pressures) =
                get_mr_from_eos(&mut e_array, &mut p_array, 2.0, tov_iters, &rtol, &atol, &config)
                    .expect("Error in looping over all EOSes");

            result_slice.slice_mut(s![0, ..]).assign(&radii);
            result_slice.slice_mut(s![1, ..]).assign(&masses);
            result_slice.slice_mut(s![2, ..]).assign(&core_pressures);

            pb.inc(1);
        });
    pb.finish();

    Ok(results)
}

pub fn get_results_tidal(
    edens: Array2<f64>,
    pressures: Array2<f64>,
    cs2s: Array2<f64>,
    config: &Config,
    rtol: &ArrayOrScalar,
    atol: &ArrayOrScalar,
) -> Result<Array3<f64>, Box<dyn Error>> {
    let eos_num = edens.nrows();

    let tov_iters = config.settings.tov_iters;

    let mut results = Array3::<f64>::zeros((4, eos_num, tov_iters));

    let pb = ProgressBar::new(eos_num as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner} [{elapsed_precise}] [{bar}] {pos}/{len} ({per_sec}) ({eta})")
            .unwrap(),
    );

    (0..eos_num)
        .into_par_iter()
        .zip(results.axis_iter_mut(Axis(1)))
        .for_each(|(i, mut result_slice)| {
            let e_array = edens.row(i).to_owned();
            let p_array = pressures.row(i).to_owned();
            let cs2_array = cs2s.row(i).to_owned();

            let (radii, masses, lambdas, core_pressures) = get_tidal_mr_from_eos(
                e_array,
                p_array,
                &cs2_array,
                2.0,
                tov_iters,
                &rtol,
                &atol,
                config
            )
            .expect("Error in looping over all EOSes");

            result_slice.slice_mut(s![0, ..]).assign(&radii);
            result_slice.slice_mut(s![1, ..]).assign(&masses);
            result_slice.slice_mut(s![2, ..]).assign(&lambdas);
            result_slice.slice_mut(s![3, ..]).assign(&core_pressures);

            pb.inc(1);
        });
    pb.finish();

    Ok(results)
}

pub fn get_debug_results(
    edens: Array2<f64>,
    pressures: Array2<f64>,
    cs2s: Array2<f64>,
    config: &Config,
    rtol: &ArrayOrScalar,
    atol: &ArrayOrScalar,
) -> Result<Array3<f64>, Box<dyn Error>> {
    let eos_num = edens.nrows();

    let mut results_vec = Vec::with_capacity(eos_num);
    let mut max_len = 0;

    let pb = ProgressBar::new(eos_num as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner} [{elapsed_precise}] [{bar}] {pos}/{len} ({per_sec}) ({eta})")
            .unwrap(),
    );

    for i in 0..(edens.nrows()) {
        let mut e_array = edens.row(i).to_owned();
        let mut p_array = pressures.row(i).to_owned();
        let cs2_array = cs2s.row(i).to_owned();

        let mut _max_mass_bool = false;

        let (radii, masses, ys, pressures) = get_single_pcore(
            &mut e_array,
            &mut p_array,
            &cs2_array,
            config
                .settings
                .p_core
                .expect("Specify core pressure in the toml file."),
            &rtol,
            &atol,
            &mut _max_mass_bool,
        )?;

        let len = masses.len();
        if len > max_len {
            max_len = len;
        }

        results_vec.push((masses, radii, ys, pressures));
        pb.inc(1);
    }

    let mut results = Array3::<f64>::zeros((4, eos_num, max_len));

    for (i, (masses, radii, ys, pressures)) in results_vec.into_iter().enumerate() {
        let pad = |array: Array1<f64>| {
            let mut padded = Array1::<f64>::from_elem(max_len, f64::NAN);
            padded.slice_mut(s![..array.len()]).assign(&array);
            padded
        };

        results.slice_mut(s![0, i, ..]).assign(&pad(radii));
        results.slice_mut(s![1, i, ..]).assign(&pad(masses));
        results.slice_mut(s![2, i, ..]).assign(&pad(ys));
        results.slice_mut(s![3, i, ..]).assign(&pad(pressures));
    }
    pb.finish();

    Ok(results)
}
