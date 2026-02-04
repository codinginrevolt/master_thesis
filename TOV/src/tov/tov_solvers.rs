use super::tov_helpers::*;
use super::tov_setup::*;

use crate::dorpi5::*;
use crate::npyfiles::Config;
use core::f64;
use ndarray::{Array1, Axis};
use std::error::Error;
use std::vec;

const CS2_THRESHOLD: f64 = 0.001;

pub(crate) fn get_mr_from_eos(
    e_array: &mut Array1<f64>,
    p_array: &mut Array1<f64>,
    p_core_start: f64,
    tov_iters: usize,
    rtol: &ArrayOrScalar,
    atol: &ArrayOrScalar,
    config: &Config,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), Box<dyn Error>> {

    let p_core_end = p_array
            .last()
            .expect("Something's wrong with last pressure value in EOS.")
            .clone();
    
    let p_core: Array1<f64> = (Array1::geomspace(p_core_start, p_core_end, tov_iters))
        .ok_or("Error creating core pressures array")?; // in nuclear units

    let cs2_array = Array1::<f64>::ones(p_array.len()); // dummy array, not used in MR solver
    let tov = TovTide::initiate(e_array, p_array, &cs2_array);
    let derivatives_fx = |r: f64, y: &Array1<f64>| -> Array1<f64> {tov.tov_derivs(r, y)};
    let stop_fx = |x: f64, y_next: &Array1<f64>| -> bool {tov.tov_stopper(x, y_next)};

    let mut init_cond = Array1::from(vec![
        (1e-10_f64),
        0.0
        ]);
    
    let mut masses: Array1<f64> = Array1::from_elem(tov_iters, f64::NAN);
    let mut radii: Array1<f64> = Array1::from_elem(tov_iters, f64::NAN);
    let mut press_core: Array1<f64> = Array1::from_elem(tov_iters, f64::NAN);

    let max_radius = 100.0;
    let mut last_pos_index: usize = 0;

    let mod_rtol = 1e-3_f64 *  rtol;
    let mod_atol = 1e-3_f64 *  atol;


    for (i, &p_core_value) in p_core.iter().enumerate() {

        init_cond[1] = convert_eos_nuclear_to_natural(p_core_value);

        let (r_vals, y_vals) =  solve(
            derivatives_fx,
            1e-5_f64,
            init_cond.to_owned(),
            max_radius,
            &mod_rtol,
            &mod_atol,
            0,
            0.0,
            Some(Box::new(stop_fx)),
        )?;


        let final_state = y_vals.index_axis(Axis(0), y_vals.len_of(Axis(0)) - 1);
        let m: f64 = convert_mass_natural_to_solar(final_state[0]);


        let r = *r_vals
                .last()
                .expect("Failed to extract last radius value from radius profile");

        if config.settings.return_unstable{
            masses[i] = m;
            radii[i] = r;
            press_core[i] = convert_eos_natural_to_nuclear(p_core_value);    
        } else if (i==0) || (m>masses[last_pos_index]){
            masses[i] = m;
            radii[i] = r;
            press_core[i] = convert_eos_natural_to_nuclear(p_core_value);    
            last_pos_index = i;
        }
    }

    Ok((radii, masses, press_core))
}


pub(crate) fn get_tidal_mr_from_eos(
    e_array: Array1<f64>,
    p_array: Array1<f64>,
    cs2_array: &Array1<f64>,
    p_core_start: f64,
    tov_iters: usize,
    rtol: &ArrayOrScalar,
    atol: &ArrayOrScalar,
    config: &Config,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), Box<dyn Error>> {
    
    let p_core_end = p_array
        .last()
        .expect("Something's wrong with last pressure value in EOS.")
        .clone();
    
    let p_core: Array1<f64> = (Array1::geomspace(p_core_start, p_core_end, tov_iters))
        .ok_or("Error creating core pressures array")?; // in nuclear units

    let mut masses: Array1<f64> = Array1::from_elem(tov_iters, f64::NAN);
    let mut radii: Array1<f64> = Array1::from_elem(tov_iters, f64::NAN);
    let mut lamdas: Array1<f64> = Array1::from_elem(tov_iters, f64::NAN);
    let mut press_core: Array1<f64> = Array1::from_elem(tov_iters, f64::NAN);

    let mut last_pos_index: usize = 0;

    for (i, &p_core_value) in p_core.iter().enumerate() {
        let mut max_mass_bool = false;

        let mut e_array_clone = e_array.clone();
        let mut p_array_clone = p_array.clone();

        let (radii_single_pcore, 
            masses_single_pcore, 
            ys, 
            _pressures) = get_single_pcore(
            &mut e_array_clone,
            &mut p_array_clone,
            cs2_array,
            p_core_value,
            rtol,
            atol,
            &mut max_mass_bool,
        )?;


        let m_sol: f64 = masses_single_pcore.last()
            .expect("Could not retrieve last mass from ODE solution.")
            .clone(); 
        let m_nat = convert_mass_solar_to_natural(m_sol);

        let r: f64 = radii_single_pcore.last()
            .expect("Could not retrieve last radius from ODE solution.")
            .clone(); // this r is in km
        
        let ys_last = ys.last()
            .expect("Could not retrieve last y from ODE solution.")
            .clone();

        let lam = compute_tidal_deformability(m_nat, r, ys_last);

        if config.settings.return_unstable{
            masses[i] = m_sol;
            radii[i] = r;
            lamdas[i] = lam;
            press_core[i] = p_core_value;
        } else if (i==0) || (m_sol>masses[last_pos_index]){
            masses[i] = m_sol;
            radii[i] = r;
            lamdas[i] = lam;
            press_core[i] = p_core_value;
            last_pos_index = i;
        } else if max_mass_bool {
          // break solving if no phase transistion, reached m_tov
        }
    
    }

    Ok((radii, masses, lamdas, press_core))

}

pub(crate) fn get_single_pcore(
    e_array: &mut Array1<f64>,
    p_array: &mut Array1<f64>,
    cs2_array: &Array1<f64>,
    p_core: f64,
    rtol: &ArrayOrScalar,
    atol: &ArrayOrScalar,
    max_mass_bool: &mut bool,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), Box<dyn Error>> {

    let tov = TovTide::initiate(e_array, p_array, cs2_array);
    let derivatives_fx = |r: f64, y: &Array1<f64>| -> Array1<f64> {tov.tidal_derivs(r, y)};
    let stop_fx = |x: f64, y_next: &Array1<f64>| -> bool {tov.tov_stopper(x, y_next)};

    let init_cond = Array1::from(vec![
        (1e-10_f64),
        convert_eos_nuclear_to_natural(p_core),
        2.0
    ]);

    let cs2_core = tov
                .cs2_spline
                .clamped_sample(p_core)
                .expect("Expected interpolated cs2_core as f64, got None");
    if cs2_core > CS2_THRESHOLD {
                *max_mass_bool = true;
            }

    let max_radius = 100.0; // km
    let (r_vals, y_vals) =  solve(
        derivatives_fx,
        1e-5_f64,
        init_cond.to_owned(),
        max_radius,
        rtol,
        atol,
        0,
        0.0,
        Some(Box::new(stop_fx)),
    )?;

    let radius = r_vals;
    let m_profile = y_vals.column(0)
                    .mapv(|x| 
                    convert_mass_natural_to_solar(x));
    let p_profile = y_vals.column(1)
                        .mapv(|x|
                        convert_eos_natural_to_nuclear(x));
    let y_profile = y_vals.column(2).to_owned();


    Ok((radius, m_profile, y_profile, p_profile)) // (km, msol, dimensionless, MeV/fm³)
}

