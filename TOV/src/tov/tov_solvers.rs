use super::tov_setup::*;
use super::tov_helpers::*;

use crate::dorpi5:: *;
use ndarray::{Array1, Axis};
use core::f64;
use std::error::Error;


pub(crate) fn get_mr_from_eos(
    e_array: &mut Array1<f64>,
    p_array: &mut Array1<f64>,
    p_core_start:f64,
    tov_iters:usize,
    rtol: &ArrayOrScalar,
    atol: &ArrayOrScalar)
    -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), Box<dyn Error> >
    {
    let p_core_end = p_array.last().expect("Something's wrong with last pressure value in EOS.").clone();
    let tov = TovMR::initiate(e_array, p_array);

    let p_core: Array1<f64> = (Array1::geomspace(p_core_start, p_core_end, tov_iters))
                        .ok_or("Error creating core pressures array")?
                        .mapv_into(|x|tov.factors.scale_presssure_to_dimensionless(x));

    let derivatives_fx = |r: f64, y: &Array1<f64>| -> Array1<f64> {
        tov.tov_derivs(r, y)
    };
    let stop_fx = |x: f64, y_next: &Array1<f64>| -> bool {
        tov.tov_stopper(x, y_next)
    };

    let mut init_cond = Array1::from(vec![(1e-10_f64),0.0]);

    let mut masses: Array1<f64> = Array1::from_elem(tov_iters, f64::NAN);
    let mut radii: Array1<f64> = Array1::from_elem(tov_iters, f64::NAN);
    let mut press_core: Array1<f64> = Array1::from_elem(tov_iters, f64::NAN);


    let max_radius = tov.factors.scale_radius_to_dimensioneless(100.0);

    for (i, &p_core_value) in p_core.iter().enumerate(){
        init_cond[1] = p_core_value;

        let (r_vals, y_vals) =
            solve(derivatives_fx,
                tov.factors.scale_radius_to_dimensioneless(1.0e-5),
                init_cond.to_owned(),
                max_radius,
                rtol,
                atol,
                0,
                0.0,
                Some(Box::new(stop_fx))
            )?;

        let final_state = y_vals.index_axis(Axis(0), y_vals.len_of(Axis(0)) - 1);

        let m: f64 = tov.factors.restore_mass_to_solar(final_state[0]);
        if (i!= 0) && (m<masses[i-1]){
            break;
        }

        let r = tov.factors.restore_radius_to_natural(*r_vals.last().expect("Failed to extract last mass value from radius profile"));


        masses[i] = m;
        radii[i] = r;
        press_core[i] = tov.factors.restore_pressure_to_nuclear(p_core_value);

    }

    Ok( (masses, radii, press_core) )

}


 pub(crate) fn get_mr_tidal_from_eos(
    e_array: &mut Array1<f64>,
    p_array: &mut Array1<f64>,
    cs2_array: &Array1<f64>,
    p_core_start:f64,
    tov_iters:usize,
    rtol: &ArrayOrScalar,
    atol: &ArrayOrScalar)
    -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), Box<dyn Error> >
    {
    let p_core_end = p_array.last().expect("Something's wrong with last pressure value in EOS.").clone();
    
    let tov = TovMR::initiate(e_array, p_array);

    let p_core: Array1<f64> = (Array1::geomspace(p_core_start, p_core_end, tov_iters))
                            .ok_or("Error creating core pressures array")?
                            .mapv_into(|x|tov.factors.scale_presssure_to_dimensionless(x));

    let derivatives_fx = |r: f64, y: &Array1<f64>| -> Array1<f64> {
        tov.tov_derivs(r, y)
    };
    let stop_fx = |x: f64, y_next: &Array1<f64>| -> bool {
        tov.tov_stopper(x, y_next)
    };

    let mut init_cond: Array1<f64> = Array1::from(vec![tov.factors.scale_mass_to_dimensionless(1e-12_f64), 0.0]);

    let mut masses: Array1<f64> = Array1::from_elem(tov_iters, f64::NAN);
    let mut radii: Array1<f64> = Array1::from_elem(tov_iters, f64::NAN);
    let mut lamdas: Array1<f64> = Array1::from_elem(tov_iters, f64::NAN);
    let mut press_core: Array1<f64> = Array1::from_elem(tov_iters, f64::NAN);
    
    let max_radius = tov.factors.scale_radius_to_dimensioneless(100.0);


    for (i, &p_core_value) in p_core.iter().enumerate(){
        init_cond[1] = p_core_value;

        let (mut r_vals, y_vals) =
            solve(derivatives_fx,
                tov.factors.scale_radius_to_dimensioneless(1.0e-5),
                init_cond.to_owned(),
                max_radius,
                rtol,
                atol,
                0,
                0.0,
                Some(Box::new(stop_fx))
            )?;

        let m: f64 = y_vals.column(0).last().expect("Could not retrieve last mass from ODE solution.").clone();
        let m_sol: f64 =  convert_mass_natural_to_solar(m);
        if (i!= 0) && (m_sol<masses[i-1]){
            break;
        }

        let r = r_vals.last().expect("Could not retrieve last radius from ODE solver.").clone(); // this r is in km

        let mut m_profile = y_vals.column(0).mapv(|x|convert_mass_natural_to_solar(x));
        let mut p_profile = y_vals.column(1).mapv(|x|(tov.factors.restore_pressure_to_nuclear(x)));
        let mut e_profile = e_array.mapv(|x|(tov.factors.restore_pressure_to_nuclear(x)));

        let mut tidal = TovTidal::initiate(&mut r_vals, &mut m_profile, &mut p_profile, &mut e_profile, &cs2_array, &tov);

        let init_cond_tide: Array1<f64> = Array1::from(vec![(r_vals[0].powi(2)) , 2.0*(r_vals[0])]);

        let r_diff = tidal.r_diff.clone();

        let derivatives_tide = |r: f64, y: &Array1<f64>| -> Array1<f64> {
            tidal.tidal_deriv(r, y)};


        let (_r_vals_tide, y_vals_tide ) =
            solve_fixed_step(derivatives_tide,
                r_vals[0],
                init_cond_tide,
                r_vals.last()
                .unwrap_or(&max_radius)
                .clone(),
                0,
                &ArrayOrScalar::Array(r_diff),
                None
                )?;

        let final_state_tidal = y_vals_tide.index_axis(Axis(0), y_vals_tide.len_of(Axis(0)) - 1);

        let h: f64 = final_state_tidal[0];
        let beta = final_state_tidal[1];
        let y = r_vals.last().expect("some error") * beta / h; // the r here is in cm

        let lam = compute_tidal_deformability(m, r, y);

        masses[i] = m_sol;
        radii[i] = r;
        lamdas[i] = lam;
        press_core[i] = tov.factors.restore_pressure_to_nuclear(p_core_value);
    }
    Ok( (masses, radii, lamdas, press_core) )

}



pub(crate) fn get_mry_single_pcore(
    e_array: &mut Array1<f64>,
    p_array: &mut Array1<f64>,
    cs2_array: &Array1<f64>,
    p_core:f64,
    rtol: &ArrayOrScalar,
    atol: &ArrayOrScalar)
    -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), Box<dyn Error> >
    {

    let tov = TovMR::initiate(e_array, p_array);
    let derivatives_fx = |r: f64, y: &Array1<f64>| -> Array1<f64> {
        tov.tov_derivs(r, y)
    };
    let stop_fx = |x: f64, y_next: &Array1<f64>| -> bool {
        tov.tov_stopper(x, y_next)
    };

    let init_cond = Array1::from(vec![(1e-10_f64),tov.factors.scale_presssure_to_dimensionless(p_core)]);

    let max_radius = tov.factors.scale_radius_to_dimensioneless(100.0);


    let (mut r_vals, y_vals) =
        solve(derivatives_fx,
            tov.factors.scale_radius_to_dimensioneless(1.0e-2),
            init_cond.to_owned(),
            max_radius,
            rtol,
            atol,
            0,
            0.0,
            Some(Box::new(stop_fx))
        )?;

    let mut m_profile = y_vals.column(0).mapv(|x|convert_mass_natural_to_solar(x));
    let mut p_profile = y_vals.column(1).mapv(|x|(tov.factors.restore_pressure_to_nuclear(x)));
    let mut e_profile = e_array.mapv(|x|(tov.factors.restore_pressure_to_nuclear(x)));

    let mass = m_profile.mapv(|x| tov.factors.restore_mass_to_solar(x));
    let radius = r_vals.mapv(|x| tov.factors.restore_radius_to_natural(x));
    let pressure= p_profile.mapv(|x| convert_eos_natural_to_nuclear(x));

    let mut tidal = TovTidal::initiate(&mut r_vals, &mut m_profile, &mut p_profile, &mut e_profile, &cs2_array, &tov);

    let init_cond_tide: Array1<f64> = Array1::from(vec![(r_vals[0].powi(2)) , (r_vals[0])]);

    let r_diff = tidal.r_diff.clone();

    let derivatives_tide = |r: f64, y: &Array1<f64>| -> Array1<f64> {
        tidal.tidal_deriv(r, y)};


    let (_r_vals_tide, _y_vals_tide ) =
        solve_fixed_step(derivatives_tide,
            r_vals[0],
            init_cond_tide,
            r_vals.last()
            .unwrap_or(&max_radius)
            .clone(),
            0,
            &ArrayOrScalar::Array(r_diff),
            None
            )?;

    let _r = r_vals.mapv(|x| tidal.conversion_to_cgs.convert_radius_cgs_to_nat(x));

    let h =  y_vals.column(0).to_owned();
    let b =  y_vals.column(1).to_owned();


    let y_profile =  &b * &r_vals / &h;

    Ok( (mass, radius, y_profile, pressure) )
}
