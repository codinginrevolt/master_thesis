/*
Sources used:
Solving Ordinary Differential Equations I (henceforth) Vol I): https://doi.org/10.1007/978-3-540-78862-1
Solving Ordinary Differential Equations II (henceforth Vol II): https://doi.org/10.1007/978-3-642-05221-7
Numerical Methods in Engineering with Python 3 (henceforth PyImp): https://doi.org/10.1017/CBO9781139523899

TODO: enable integration for decreasing x
TODO: write documentation
*/

mod dp_coeffs;
mod dp_helpers;

use dp_coeffs::DormandPrinceCoeffs;
use dp_helpers::*;

use ndarray::{Array1, Array2, Axis, concatenate};
use std::error::Error;
use std::ops::Mul;


pub enum ArrayOrScalar {
    Array(Array1<f64>),
    Scalar(f64),
}
impl<'a> Mul<&'a ArrayOrScalar> for f64 {
    type Output = ArrayOrScalar;

    fn mul(self, rhs: &'a ArrayOrScalar) -> ArrayOrScalar {
        match rhs {
            ArrayOrScalar::Array(arr) => ArrayOrScalar::Array(self * arr),
            ArrayOrScalar::Scalar(val) => ArrayOrScalar::Scalar(self * val),
        }
    }
}

pub fn solve<'a, F>(mut f: F,
                     mut x: f64, 
                     mut y: Array1<f64>, 
                     x_stop: f64, 
                     rtol: &ArrayOrScalar, 
                     atol: &ArrayOrScalar, 
                     max_iter: u32, 
                     h:f64,
                     custom_stop: Option<Box<dyn for<'b> Fn(f64, &'b Array1<f64>) -> bool + 'a>>)
                    -> Result<(Array1<f64>, Array2<f64>), Box<dyn Error> >
where
    F: FnMut(f64, &Array1<f64>) -> Array1<f64>,
{
    /* 
    Inputs:
    f is a function defining the ODE
    x is the initial x value
    y is an array containing initial values at x
    x_stop when integration should stop
    rtol is relative tolerance 
    atol is absolute tolerance
    max_iter is maximum iterations for integrator to run, set to '0' for default val
    h is initial step size, set to '0.0' for automatic initial step size estimate
    custom_stop is available as a function (closure really) to stop integrating based on a condition.
    if no custom stop is needed, pass None 
    
    Output:
    (x_value, y_value) returned as a tuple
     */
    // setting up

    // custom_stop always available but in case of None input, custom_stop always returns false.
    let stop_fn = custom_stop.unwrap_or_else(|| Box::new(default_stop_fn));

    let max_iter = if max_iter == 0 { 100000 } else { max_iter };
    let mut iteration: u32 = 0;
    let coeffs: DormandPrinceCoeffs = DormandPrinceCoeffs::new();
    let atol = convert_arrayorscalar(atol, y.len());
    let rtol = convert_arrayorscalar(rtol, y.len());

    let h_min: f64 = 10.0 * f64::EPSILON * x.abs();
    let p: f64 = 5.0; // vol II pg 29. DOPRI5 has p=5 eq a pg 273 PyImp
    let alpha:f64 = 0.7/p; // eq 2.48 vol II pg 31
    let beta:f64 = 0.4/p;
    let mut dont_stop:bool = true;
    
    let mut x_vals = Array1::from_elem(1, x);
    let mut y_vals = Array2::from_elem((1, y.len()), 0.0);
    y_vals.row_mut(0).assign(&y);



    let mut dy = Array1::<f64>::zeros(y.len());
    let mut y_next = Array1::<f64>::zeros(y.len());

    let fx_eval= f(x, &y);
    let mut h: f64 = if h == 0.0 {
        initial_h(x, &y, &fx_eval, &mut f, &atol, &rtol)?
    } else {
        h
    };

    let mut k = init_stages(y.len());
    k.0.assign(&(h * &fx_eval));

    let mut sc: Array1<f64> = Array1::<f64>::zeros(y.len());
    let mut capital_e: Array1<f64> = Array1::<f64>::zeros(y.len());
    let mut err:f64 = 0.0;
    let mut err_old: f64 = 1e-4_f64;
    let mut err_inv:f64 = 0.0;

    // The juice:
    while dont_stop {

        update_stages(x, &y, h, &coeffs, &mut k, &mut f);
        compute_next_func_value(&y, &mut dy, &mut y_next, &k, &coeffs);

        sc.assign(&compute_sci(&y, &y_next, &atol, &rtol));

        let _ = compute_error(&mut capital_e, &mut err, &mut err_inv, &mut err_old, &sc, &k, &coeffs, alpha, beta);
        
        let h_next: f64;

        // Accepting/Rejecting step
        if err<=1.0{

            if stop_fn(x, &y_next) {
                break;
            } // checking at top instead of bottom so if condition met, the last returned value will not be invalid

            h_next = h * 5.0_f64.min(0.1_f64.max(0.9*err_inv)); //eq 4.13 pg 168 vol I
            k.0 = &k.6 * h_next/h; // eq c pg 274 in PyImp 

            let x_next = x + h;

            if x_next>x_stop{
                h = x_stop - x;
                x = x_stop;
                dont_stop = false;
            }
            else {
                x = x_next;
                h = h_next;
            }
            
            y.assign(&y_next);


            let new_x = Array1::from_elem(1, x);
            x_vals = concatenate(Axis(0), &[x_vals.view(), new_x.view()])
            .map_err(|e| format!("Concatenation failed. Next X computed at step either empty or overflow error: {:?}", e))?;

            y_vals.push_row(y_next.view())
            .map_err(|e| format!("Concatenation failed. Y computed at current step either empty or overflow error: {:?}", e))?;
                    }
        else {
            h_next = h * 1.0_f64.min(0.1_f64.max(0.9*err_inv)); // eq 4.13 vol I, but using the max 1 increase in step
            k.0 = &k.0 * h_next/h; // eq d pg 274 in PyImp
            h = h_next;
        }

        if iteration>=max_iter{
            println!("Maximum iterations reached.");
            break};

        if h.abs() < h_min {
            println!("Step size underflow. The step size has become too small to continue.");
            break;
        }

        iteration += 1;
    }
    Ok((x_vals, y_vals))
}

#[allow(dead_code)]
pub fn solve_fixed_step<'a, F>
    (mut f: F,
    mut x: f64, 
    mut y: Array1<f64>, 
    x_stop: f64, 
    max_iter: usize, 
    step:&ArrayOrScalar,
    custom_stop: Option<Box<dyn for<'b> Fn(f64, &'b Array1<f64>) -> bool + 'a>>)
    -> Result<(Array1<f64>, Array2<f64>), Box<dyn Error>>
where
F: FnMut(f64, &Array1<f64>) -> Array1<f64>,
{
    let max_iter:usize = if max_iter == 0 { 100000 } else { max_iter };
    let mut iteration: usize = 0;
    let coeffs: DormandPrinceCoeffs = DormandPrinceCoeffs::new();

    let mut dont_stop:bool = true;
    
    let mut x_vals = Array1::from_elem(1, x);
    let mut y_vals = Array2::from_elem((1, y.len()), 0.0);
    y_vals.row_mut(0).assign(&y);



    let mut dy = Array1::<f64>::zeros(y.len());
    let mut y_next = Array1::<f64>::zeros(y.len());

    let fx_eval= f(x, &y);
    
    let mut h = match step{
        ArrayOrScalar::Scalar(h) => *h,
        ArrayOrScalar::Array(h_array) => h_array[iteration],
    };

    let mut k = init_stages(y.len());

    k.0.assign(&(h * &fx_eval));

    let max_step = match step {
        ArrayOrScalar::Scalar(_h) => {
            // If h is an array, set to None to indicate that the array will be used within the loop
            None
        }
        ArrayOrScalar::Array(h_array) => {
            Some(h_array.len()-1)
        }
    };

    // custom_stop always available but in case of None input, custom_stop always returns false.
    let stop_fn = custom_stop.unwrap_or_else(|| Box::new(default_stop_fn));

    while dont_stop {

        update_stages(x, &y, h, &coeffs, &mut k, &mut f);
        compute_next_func_value(&y, &mut dy, &mut y_next, &k, &coeffs);

        // all steps accepted for this

        k.0.assign(&k.6);
        
        let x_next = x + h;

        if x_next>x_stop{
            h = x_stop - x;
            x = x_stop;
            dont_stop = false;
        }
        else {
            x = x_next;
        }

        y.assign(&y_next);

        let new_x = Array1::from_elem(1, x);
        x_vals = concatenate(Axis(0), &[x_vals.view(), new_x.view()])
        .map_err(|e| format!("Concatenation failed. Next X computed at step either empty or overflow error: {:?}", e))?;

        y_vals.push_row(y_next.view())
        .map_err(|e| format!("Concatenation failed. Y computed at current step either empty or overflow error: {:?}", e))?;
        
        
        if iteration>=max_iter{
            println!("Maximum iterations reached.");
            break};


        if stop_fn(x, &y_next) {
            break;
        }

        if let Some(max_step) = max_step {
            if iteration >= max_step {
                break;
            }
        }

        iteration += 1;

        if dont_stop==true{
            h = match step{
                ArrayOrScalar::Scalar(h) => *h,
                ArrayOrScalar::Array(h_array) => h_array[iteration],
            };
        }
    }

    Ok((x_vals, y_vals))

}
