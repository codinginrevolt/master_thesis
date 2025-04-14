use ndarray::{Array1, Zip};
use crate::dorpi5::dp_coeffs::DormandPrinceCoeffs;
use crate::dorpi5::ArrayOrScalar;
use std::error::Error;

pub(crate) fn init_stages(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>){
    let k0 = Array1::<f64>::zeros(n);
    let k1 = Array1::<f64>::zeros(n);
    let k2 = Array1::<f64>::zeros(n);
    let k3 = Array1::<f64>::zeros(n);
    let k4 = Array1::<f64>::zeros(n);
    let k5 = Array1::<f64>::zeros(n);
    let k6 = Array1::<f64>::zeros(n);

    let k = (k0,k1,k2,k3,k4,k5,k6);

    k
}

pub(crate) fn update_stages<F>(x: f64, y: &Array1<f64>, h: f64, coeffs: &DormandPrinceCoeffs, k: &mut (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), mut f: F)
where
F: FnMut(f64, &Array1<f64>) -> Array1<f64>,
{
    // k assignments translated from pg 275 PyImp

    k.1.assign( &(h * f(x + coeffs.a1 * h, &(y + coeffs.b10 * &k.0) ))) ;
    k.2.assign( &(h * &f(x + coeffs.a2 * h, &(y + coeffs.b20 * &k.0 + coeffs.b21 * &k.1))));
    k.3.assign( &(h * &f(x + coeffs.a3 * h, &(y  + coeffs.b30 * &k.0 + coeffs.b31 * &k.1 + coeffs.b32 * &k.2))) );
    k.4.assign( &(h * &f(x + coeffs.a4 * h, &(y + coeffs.b40 * &k.0 + coeffs.b41 * &k.1 + coeffs.b42 * &k.2 + coeffs.b43 * &k.3))) );
    k.5.assign( &(h * &f(x + coeffs.a5 * h, &(y + coeffs.b50 * &k.0 + coeffs.b51 * &k.1 + coeffs.b52 * &k.2 + coeffs.b53 * &k.3 + coeffs.b54 * &k.4))));
    k.6.assign( &(h * &f(x + coeffs.a6 * h, &(y + coeffs.b60 * &k.0 + coeffs.b62 * &k.2 + coeffs.b63 * &k.3 + coeffs.b64 * &k.4 + coeffs.b65 * &k.5))) );
}

pub(crate) fn compute_error(capital_e: &mut Array1<f64>, err: &mut f64, err_inv: &mut f64, err_old: &mut f64, sc: &Array1<f64>, k: &(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), coeffs: &DormandPrinceCoeffs, alpha: f64, beta: f64)-> Result<(), Box<dyn Error>>{
    // capital_e assignments translated from pg 275 PyImp, in vol I&II terms, capital_e is difference in 4th and 5th order estimation (y - y_hat)
    capital_e.assign( &((coeffs.c0 - coeffs.d0) * &k.0 + (coeffs.c2 - coeffs.d2) * &k.2 + 
    (coeffs.c3 - coeffs.d3) * &k.3 + (coeffs.c4 - coeffs.d4) * &k.4 + 
    (coeffs.c5 - coeffs.d5) * &k.5 - coeffs.d6 * &k.6));

    // eq 4.11 pg 168 vol I
    *err = (((&*capital_e/sc) * (&*capital_e/sc))
            .mean()
            .ok_or("Error in step error estimation in DORPI5")?)
            .sqrt();

    // below: 2.43c pg29 vol II, TOL=1 since scale version of eq 4.11 pg 168 vol I
    *err_inv = (1.0/err.abs()).powf(alpha) * (err_old.abs()).powf(beta); 
    *err_old = *err;
    Ok(())
}


pub(crate) fn compute_next_func_value(y: &Array1<f64>, dy: &mut Array1<f64>, y_next: &mut Array1<f64>, k: &(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), coeffs: &DormandPrinceCoeffs){
    dy.assign(  &(coeffs.c0 * &k.0 + coeffs.c2 * &k.2 + coeffs.c3 * &k.3 + coeffs.c4 * &k.4 + coeffs.c5 * &k.5) );
    y_next.assign(&(y + &*dy));
}

pub(crate) fn initial_h<F>(x:f64, y: &Array1<f64>, f0: &Array1<f64>, mut fx:F, atol: &Array1<f64>, rtol: &Array1<f64>) -> Result<f64, Box<dyn Error>>
where
    F: FnMut(f64, &Array1<f64>) -> Array1<f64>,
{
    /*
    Implementing automatic initial step estimator 
    Algorithm from pg 169 of vol I 
     */

    // step a
    let sc_y0:Array1<f64> = compute_sci(y, y, atol, rtol);
    let sc_f0:Array1<f64> = compute_sci(f0, f0, atol, rtol);


    let d0 = ((y/&sc_y0 * y/&sc_y0)
                .mean()
                .ok_or("Error in getting mean for d0 in DOPRI5")?)
                .sqrt();
    let d1: f64 = ((f0/&sc_f0 * f0/&sc_f0)
                .mean()
                .ok_or("Error in getting mean for d1 in DOPRI5")?)
                .sqrt();

    // step b
    let h0 = 
    if d0 < 10e-5 || d1 < 10e-5 {10e-6} 
    else {0.01 * (d0 / d1)};

    // step c
    let y1:Array1<f64> = y + &(h0 * f0);

    // step d
    let f1: Array1<f64> = fx(x+h0, &y1);
    let diff = &f1 - f0;
    let sc_f1:Array1<f64> =  compute_sci(&diff, &diff, atol, rtol);

    let d2 = ((&diff/&sc_f1 * &diff/&sc_f1)
                .mean()
                .ok_or("Error in getting mean for d2 in DOPRI5")?)
                .sqrt() / h0;

    // step e
    let h1 = 
    if d1.max(d2)<=10e-15 {10e-6_f64.max(h0 * 10e-3)}
    else {(0.01 / (d1.max(d2))).powf(0.2)}; // p = 4 in the case of DOPRI5 since O(4) for err

    // step f
    let h:f64 = h1.min(100.0 * h0);


    Ok(h)
}


pub(crate) fn compute_sci(y0: &Array1<f64>, y1: &Array1<f64>, atol: &Array1<f64>, rtol: &Array1<f64>) -> Array1<f64> {
    // eq 4.10 pg 167 vol I

    // Compute the element-wise maximum of the absolute values of y0 and y1
    let max_abs_y = Zip::from(y0).and(y1).map_collect(|&a, &b| a.abs().max(b.abs()));

    // Compute atol + max_abs_y * rtol
    atol + &(max_abs_y * rtol)
}

pub(crate) fn convert_arrayorscalar(input: &ArrayOrScalar, len: usize) -> Array1<f64>{
    match input {
        ArrayOrScalar::Scalar(val) => Array1::from_elem(len, *val),
        ArrayOrScalar::Array(val) => val.to_owned()
    }
}

pub(crate) fn default_stop_fn(_: f64, _: &Array1<f64>) -> bool {
    false
}

