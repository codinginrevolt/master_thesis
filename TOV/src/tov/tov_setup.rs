use crate::tov::tov_helpers::*;

use splines::{Interpolation, Key, Spline};
use ndarray::Array1;

use std::f64::consts::PI;


pub struct TovTide {
    eden_spline: Spline<f64, f64>,
    pub cs2_spline: Spline<f64, f64>,
}

impl <'a> TovTide {
    pub fn initiate(e_array: &'a mut Array1<f64>, p_array: &'a mut Array1<f64>,  cs2: &'a Array1<f64>) -> Self {
        e_array.mapv_inplace(|x| convert_eos_nuclear_to_natural(x));
        p_array.mapv_inplace(|x| convert_eos_nuclear_to_natural(x));

        let eden_spline: Spline<f64, f64> = Spline::from_iter(
            p_array.iter().cloned().zip(e_array.iter().cloned()).map(|(p, e)| Key::new(p, e, Interpolation::Linear)),
        );

        let cs2_spline: Spline<f64, f64> = Spline::from_iter(
            p_array.iter().cloned().zip(cs2.iter().cloned()).map(|(p, cs2)| Key::new(p, cs2, Interpolation::Linear)),
        );   

        TovTide {eden_spline, cs2_spline}
    }

    pub fn tov_derivs(&self, x: f64, f: &Array1<f64>) -> Array1<f64> {
        let m = f[0];
        let p = f[1];

        let eden: f64 = self.eden_spline.clamped_sample(p).expect("Expected interpolated eden as f64, got None");

        let dmdr = 4.0 * PI * x.powi(2) * eden;
        let dpdr =  - (p+eden)*(m + (4.0 * PI * x.powi(3) * p)) / (x*(x-(2.0*m)));

        return Array1::from_vec(vec![dmdr, dpdr]);
    }


    pub fn tidal_derivs(&self, x: f64, f: &Array1<f64>) -> Array1<f64> {
        let m = f[0];
        let p = f[1];
        let y =  f[2];

        let eden: f64 = self.eden_spline.clamped_sample(p).expect("Expected interpolated eden as f64, got None");
        let cs2: f64 = self.cs2_spline.clamped_sample(p).expect("Expected interpolated cs2 as f64, got None");

        let dmdr = 4.0 * PI * x.powi(2) * eden;
        let dpdr =  - (p+eden)*(m + (4.0 * PI * x.powi(3) * p)) / (x*(x-(2.0*m)));

        let dydr1 = (4.0 *  (m + 4.0 * PI * x.powi(3) * p).powi(2)) / (x * (x - 2.0*m).powi(2));
        let dydr2 = 6.0 / (x - 2.0*m);
        let dydr3 = y.powi(2)/x;
        let dydr4 = y * (x + (4.0*PI*x.powi(3) * (p-eden) )) / (x * (x - 2.0*m));
        let dydr5 = (4.0*PI*x.powi(2))/(x-2.0*m) 
                    * (5.0*eden + 9.0*p + ((eden+p)/cs2));
        let dydr = dydr1 + dydr2 - dydr3 - dydr4 - dydr5;

        return Array1::from_vec(vec![dmdr, dpdr, dydr]);
    }


    pub fn tov_stopper(&self, _x: f64, y_next: &Array1<f64>) -> bool {
        // stop at surface
        y_next[1] < 0.0
    }
}


pub fn compute_tidal_deformability(m:f64, r:f64, y:f64) -> f64{
    let c = m/r; 

    let first_factor = (1.0 - 2.0*c)*(1.0 - 2.0*c);
    let second_factor = 2.0 + (2.0 * c * (y-1.0)) - y;

    let first_term = 2.0 * c * (6.0 - (3.0*y) + (3.0*c * (5.0*y - 8.0)) );
    let second_term = 4.0 * (c*c*c) * (13.0 - (11.0*y) + (c * (3.0*y - 2.0)) + (2.0 * (c*c) * (1.0 + y)) );
    let third_term = 3.0 * (1.0 - (2.0*c))*(1.0 - (2.0*c)) 
                    * (2.0 - y + (2.0*c * (y - 1.0)))
                    * ((1.0 - (2.0*c) ).ln());

    let lam = (16.0/15.0) * first_factor * second_factor * (1.0/(first_term + second_term + third_term));

    return lam;
}

