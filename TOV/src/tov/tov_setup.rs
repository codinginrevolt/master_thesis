use crate::tov::tov_helpers::*;

use splines::{Interpolation, Key, Spline};
use ndarray::Array1;

use crate::dorpi5::NUM_STAGES_UPDATED; //modify this constant for number of stages updated by calling fx for the ode solver being used

use std::f64::consts::PI;

pub struct TovMR {
    eden_spline: Spline<f64, f64>,
    pub factors: Factors,
}

impl <'a> TovMR {
    pub fn initiate(e_array: &'a mut Array1<f64>, p_array: &'a mut Array1<f64>) -> Self {
        let factors= Factors::initiate(1.0, 1.0, P0_MEVFM3);

        e_array.mapv_inplace(|x| factors.scale_edens_to_dimensionless(x));
        p_array.mapv_inplace(|x|factors.scale_presssure_to_dimensionless(x));

        let eden_spline: Spline<f64, f64> = Spline::from_iter(
            p_array.iter().cloned().zip(e_array.iter().cloned()).map(|(p, e)| Key::new(p, e, Interpolation::Linear)),
        );

        TovMR {eden_spline, factors}
    }

    pub fn tov_derivs(&self, x: f64, f: &Array1<f64>) -> Array1<f64> {
        let m = f[0];
        let p = f[1];

        let eden: f64 = self.eden_spline.clamped_sample(p).expect("Expected interpolated eden as f64, got None");

        let dmdx = self.factors.four_pi_e_r2 * (x * x) * eden;

        let dpdx_factor1 = self.factors.m0r0_ratio / self.factors.p0e0_ratio;
        let dpdx_factor2 = self.factors.four_pi_p_r2 / self.factors.m0r0_ratio;

        let dpdx_term1 = eden + (self.factors.p0e0_ratio * p);
        let dpdx_term2 = m + (dpdx_factor2 * p * (x * x * x) );
        let dpdx_term3 = (x*x) * (1.0 - (2.0 * self.factors.m0r0_ratio * (m/x) ) );

        let dpdx = - dpdx_factor1 * (dpdx_term1 * dpdx_term2)/dpdx_term3;

        return Array1::from_vec(vec![dmdx, dpdx]);
    }


    pub fn tov_stopper(&self, _x: f64, y_next: &Array1<f64>) -> bool {
        // stop at surface
        y_next[1] < 0.0
}
}


pub struct TovTidal<'a>{
    pub r_diff: Array1<f64>,
    m: &'a Array1<f64>,
    p: &'a Array1<f64>,
    eden_spline: Spline<f64, f64>,
    cs2_spline: Spline<f64, f64>,
    pub conversion_to_cgs: ConversionToCGS,
    step: usize,
    substep: usize,
}
impl <'a>TovTidal<'a> {
    pub fn initiate (r: &mut Array1<f64>, m: &'a mut Array1<f64>,p: &'a mut Array1<f64>, e: &'a mut Array1<f64>, cs2: &'a Array1<f64>, _tov: &'a TovMR ) -> Self 
    {
        let conversion_to_cgs = ConversionToCGS::initiate(1.0e10);

        r.mapv_inplace(|x|conversion_to_cgs.convert_radius_nat_to_cgs(x));
        let r_diff: Array1<f64> = r.windows(2)
                                .into_iter()
                                .map(|w| w[1] - w[0])
                                .collect();

        m.mapv_inplace(|x| conversion_to_cgs.convert_mass_msol_to_cgs(x));

        p.mapv_inplace(|x|conversion_to_cgs.scale_eos_nuc_to_cgs(x));
        e.mapv_inplace(|x|conversion_to_cgs.scale_eos_nuc_to_cgs(x)); 

        let eden_spline: Spline<f64, f64> = Spline::from_iter(
            p.iter().cloned().zip(e.iter().cloned()).map(|(p, e)| Key::new(p, e, Interpolation::Linear)),
        );
        let cs2_spline: Spline<f64, f64> = Spline::from_iter(
            p.iter().cloned().zip(cs2.iter().cloned()).map(|(p, cs2)| Key::new(p, cs2, Interpolation::Linear)),
        );        
        let step = 0;
        let substep:usize = 0;

        TovTidal { r_diff, m, p, eden_spline, cs2_spline, conversion_to_cgs, step, substep}
    }
    

    pub fn tidal_deriv(&mut self, r: f64, f: &Array1<f64>) -> Array1<f64>{

        let h = f[0];
        let beta = f[1];
        let m: f64 = self.m[self.step];
        let p: f64 = self.p[self.step];

        if self.substep == NUM_STAGES_UPDATED{ 
            self.step += 1;
            self.substep = 0;
        } // keeps track of when a full step is done for the ode solver
        self.substep += 1;

        let eps: f64 = self.eden_spline.clamped_sample(p).expect("Expected interpolated eden as f64, got None");
        let dpde: f64 = self.cs2_spline.clamped_sample(p).expect("Expected interpolated eden as f64, got None");

        let dhdr = beta;

        let factor = 1.0 / (1.0 - 2.0 * self.conversion_to_cgs.g_csg * m / (self.conversion_to_cgs.c_cgs.powi(2) * r));

        let term1 = 2.0 * PI * 
            self.conversion_to_cgs.g_csg / self.conversion_to_cgs.c_cgs.powi(4) * (5.0 * eps + 9.0 * p
            + (eps + p) / dpde );
        
        let term2 = 3.0 / r.powi(2);
        
        let term3 = (
            (self.conversion_to_cgs.g_csg * m / (self.conversion_to_cgs.c_cgs.powi(2) * r.powi(2)))
            + 4.0 * PI * self.conversion_to_cgs.g_csg * r * p / self.conversion_to_cgs.c_cgs.powi(4)
        ).powi(2);
        
        let term4 = -1.0 + self.conversion_to_cgs.g_csg * m / (self.conversion_to_cgs.c_cgs.powi(2) * r)
            + 2.0 * PI * self.conversion_to_cgs.g_csg * r.powi(2) * (eps - p) / self.conversion_to_cgs.c_cgs.powi(4);
        
        let dbdr = 2.0 * factor * h * ( - term1 + term2 + 2.0 * factor * term3)
            + 2.0 * beta / r * factor * term4;
        
        return Array1::from_vec(vec![dhdr, dbdr]);

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