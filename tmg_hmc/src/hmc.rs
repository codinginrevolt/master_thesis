use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Normal, Distribution};
use std::f64::consts::PI;

pub struct LinearConstraint {
    pub f: DVector<f64>,
    pub g: f64,
}

pub struct HmcSampler {
    dim: usize,
    rng: StdRng,
    norm_dist: Normal<f64>,
    last_sample: DVector<f64>,
    linear_constraints: Vec<LinearConstraint>,
}


impl HmcSampler {
    pub fn new(dim: usize, seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        let norm_dist = Normal::new(0.0, 1.0).unwrap();
        Self {
            dim,
            rng,
            norm_dist,
            last_sample: DVector::zeros(dim),
            linear_constraints: Vec::new(),
        }
    }
    pub fn set_initial_value(&mut self, initial_value: DVector<f64>){
        self.last_sample = initial_value;
    }

    pub fn sample_next(&mut self, return_trace: bool) -> HmcResult {
        let mut trace_points = DMatrix::<f64>::zeros(self.dim, 0);
        
        let total_time = PI/2.0;
        let mut b = self.last_sample.clone();
        let mut a = DVector::<f64>::zeros(self.dim);

        loop {
            let mut velsign = 0.0;
            for i in 0..self.dim{
                a[i] = self.norm_dist.sample(&mut self.rng); // init_velcity
            }

            let mut time_left = total_time;

            let mut t1: f64;
            let mut cn1: usize;

            loop {
                
                (t1, cn1) = self.get_next_linear_time_hit(&a, &b);

                
                let t = t1;

                if (t == 0.0) | (time_left<t){ // no wall hits left or no time left
                    break;
                }
                else {
                    if return_trace{
                        self.update_trace(&a, &b, &t1, &mut trace_points);
                    }
                
                    time_left -= t;
                    let cos_t = t.cos();
                    let sin_t = t.sin();
                    let new_sample = sin_t*&a + cos_t*&b;
                    let hit_vel = cos_t*&a - sin_t*&b;
                    b = new_sample;

                    let ql = &self.linear_constraints[cn1];
                    let f2 = (ql.f).dot(&ql.f);
                    let alpha = (ql.f).dot(&hit_vel)/f2;
                    a = &hit_vel - 2.0*alpha*(&ql.f); // reflected velocity
                    velsign = a.dot(&ql.f);

                    if velsign<0.0 {break;}
                }
            }

            if velsign<0.0{continue;}

            let bb = time_left.sin()*&a + time_left.cos()*&b;

            let check: f64 = self.verify_constraints(&bb);
            if check>=0.0{
                self.last_sample=bb;
                
                if return_trace{
                    self.update_trace(&a, &b, &time_left, &mut trace_points);
                    return HmcResult::Trace(trace_points.transpose());
                }
                else {
                    return HmcResult::Sample(self.last_sample.clone());
                }
            }
        }
    }

    pub fn add_linear_constraint(&mut self, f: DVector<f64>, g: f64){
        let new_constraint = LinearConstraint{f,g};
        self.linear_constraints.push(new_constraint);
    }


    fn get_next_linear_time_hit(&self, a: &DVector<f64>, b: &DVector<f64>) -> (f64, usize){
        let mut hit_time = 0.0;
        let mut cn = 0;
        let min_t = 0.00001;

        for (i, lc) in self.linear_constraints.iter().enumerate(){
            let fa = (lc.f).dot(a);
            let fb = (lc.f).dot(b);
            let u = (fa*fa + fb*fb).sqrt();

            if (u>lc.g) && (u>-lc.g){
                let phi = -fa.atan2(fb);
                let mut t1 = (-lc.g/u).acos() - phi;

                if t1<0.0 {
                    t1 += 2.0*PI;
                }
                if t1.abs()  < min_t{
                    t1 = 0.0;
                }
                else if (t1-2.0*PI).abs() < min_t {
                    t1 = 0.0;
                }

                let mut t2 = -t1 - 2.0*phi;  // -4*pi < t2 < 3*pi
                if t2<0.0 {
                    t2 += 2.0*PI; // -2*pi < t2 < 2*pi
                }
                if t2<0.0 {
                    t2 += 2.0*PI; // 0 < t2 < 2*pi
                }

                if t2.abs() < min_t {
                    t2 = 0.0;
                }
                else if (t2 - 2.0*PI).abs() < min_t{
                    t2 = 0.0;
                }

                let t = if t1 == 0.0 {t2}
                             else if t2 == 0.0 {t1}
                             else {t1.min(t2)};

                if (t>min_t) && (hit_time == 0.0 || t < hit_time){
                    hit_time = t;
                    cn = i;
                }
            }
        }
        (hit_time, cn)
    }

    fn update_trace(&self, a: &DVector<f64>, b: &DVector<f64>, t: &f64, trace_points: &mut DMatrix<f64>){
        let step_size= 0.01;
        let steps = (t/step_size).floor() as usize;

        let c = trace_points.ncols();

        trace_points.resize_horizontally_mut(c + steps + 1, 0.0);

        for i in 0..steps{
            let ang = i as f64 *step_size;
            let bb = ang.sin()*a + ang.cos()*b;

            trace_points.set_column(c+1, &bb);
        }
        let bb = t.sin()*a + t.cos()*b;

        trace_points.set_column(c+1, &bb);


    }

    fn verify_constraints(&self, bb: &DVector<f64>) -> f64{

        let mut r = 0.0;

        for (i, lc) in self.linear_constraints.iter().enumerate(){
            let check = (lc.f).dot(bb) + lc.g;
            if (i==0) || (check<r) {
                r = check;
            }
        }
        r
    }

}

pub enum HmcResult {
    Sample(DVector<f64>),
    Trace(DMatrix<f64>),
}
