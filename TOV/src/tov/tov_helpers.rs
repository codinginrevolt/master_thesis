use std::f64::consts::PI;

const MEVFM3_TO_KM2: f64 = 1.3234e-6;
const MSOL_TO_KM: f64 = 1.4766;

pub const fn convert_eos_nuclear_to_natural(x:f64) -> f64 {
    x * MEVFM3_TO_KM2
}
pub const fn convert_eos_natural_to_nuclear(x:f64) -> f64 {
    x / MEVFM3_TO_KM2
}

pub const fn convert_mass_natural_to_solar(x: f64) -> f64 {
    x / MSOL_TO_KM
}

pub const P0_MEVFM3: f64 = 1.0e-6; // MeV/fm3

pub struct Factors{
    pub r0: f64,
    pub m0: f64,
    pub p0_mevfm3: f64,
    pub e0_mevfm3: f64,
    pub m0r0_ratio: f64,
    pub p0e0_ratio: f64,
    pub four_pi_e_r2: f64,
    pub four_pi_p_r2: f64,


}
impl Factors{
    pub fn initiate(r0: f64, m0r0_ratio: f64, p0_mevfm3: f64) -> Self{
        let m0: f64 = m0r0_ratio * r0;
        let p0: f64 = convert_eos_nuclear_to_natural(p0_mevfm3);
        let e0:f64 = p0;
        let p0e0_ratio:f64 = p0/e0; 
        let e0_mevfm3: f64 = convert_eos_natural_to_nuclear(e0);
        let four_pi_e_r2: f64 = 4.0 * PI * e0 * r0 * r0;
        let four_pi_p_r2: f64 = 4.0 * PI * p0 * r0 * r0;

        Factors {
            r0,
            m0,
            p0_mevfm3,
            e0_mevfm3,
            m0r0_ratio, 
            p0e0_ratio, 
            four_pi_e_r2, 
            four_pi_p_r2,
            }
    }


    pub fn scale_edens_to_dimensionless(&self, e: f64) -> f64 {
        // from nuclear units
        e/self.e0_mevfm3
    }
    
    pub fn scale_presssure_to_dimensionless(&self, p: f64) -> f64 {
        // from nuclear units
        p/self.p0_mevfm3
    }
    
    pub fn scale_radius_to_dimensioneless(&self, r: f64) -> f64 {
        r/self.r0
    }
    
    
    pub fn scale_mass_to_dimensionless(&self, m: f64) -> f64 {
        // from natural units
        m/self.m0
    }
    
    pub fn restore_mass_to_natural(&self, m: f64) -> f64 {
        m * self.m0 //resulting mass in km
    }
    
    pub fn restore_mass_to_solar(&self, m: f64) -> f64 {
        convert_mass_natural_to_solar(self.restore_mass_to_natural(m))
    }
    
    pub fn restore_radius_to_natural(&self, x: f64) -> f64 {
        x * self.r0
    }
    
    pub fn restore_pressure_to_nuclear(&self, p: f64) -> f64{
        p*self.p0_mevfm3
    }

}

pub struct  ConversionToCGS{
    pub g_cgs: f64, // erg.cm/g²
    pub c_cgs: f64, // cm/s²
    eos_nuclear_to_cgs : f64, // erg/cm³
    r_km_to_cm: f64, // cm
    m_msol_to_g: f64, // g
    eos0: f64,
}
impl ConversionToCGS{
    pub fn initiate(eos0_cgs: f64) -> Self {
        let g_cgs = 6.67408e-8;
        let c_cgs: f64 = 2.99792458e10;
        let eos_nuclear_to_cgs= 1.6012766e33;
        let r_km_to_cm = 1.0e5;
        let m_msol_to_g = 1.988435e33;
        let eos0 = eos0_cgs;

        ConversionToCGS{ g_cgs, c_cgs, eos_nuclear_to_cgs, r_km_to_cm, m_msol_to_g, eos0}
    }

    pub fn convert_mass_msol_to_cgs(&self, x:f64) -> f64{
        x * self.m_msol_to_g
    }

    pub fn convert_radius_nat_to_cgs(&self, x:f64) -> f64 {
        x * self.r_km_to_cm
    }

    pub fn convert_eos_nuc_to_cgs(&self, x:f64) -> f64 {
        x * self.eos_nuclear_to_cgs / 10.0
    }

    pub fn convert_radius_cgs_to_nat(&self, x:f64)->f64{
        x/self.r_km_to_cm
    }

    pub fn scale_eos_nuc_to_cgs(&self, x:f64) -> f64 {
        self.convert_eos_nuc_to_cgs(x)/self.eos0
    }
}