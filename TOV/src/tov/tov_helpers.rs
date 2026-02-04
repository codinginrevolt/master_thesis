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

pub const fn convert_mass_solar_to_natural(x: f64) -> f64 {
    x * MSOL_TO_KM
}
