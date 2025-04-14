pub(crate) struct DormandPrinceCoeffs {
    pub(crate) a1: f64, pub(crate) a2: f64, pub(crate) a3: f64, pub(crate) a4: f64, pub(crate) a5: f64, pub(crate) a6: f64,
    pub(crate) c0: f64, pub(crate) c2: f64, pub(crate) c3: f64, pub(crate) c4: f64, pub(crate) c5: f64,
    pub(crate) d0: f64, pub(crate) d2: f64, pub(crate) d3: f64, pub(crate) d4: f64, pub(crate) d5: f64, pub(crate) d6: f64,
    pub(crate) b10: f64, pub(crate) b20: f64, pub(crate) b21: f64,
    pub(crate) b30: f64, pub(crate) b31: f64, pub(crate) b32: f64,
    pub(crate) b40: f64, pub(crate) b41: f64, pub(crate) b42: f64, pub(crate) b43: f64,
    pub(crate) b50: f64, pub(crate) b51: f64, pub(crate) b52: f64, pub(crate) b53: f64, pub(crate) b54: f64,
    pub(crate) b60: f64, pub(crate) b62: f64, pub(crate) b63: f64, pub(crate) b64: f64, pub(crate) b65: f64,
}

impl DormandPrinceCoeffs {
    // Constructor to initialize the constants
    pub(crate) fn new() -> Self {
        DormandPrinceCoeffs {
            a1: 0.2, a2: 0.3, a3: 0.8, a4: 8.0 / 9.0, a5: 1.0, a6: 1.0,
            c0: 35.0 / 384.0, c2: 500.0 / 1113.0, c3: 125.0 / 192.0,
            c4: -2187.0 / 6784.0, c5: 11.0 / 84.0,
            d0: 5179.0 / 57600.0, d2: 7571.0 / 16695.0, d3: 393.0 / 640.0,
            d4: -92097.0 / 339200.0, d5: 187.0 / 2100.0, d6: 1.0 / 40.0,
            b10: 0.2, b20: 0.075, b21: 0.225,
            b30: 44.0 / 45.0, b31: -56.0 / 15.0, b32: 32.0 / 9.0,
            b40: 19372.0 / 6561.0, b41: -25360.0 / 2187.0, b42: 64448.0 / 6561.0,
            b43: -212.0 / 729.0,
            b50: 9017.0 / 3168.0, b51: -355.0 / 33.0, b52: 46732.0 / 5247.0,
            b53: 49.0 / 176.0, b54: -5103.0 / 18656.0,
            b60: 35.0 / 384.0, b62: 500.0 / 1113.0, b63: 125.0 / 192.0,
            b64: -2187.0 / 6784.0, b65: 11.0 / 84.0,
        }
    }
}
