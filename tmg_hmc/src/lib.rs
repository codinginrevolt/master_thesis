mod hmc;

extern crate rand;
extern crate rand_distr;
extern crate ndarray;

use ndarray::ShapeError;
use rand::Rng;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyTuple;
use pyo3::types::PyList;
use ndarray::{ArrayView1, ArrayView2, Array2};
use hmc::*;


/// Returns a sample from the HMC sampler.
/// Assumes canonical form, c.f DOI 10.1080/10618600.2013.788448
/// Parameters
/// ----------
/// n : int
///     The number of samples to draw.
/// initial : numpy.ndarray
///     The initial vector.
/// numlin : int
///     Number of linear constraints.
/// seed : int or None
///     The random seed.
/// f : numpy.ndarray or None
///     Matrix for constraints.
/// g : numpy.ndarray or None
///     Vector for constraints.
/// return_trace : bool
///     Whether to return the trace.
///
/// Returns
/// -------
/// numpy.ndarray
///     The samples. 
/// If `return_trace` is True, returns a tuple (samples, traces).
#[pyfunction]
fn sample<'py>(py: Python<'py>,
                n: usize,
                initial: PyReadonlyArray1<f64>,
                numlin: usize,  
                seed: Option<u64>,
                f: PyReadonlyArray2<f64>,
                g: PyReadonlyArray1<f64>,
                return_trace: bool
            ) -> PyResult<PyObject> {
                let rng_seed = seed.unwrap_or_else(|| {
                    let mut rng = rand::rng();
                    rng.random()
                });

                let initial_v = initial.as_array();
                let f_m = f.as_array();
                let g_v = g.as_array();


                let outcome = sample_hmc(n, rng_seed, initial_v, numlin, f_m, g_v, return_trace)
                        .map_err(|e| PyValueError::new_err(format!("Sampling error: {e}")))?;

                match outcome{
                    Rezz::SampleOnly(samples)=>{
                        let samples_py = PyArray2::from_array(py, &samples);
                        Ok(samples_py.into())
                    },
                    Rezz::SampleTrace(samples, traces) => {
                        let samples_py = PyArray2::from_array(py, &samples);
                        let traces_py: Bound<'_, PyList> = PyList::new(py, 
                            traces.iter()
                            .map(|mat| 
                                PyArray2::from_array(py, mat))
                            )?;

                        let samples_obj = samples_py.extract::<PyObject>()?;
                        let traces_obj = traces_py.extract::<PyObject>()?;
                        
                        let tuple = PyTuple::new(py, &[samples_obj, traces_obj])?;

                        Ok(tuple.into()) 
                    }
            }
}

/// This module provides exact HMC sampling for truncated multivariate Gaussians constrained linearly. 
/// Refer to https://doi.org/10.1080/10618600.2013.788448 for the background
/// This is a Python/Rust translation of the R package "tmg — Truncated Multivariate Gaussian Sampling"
#[pymodule]
fn tmg_hmc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    Ok(())
}

fn sample_hmc<'a>(
    n: usize,
    seed: u64,
    initial: ArrayView1<'a, f64>,
    numlin: usize,
    f: ArrayView2<'a, f64>,
    g: ArrayView1<'a, f64>,
    return_trace: bool,
) -> Result<Rezz, ShapeError> {

    let dim = initial.len();
    let mut hmc1 = HmcSampler::new(dim, seed);

    for i in 0..numlin{
        hmc1.add_linear_constraint(f.row(i), g[i]);
    }
    hmc1.set_initial_value(initial);

    let mut samples = Array2::<f64>::zeros((n, dim));
    let mut traces: Vec<Array2<f64>> = Vec::new();

    for i in 0..n{
        let hmc_return = hmc1.sample_next(return_trace)?;

        match hmc_return{
            HmcResult::Sample(dvec) => {
                samples.row_mut(i).assign(dvec);
            },
            HmcResult::Trace(dmat) => {
                let last_row_idx = dmat.nrows() - 1;
                let last_row = dmat.row(last_row_idx);
                samples.row_mut(i).assign(&last_row);
                traces.push(dmat);   
            }

        }
    }
    match return_trace{
        true => return Ok(Rezz::SampleTrace(samples, traces)),
        false => return Ok(Rezz::SampleOnly(samples)),
    }
}

enum Rezz {
    SampleOnly(Array2<f64>),
    SampleTrace(Array2<f64>, Vec<Array2<f64>>),
}

