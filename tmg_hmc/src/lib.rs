mod hmc;

extern crate nalgebra;
extern crate rand;
extern crate rand_distr;
extern crate ndarray;

use rand::Rng;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array2};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyTuple;
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
                f: Option<PyReadonlyArray2<f64>>,
                g: Option<PyReadonlyArray1<f64>>,
                return_trace: bool
            ) -> PyResult<PyObject> {

                let rng_seed = seed.unwrap_or_else(|| {
                    let mut rng = rand::rng();
                    rng.random()
                });

                let initial_v = DVector::from_column_slice(initial.as_slice()?);
                
                let f_m = f.map(|arr|{
                    let arr = arr.as_array();
                    if !arr.is_standard_layout() {
                        return Err(PyValueError::new_err(
                            "Input array 'f' must be contiguous (C-order)",
                        ));
                    }
                    let slice = arr.as_slice().expect("Empty or non contiguous array f");
                    Ok(DMatrix::from_row_slice(arr.nrows(), arr.ncols(), slice))
                }).transpose()?; // transpose()? Turns Option<Result<T, Err>> to Option<T>

                let g_v = g.map(|arr|{
                    let arr = arr.as_array();
                    if !arr.is_standard_layout(){
                        return Err(PyValueError::new_err(
                            "Input array 'g' must be contiguous (C-order)",
                        ));
                    }
                    let slice = arr.as_slice().expect("Empty or non contiguous array g");
                    Ok(DVector::from_column_slice(slice))
                }).transpose()?;

                let outcome = sample_hmc(n, rng_seed, initial_v, numlin, f_m, g_v, return_trace);

                match outcome{
                    Rezz::SampleOnly(samples)=>{
                        let slice = samples.as_slice();
                        let arr = Array2::from_shape_vec(
                            (samples.ncols(), samples.nrows()),
                            slice.to_vec())
                            .map_err(|_e| PyValueError::new_err("Samples matrix was non-contiguous"))?
                            .t()
                            .to_owned();

                        let samples_py = PyArray2::from_array(py, &arr);
                        Ok(samples_py.into())
                    },
                    Rezz::SampleTrace(samples, traces) => {
                        let slice = samples.as_slice();
                        let arr = Array2::from_shape_vec(
                            (samples.ncols(), samples.nrows()),
                            slice.to_vec())
                            .map_err(|_e| PyValueError::new_err("Samples matrix was non-contiguous"))?
                            .t()
                            .to_owned();

                        let samples_py = PyArray2::from_array(py, &arr);
                        
                        let traces_vec: Result<Vec<_>, PyErr> = traces.iter()
                            .map(|mat| {
                            let slice = mat.as_slice();
                            Array2::from_shape_vec(
                                (mat.ncols(), mat.nrows()),
                                slice.to_vec()
                            )
                            .map_err(|_e| PyValueError::new_err("Trace matrices were non-contiguous"))
                            .map(|arr| arr.t().to_owned().into_pyarray(py))
                        }).collect();

                        let traces_vec = traces_vec?;
                        let traces_py = PyTuple::new(py, traces_vec)?;
                        
                        let samples_obj = samples_py.extract::<PyObject>()?;
                        let traces_obj = traces_py.extract::<PyObject>()?;
                        
                        let tuple = PyTuple::new(py, &[samples_obj, traces_obj])?;

                        Ok(tuple.into())
                    
                    }

            }

}

/// This module provides exact HMC sampling for truncated multivariate Gaussians constrained linearly. (Refer to https://doi.org/10.1080/10618600.2013.788448 for the background)
#[pymodule]
fn tmg_hmc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    Ok(())
}

fn sample_hmc(
    n: usize,
    seed: u64,
    initial: DVector<f64>,
    numlin: usize,
    f: Option<DMatrix<f64>>,
    g: Option<DVector<f64>>,
    return_trace: bool,
) -> Rezz {
    println!("Hi from Rust!");
    let dim = initial.len();

    let mut hmc1 = HmcSampler::new(dim, seed);

    if numlin>0 {
        let f = f.expect("F in (Fx-g>=0) needed for linear constraints");
        let g = g.expect("g in (Fx-g>=0) needed for linear constraints");

        for i in 0..numlin{
            hmc1.add_linear_constraint(f.row(i).transpose(), g[i]);
        }
    }

    hmc1.set_initial_value(initial);

    let mut samples = DMatrix::<f64>::zeros(n, dim);
    let mut traces: Vec<DMatrix<f64>> = Vec::new();

    for i in 0..n{
        let hmc_return = hmc1.sample_next(return_trace);

        match hmc_return{
            HmcResult::Sample(dvec) => {
                samples.set_row(i,  &dvec.transpose());
            },
            HmcResult::Trace(dmat) => {
                let last_row_idx = dmat.nrows() - 1;
                let last_row = dmat.row(last_row_idx);
                samples.set_row(i, &last_row);
                traces.push(dmat);   
            }

        }
    }
    match return_trace{
        true => return Rezz::SampleTrace(samples, traces),
        false => return Rezz::SampleOnly(samples),
    }
}

enum Rezz {
    SampleOnly(DMatrix<f64>),
    SampleTrace(DMatrix<f64>, Vec<DMatrix<f64>>),
}

