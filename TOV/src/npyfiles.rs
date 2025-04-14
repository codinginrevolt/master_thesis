use ndarray::{Array2, ArrayD, s, Array3};
use ndarray_npy::read_npy;
use ndarray_npy::write_npy;

use std::error::Error;
use std::fs::create_dir_all;
use std::fs::read_to_string;


use toml::from_str;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Config {
    input: Input,
    pub settings: Settings,
    output: Output,
}

#[derive(Deserialize)]
struct Input {
    path : String,
    filename: String,
}

#[derive(Deserialize)]
pub struct Settings {
    pub tov_iters : usize,
    pub kind: TovType,
    pub p_core: Option<f64>,
}

#[derive(Deserialize)]
pub enum TovType {
    MR,
    Tidal,
    Debug,
    }

#[derive(Deserialize)]
struct Output{
    path : String,
    filename: String,
}

pub fn load_config(filepath:&str) -> Config {
    let toml_str = read_to_string(filepath).expect("Failed to read config file.");
    let config: Config = from_str(&toml_str).expect("Failed to convert deserialise toml to rust data");

    return config
}

pub fn load_npyfile(config:&Config) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), Box<dyn Error>> {
    
    let filext: &str = ".npy";

    let input_filepath = format!("{}/{}{}", config.input.path, config.input.filename, filext);

    let data: ArrayD<f64> = read_npy(input_filepath)?;
    
    println!("Data loaded from {}.npy ", config.input.filename);


    let edens: Array2<f64> = data.slice(s![0, .., ..]).to_owned().into_dimensionality()?;
    let pressures: Array2<f64>= data.slice(s![1, .., ..]).to_owned().into_dimensionality()?; 
    let cs2s: Array2<f64> = data.slice(s![2, .., ..]).to_owned().into_dimensionality()?; 
    Ok((edens, pressures, cs2s))
}

pub fn write_npyfile(results: Array3<f64>, config: &Config) -> Result<(), Box<dyn Error>> {
    // Ensure the directory exists
    if let Some(parent) = std::path::Path::new(&config.output.path).parent() {
        create_dir_all(parent)?;
    }

    let filext: &str = ".npy";

    let full_path = match config.settings.kind {
        TovType::MR => &format!("{}/{}_mr{}", config.output.path, config.output.filename, filext),
        TovType::Tidal => &format!("{}/{}_tidal{}", config.output.path, config.output.filename, filext),
        TovType::Debug => &format!("{}/{}_debug{}", config.output.path, config.output.filename, filext)
    };

    write_npy(full_path, &results)?;

    match config.settings.kind {
    TovType::MR => println!("Data written to {}_mr.npy", config.output.filename),
    TovType::Tidal => println!("Data written to {}_tidal.npy", config.output.filename),
    TovType::Debug => println!("Data written to {}_debug.npy", config.output.filename)
    };

    Ok(())
}