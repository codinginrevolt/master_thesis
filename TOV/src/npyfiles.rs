use ndarray::{s, Axis, Array2, Array3, ArrayD};
use ndarray_npy::read_npy;
use ndarray_npy::write_npy;

use std::error::Error;
use std::fs::{create_dir_all, read_to_string, File};
use std::io::Write;
use std::path::Path;

use serde::Deserialize;
use toml::from_str;

#[derive(Deserialize)]
pub struct Config {
    input: Input,
    pub settings: Settings,
    pub output: Output,
}

#[derive(Deserialize)]
struct Input {
    path: String,
    filename: String,
}

#[derive(Deserialize)]
pub struct Settings {
    pub tov_iters: usize,
    pub kind: TovType,
    pub return_unstable: bool,
    pub p_core: Option<f64>,
}

#[derive(Deserialize)]
pub enum TovType {
    MR,
    Tidal,
    Debug,
}

#[derive(Deserialize)]
pub struct Output {
    pub out_type: OutType,
    path: String,
    filename: String,
}

#[derive(Deserialize)]
#[allow(non_camel_case_types)]
pub enum OutType{
    npy,
    dat,
    both,
}

pub fn load_config(filepath: &str) -> Config {
    let toml_str = read_to_string(filepath).expect("Failed to read config file.");
    let config: Config =
        from_str(&toml_str).expect("Failed to convert deserialise toml to rust data");

    return config;
}

pub fn load_npyfile(
    config: &Config,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), Box<dyn Error>> {
    let filext: &str = ".npy";

    let input_filepath = format!("{}/{}{}", config.input.path, config.input.filename, filext);

    let data: ArrayD<f64> = match read_npy(input_filepath) {
        Ok(data) => {
            println!("Data loaded from {}.npy ", config.input.filename);
            data
        },
        Err(e) => panic!("Failed to read NPY file at '{}': {}", format!("{}/{}{}", config.input.path, config.input.filename, filext), e),
    };
    

    let edens: Array2<f64> = data.slice(s![0, .., ..]).to_owned().into_dimensionality()?;
    let pressures: Array2<f64> = data.slice(s![1, .., ..]).to_owned().into_dimensionality()?;
    let cs2s: Array2<f64> = data.slice(s![2, .., ..]).to_owned().into_dimensionality()?;
    Ok((edens, pressures, cs2s))
}

pub fn write_npyfile(results: Array3<f64>, config: &Config) -> Result<(), Box<dyn Error>> {
    // Ensure the directory exists
    if !Path::new(&config.output.path).is_dir() {
        create_dir_all(&config.output.path)?;
    }

    let filext: &str = ".npy";

    let full_path = match config.settings.kind {
        TovType::MR => &format!(
            "{}/{}_mr{}",
            config.output.path, config.output.filename, filext
        ),
        TovType::Tidal => &format!(
            "{}/{}_tidal{}",
            config.output.path, config.output.filename, filext
        ),
        TovType::Debug => &format!(
            "{}/{}_debug{}",
            config.output.path, config.output.filename, filext
        ),
    };

    write_npy(full_path, &results)?;

    match config.settings.kind {
        TovType::MR => println!("Data written to {}_mr.npy", config.output.filename),
        TovType::Tidal => println!("Data written to {}_tidal.npy", config.output.filename),
        TovType::Debug => println!("Data written to {}_debug.npy", config.output.filename),
    };

    Ok(())
}


pub fn write_datfile(results: Array3<f64>, config: &Config) -> Result<(), Box<dyn Error>> {

    let save_dir = format!("{}/{}", config.output.path, config.output.filename);
    if !Path::new(&save_dir).is_dir() {
        create_dir_all(&save_dir)?;
    }
    for (i, results_slice) in results.axis_iter(Axis(1)).enumerate() {
        let nan_mask = results_slice.column(1).mapv(|v| !v.is_nan());
        
        let valid_indices: Vec<_> = nan_mask
            .indexed_iter()
            .filter_map(|(idx, &is_valid)| if is_valid { Some(idx) } else { None })
            .collect();

        let no_nans_results_slice = results_slice.select(Axis(0), &valid_indices);

        let file_path = format!("{}/{}.dat", save_dir, i);
        let mut file = File::create(file_path)?;

        match config.settings.kind {
            TovType::MR => writeln!(file, "# r[km]    m[Msol]   core_pressure[MeV/fm³]")?,
            TovType::Tidal => writeln!(file, "# r[km]    m[Msol]    lambda    core_pressure[MeV/fm³]")?,
            TovType::Debug => writeln!(file, "# r[km]    m(r)[Msol]    y(r)    pressure(r)[MeV/fm³]")?,
        };
        for column in no_nans_results_slice.columns() {
            writeln!(file, "{}", column.iter().
                                    map(|v| v.to_string()).
                                    collect::<Vec<_>>().
                                    join(" "))?;        }
    }

    match config.settings.kind {
        TovType::MR => println!(".dat files created in {}/", save_dir),
        TovType::Tidal => println!(".dat files created in {}/", save_dir),
        TovType::Debug => println!(".dat files created in {}/", save_dir),
    };

    Ok(())
}
