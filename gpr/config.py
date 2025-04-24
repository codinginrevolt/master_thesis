from dataclasses import dataclass

@dataclass
class SamplingConfig:
    samples_n: int
    n_end: float
    n_points: int
    mu_start: float
    mu_end: float
    convert_eos: bool
    return_only_eos: bool
    return_connecting: bool
    return_normscale: bool
    save_dir: str
    filename: str

def parse_config(raw_config) -> SamplingConfig:
    s = raw_config['samples']
    o = raw_config['output']

    mu_start = s['mu_start'] if s['mu_start'] != 0.0 else 2.2 #GeV
    mu_end = s['mu_end'] if s['mu_end'] != 0.0 else 2.8

    convert_eos = s['convert_eos']
    return_only_eos = s['return_only_eos'] if convert_eos else False
    return_connecting = s['return_connecting'] if convert_eos else False
    return_normscale = s['return_normscale'] if convert_eos else False

    if return_connecting:
        return_normscale = False  # enforce mutual exclusion

    ## validations ##
    if s['samples_number'] <= 0:
        raise ValueError("samples_number must be > 0")

    if s['n_end'] <= 0:
        raise ValueError("n_end must be > 0")

    if s['n_points'] <= 0:
        raise ValueError("n_points must be > 0")

    if mu_start <= 0 or mu_end <= 0:
        raise ValueError("mu_start and mu_end must be > 0")

    if mu_end <= mu_start:
        raise ValueError("mu_end must be greater than mu_start")

    if not isinstance(s['convert_eos'], bool):
        raise TypeError("convert_eos must be a boolean")

    if not o['directory'] or not isinstance(o['directory'], str):
        raise ValueError("Output directory must be a non-empty string")

    if not o['filename'] or not isinstance(o['filename'], str):
        raise ValueError("Output filename must be a non-empty string")
    ####

    return SamplingConfig(
        samples_n=s['samples_number'],
        n_end=s['n_end'],
        n_points=s['n_points'],
        mu_start=mu_start,
        mu_end=mu_end,
        convert_eos=convert_eos,
        return_only_eos=return_only_eos,
        return_connecting=return_connecting,
        return_normscale=return_normscale,
        save_dir=o['directory'],
        filename=o['filename']
    )
