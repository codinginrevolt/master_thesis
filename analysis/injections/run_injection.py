import json
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

import bilby
from bilby.core.prior import Categorical

from nmma.gw.gw_likelihood import GravitationalWaveTransientLikelihood
from nmma.joint.conversion import MultimessengerConversion

##### CONFIG ########

minimum_frequency = 5.
reference_frequency = 5.
sampling_frequency = 4096

injection_parameters = {"mass_1": 1.41353439637635,
                        "mass_2": 1.3801062170723315,
                        "EOS": 32, #index
                        #"lambda_1": 491.622427756644,
                        #"lambda_2": 564.2120342541504,
                        "luminosity_distance": 677.4275000000001,
                        "ra": 2.869891377490007,
                        "dec": 0.07834986458854853,
                        "theta_jn": 2.740930552492068,
                        "psi": 0.33271964977617285,
                        "phase": 2.2784413443179354,
                        "geocent_time": 1696672400.0,
                        "a_1": 0.004324260113565681,
                        "a_2": 0.013737479750685747,
                        "tilt_1": 1.9881760730635827,
                        "tilt_2": 0.3920101596607665,
                        "phi_12": 5.056086360158561,
                        "phi_jl": 5.714484404266998
                        }


NEOS = 20_000
eos_path = "../eos/"
conversion = MultimessengerConversion(args=args,
                                        messengers=["gw"],
                                        ana_modifiers=["tabulated_eos"])
param_conv_func = conversion.convert_to_multimessenger_parameters
    
###################

def main():

    
    CHIEFF = (injection_parameters['a_1'] * injection_parameters['mass_1'] + injection_parameters['a_2'] * injection_parameters['mass_2']) / (injection_parameters['mass_1'] + injection_parameters['mass_2'])
    MCHIRP = bilby.gw.conversion.component_masses_to_chirp_mass(injection_parameters['mass_1'], injection_parameters["mass_2"])
    
    
    duration = bilby.gw.utils.calculate_time_to_merger(frequency = minimum_frequency, mass_1 = injection_parameters['mass_1'], mass_2 = injection_parameters['mass_2'], chi=CHIEFF, safety=1.1)
    duration = int(duration) + 1.
    
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
        waveform_arguments=dict(waveform_approximant="IMRPhenomXP_NRTidalv3", 
                                reference_frequency=reference_frequency,
                                minimum_frequency=minimum_frequency),
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters
    )
    
    
    ###################
    # INTERFEROMETERS #
    ###################
    
    ifos = bilby.gw.detector.InterferometerList(["ET"])
    for ifo in ifos:
        ifo.minimum_frequency = minimum_frequency
    
    ifos.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency,
                                                    duration=duration,
                                                    start_time=injection_parameters["geocent_time"] - duration + 2)
    ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)


    #########################
    # PRIORS AND LIKELIHOOD #
    #########################   

    priors = bilby.core.prior.PriorDict(filename='./bns.prior')
    priors['chirp_mass'] = bilby.core.prior.Uniform(name='chirp_mass', minimum=MCHIRP-0.01, maximum=MCHIRP+0.01)
    priors["geocent_time"] = bilby.core.prior.Uniform(name='geocent_time', minimum=injection_parameters["geocent_time"]-0.1, maximum=injection_parameters["geocent_time"]+0.1)
    priors["EOS"] = Categorical(NEOS, name="EOS")
    
    # make waveform generator for likelihood evaluations
    search_waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.binary_neutron_star_frequency_sequence,
        waveform_arguments=dict(waveform_approximant="IMRPhenomXP_NRTidalv3",
                                reference_frequency=reference_frequency))

    args = SimpleNamespace()
    args.eos_to_ram = False
    args.eos_data = eos_path

    # make multi-banded likelihood
    likelihood = GravitationalWaveTransientLikelihood(
        priors=priors,
        param_conv_func=param_conv_func,
        interferometers=ifos,
        waveform_generator=search_waveform_generator,
        gw_likelihood_type="MBGravitationalWaveTransient",
        reference_chirp_mass=priors["chirp_mass"].minimum,
        distance_marginalization=False,
        phase_marginalization=True,
        time_reference="geocent_time",
        reference_frame="sky"
    )


    ############
    # SAMPLING #
    ############
    
    
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=1024,
        naccept=60,
        npool=192,
        check_point_plot=True,
        check_point_delta_t=1800,
        print_method='interval-60',
        sample='acceptance-walk',
        conversion_function=param_conv_func,
        injection_parameters=injection_parameters,
        outdir="./outdir",
        label="ET_injection"
    )
    
    
    result.save_to_file()
    result.save_posterior_samples()


if __name__=="__main__":
    main()