import numpy as np
import logging
from qupyt.pulse_sequences.yaml_sequence import YamlSequence, ComplexSequence

def check_valid_pulse_durations(pulse_durations, sampling_rate, tol=1e-12):

    dt = 1.0e6 / sampling_rate
    invalid = []

    for t in pulse_durations:
        # Check if t/dt is (nearly) integer
        n = t / dt
        if abs(n - round(n)) > tol:
            invalid.append(t)

    return len(invalid) == 0, invalid

def _value_for_step(step: int, *, sweep_values, pulse_sequence_steps: int, warn_prefix: str = "generate_sequence()"):
    if sweep_values is None:
        return None

    if isinstance(sweep_values, (list, tuple)):
        values = list(sweep_values)
    else:
        values = [sweep_values]

    if pulse_sequence_steps <= 1:
        return values[-1]

    # Endpoint form: [start, stop] -> interpolate across all steps
    if len(values) == 2:
        start, stop = float(values[0]), float(values[1])
        frac = step / (pulse_sequence_steps - 1)
        return start + (stop - start) * frac

    # Explicit list form: must match steps
    if len(values) == pulse_sequence_steps:
        return values[step]

    logging.warning(
        "%s: sweep_values must be either [start, stop] or have length == pulse_sequence_steps. "
        "Got length %d with pulse_sequence_steps=%d. No sweep applied.",
        warn_prefix,
        len(values),
        pulse_sequence_steps,
    )
    return None

def compute_calibration_signal(frames, dt, tau, f_target):
    '''
    this function computes the closest calibration frequency
    to f_target with the input parameters to ensure an integer number of cycle
    '''
    freqs = np.fft.rfftfreq(frames, dt * 1e-6)
    index = np.argmin(abs(freqs - f_target))
    f = abs(1/(4 * tau * 1e-6) - freqs[index])

    return float(f)

def generate_sequence(params: dict):
    """
    Interface function to be called when this module
    is imported into the running python instance.
    """
    pulse_sequence_steps = int(params.get("pulse_sequence_steps", 1))

    sweep_param_raw = params.get("sweep_param", None)
    sweep_param = None if sweep_param_raw in (None, "") else str(sweep_param_raw)
    sweep_values = params.get("sweep_values", None)

    if sweep_param is not None and sweep_param not in params:
        logging.warning(
            "generate_sequence(): sweep_param=%r was provided, but no such key exists in params.",
            sweep_param,
        )


    for ps_step in range(pulse_sequence_steps):
        # ... existing code ...
        if sweep_param is not None and sweep_param in params:
            v = _value_for_step(
                ps_step,
                sweep_values=sweep_values,
                pulse_sequence_steps=pulse_sequence_steps,
                warn_prefix="generate_sequence()",
            )
            if v is not None:
                params[sweep_param] = v

        backup_params = gen_sync(
            float(params.get('AWG_frequency')),
            str(params.get('seq_type')),
            int(params.get('N')),
            float(params.get('tau')),
            float(params.get('pi_half')),
            float(params.get('pi')),
            float(params.get('laser_duration')),
            int(params.get('number_measurements') / 2),
            float(params.get('sampling_rate')),
            float(params.get('calibration_target_frequency')),
            float(params.get('ts_start')),
            float(params.get('ts_end')),
            float(params.get('duration_offset')),
            float(params.get('readout_offset')),
            float(params.get('MW_amplitude')),
            float(params.get('AOM_frequency')),
            float(params.get('readout_phase')),
            float(params.get('global_phase')),
            ps_step,
    )
    return backup_params

def gen_sync(AWG_frequency: float,
              seq_type: str,
              N: int,
              tau: float,
              pi_half: float,
              pi: float,
              laser_duration: float,
              number_measurements: int,
              sampling_rate: float,
              calibration_target_frequency: float = 1000,
              ts_start: float = 1,
              ts_end: float = 1,
              duration_offset: float = 0,
              readout_offset: float = 1.0,
              MW_amplitude: float = 1.0,
              AOM_frequency: float = 250e6,
              readout_phase: float = 0,
              global_phase: float = 0,
              ps_step: int = 0,
              ) -> dict:

    buffer_between_pulses = 0.5
    read_trigger_duration = 1

    pulse_dict = {
        "tau": tau,
        "pi_half": pi_half,
        "pi": pi,
        "laser_duration": laser_duration,
        "ts_start": ts_start * tau,
        "ts_end": ts_end * tau,
        "duration_offset": duration_offset,
        "readout_offset": readout_offset,
        "buffer_between_pulses": buffer_between_pulses,
        "read_trigger_duration": read_trigger_duration

    }

    valid, invalid_values = check_valid_pulse_durations(pulse_dict.values(),sampling_rate)
    invalid_names = [name for name, value in pulse_dict.items() if value in invalid_values]

    if invalid_names:
        invalid_us = [value * 1e6 for value in invalid_values]  # convert to ns
        raise ValueError(
            f"Requested spacing {1e9 / sampling_rate:.3f} ns is not reached for: "
            f"{', '.join(f'{n} = {v:.3f} µs' for n, v in zip(invalid_names, invalid_us))}"
        )

    pulses_duration = 8 * N * 2 * tau + ts_start * tau + ts_end * tau + 1 * pi_half + duration_offset
    time_half = 2 * buffer_between_pulses + pulses_duration + laser_duration
    
    # makes sure the seq is an interger*4 multiple of tau.
    time_half = tau*(int(time_half/tau)
                     + 4 - int(time_half/tau) % 4)
    total_time = 2 * time_half

    sync = YamlSequence(duration=total_time)

    sync_mw = ComplexSequence(
        sync,
        channel='MW',
        tau=tau,
        pi_half_pulse_dur=pi_half,
        pi_pulse_dur=pi,
        amplitude=MW_amplitude,
        mixing_freq=AWG_frequency,
        blocks=['wait_loop', 'start_block', 'sensing_block'],
        global_phase=global_phase,
        ts_start=ts_start,
        ts_end=ts_end
    )
    sync_mw.gen_phases(seq_type=seq_type, n=N,
                         readout_phase=np.pi/2 + readout_phase)
    sync_mw.write_sequence(buffer_between_pulses)

    sync_mw.tau_counter = 0
    sync_mw.gen_phases(seq_type=seq_type, n=N,
                         readout_phase=-np.pi/2 + readout_phase)
    sync_mw.write_sequence(time_half + buffer_between_pulses)


    sync.add_pulse(
        'START',
        0.1,
        read_trigger_duration,
        sequence_blocks=['start_block']
        )

    for i in range(2):
        sync.add_pulse('LASER',
                   i * time_half + 2 * buffer_between_pulses + pulses_duration,
                   laser_duration,
                   frequency=AOM_frequency,
                   amplitude=0.2,
                   sequence_blocks=['wait_loop', 'start_block','sensing_block'])

        sync.add_pulse("READ",
                   i * time_half + 2 * buffer_between_pulses + pulses_duration + readout_offset,
                   read_trigger_duration,
                   sequence_blocks=['sensing_block']
                   )

    sync.add_pulse('CALIB', 0, total_time,
                   sequence_blocks=['sensing_block'])

    sync.sequencing_order = ['wait_loop','start_block','wait_loop','sensing_block']
    sync.sequencing_repeats = [1,2,1, number_measurements]
    sync.write(ps_step)

    calibration_frequency = compute_calibration_signal(number_measurements, total_time, tau, calibration_target_frequency)

    return {
        'dynamic_devices': {
            'rf_calibration_source': {
                'config': {'frequency': ['channel_2',[ calibration_frequency,  calibration_frequency]]}
            }
        },
        'duration_pulseseq_cycle': total_time}
