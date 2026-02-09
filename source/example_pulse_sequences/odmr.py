"""
Generate a sample ODMR pulse sequence file.
This pulse sequence is not suited to be used directly in a measurement.
Instead, the pulse sequence needs to be adjusted to the particularities
of the hardware used.
"""
# pylint: disable=logging-format-interpolation
# pylint: disable=logging-fstring-interpolation
# pylint: disable=logging-not-lazy
import logging
from qupyt.pulse_sequences.yaml_sequence import YamlSequence


def generate_sequence(params: dict):
    """
    Interface function to be called when this module
    is imported into the running python instance.
    """
    pulse_sequence_steps = int(params.get("pulse_sequence_steps", 1))

    sweep_param_raw = params.get("sweep_param", None)
    sweep_param = None if sweep_param_raw in (None, "") else str(sweep_param_raw)
    sweep_values = params.get("sweep_values", None)

    print(sweep_param, pulse_sequence_steps, sweep_values)

    if sweep_param is not None and sweep_param not in params:
        logging.warning(
            "generate_sequence(): sweep_param=%r was provided, but no such key exists in params.",
            sweep_param,
        )

    def _value_for_step(step: int):
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
            "generate_sequence(): sweep_values must be either [start, stop] or have length == pulse_sequence_steps. "
            "Got length %d with pulse_sequence_steps=%d. No sweep applied.",
            len(values),
            pulse_sequence_steps,
        )
        return None

    for ps_step in range(pulse_sequence_steps):
        print('PS_step', ps_step, 'sequence_setps',pulse_sequence_steps)
        if sweep_param is None:
            ps_step = 0
        if sweep_param is not None and sweep_param in params:
            v = _value_for_step(ps_step)
            if v is not None:
                params[sweep_param] = v

        backup_params = gen_esr(
            params.get("mw_duration", 10),
            params.get("laser_duration", 10),
            params.get("readout_time", 1),
            params.get("referenced_measurements", 100),
            params.get("max_framerate", 1000),
            ps_step,
        )
    return backup_params



def gen_esr(
        mw_duration: float,
        laser_duration: float,
        readout_time: float,
        referenced_measurements: int,
        max_framerate: float = 10000,
        ps_step: int= 0
) -> dict:
    """
    Implementation of the ESR pulsesequence.
    Will compute the optimal parameters for ESR
    pulse sequence.
    """
    # All times in microseconds.
    readout_and_repol_gap = 2
    buffer_between_pulses = 1
    read_trigger_duration = 2

    # In this example the pulse sequence is designed to perform both
    # a readout (microwave on) and a referece measurement (microwave off)
    # for common mode noise rejection.
    # The readout and reference measurements get grouped together into one
    # logical block of the pulse sequence to avoid playback errors.
    # We compute the time it takes to perform the measurement step,
    # and double the time needed to take into account the reference.
    time_half = buffer_between_pulses * 3 + \
        mw_duration + laser_duration + readout_and_repol_gap
    time_half = max(time_half, 1/max_framerate * 1e6)
    total_time = 2 * time_half

    esr = YamlSequence(duration=total_time)

    # Add the microwave pulse at the beginning of the pulse sequece
    # starting after one buffer time. There is only one microwave pulse
    # per sequence block.
    esr.add_pulse(
        "MW",  # pulse channel, see YAML config file.
        buffer_between_pulses,  # Starting time of the pulse.
        mw_duration,  # Pulse duration.
        # This pulse appears in two sequence blocks.
        sequence_blocks=['wait_loop', 'block_0']
    )
    # Write the pulses that appear in the microwave and referece section.
    for i in range(2):
        esr.add_pulse(
            "LASER",
            i * time_half
            + 2 * buffer_between_pulses
            + mw_duration,
            readout_time,
            sequence_blocks=['wait_loop', 'block_0']
        )
        esr.add_pulse(
            "LASER",
            i * time_half
            + 2 * buffer_between_pulses
            + mw_duration
            + readout_time
            + readout_and_repol_gap,
            laser_duration
            - readout_time,
            sequence_blocks=['wait_loop', 'block_0']
        )
        esr.add_pulse(
            "READ",
            i * time_half
            + buffer_between_pulses
            + mw_duration,
            read_trigger_duration,
            sequence_blocks=['block_0']
        )
    # Here we sequence the defined sequence blocks in the order and number
    # of repetitions we want them to be played during the measurement.
    # This defines the order of the pulse blocks.
    esr.sequencing_order = ['wait_loop', 'block_0']
    # This defines how often each block in the sequence gets repeated.
    esr.sequencing_repeats = [1, int(referenced_measurements/2) + 10]
    esr.write(ps_step)
    # you can return a dict here that added / updates the configuration file.
    return {}
