"""
Generate a sample ODMR pulse sequence file.
This pulse sequence is not suited to be used directly in a measurement.
Instead, the pulse sequence needs to be adjusted to the particularities
of the hardware used.
"""
# pylint: disable=logging-format-interpolation
# pylint: disable=logging-fstring-interpolation
# pylint: disable=logging-not-lazy
from qupyt.pulse_sequences.yaml_sequence import YamlSequence


def generate_sequence(params: dict):
    """
    Interace function to be called when this module
    is imported into the running python instance.
    """
    return gen_esr(
        params.get('mw_duration', 10),
        params.get('laserduration', 10),
        params.get('readout_time', 1),
        params.get('referenced_measurements', 100),
        params.get('max_framerate', 1000)
    )


def gen_esr(
        mw_duration: float,
        laserduration: float,
        readout_time: float,
        referenced_measurements: int,
        max_framerate: float = 10000
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
        mw_duration + laserduration + readout_and_repol_gap
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
            laserduration
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
    esr.write()
    # you can return a dict here that added / updates the configuration file.
    return {}
