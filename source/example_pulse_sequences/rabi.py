"""
Generate the pulse sequence for Tektronix AWG using the
Mikrotron 1.1 CXP camera.
"""

# pylint: disable=logging-format-interpolation
# pylint: disable=logging-fstring-interpolation
# pylint: disable=logging-not-lazy
import logging
import math
import numpy as np
from qupyt.pulse_sequences.yaml_sequence import YamlSequence


def generate_sequence(params: dict):
    """
    Interace function to be called when this module
    is imported into the running python instance.
    """
    return gen_rabi(
        float(params.get("min_pulse", 0.001)),
        float(params.get("max_pulse", 0.1)),
        int(params.get("steps", 100) / 2),
        float(params.get("laserduration", 30)),
        float(params.get("readout_time", 3)),
        float(params.get("mixing_freq", 0)),
        float(params.get("samprate", 5e9)),
        float(params.get("max_framerate", 1000)),
    )


def gen_rabi(
    min_pulse: float,
    max_pulse: float,
    steps: int,
    laserduration: float,
    readout_time: float,
    mixing_freq: float,
    samprate: float,
    max_framerate: float,
) -> dict:
    """
    Implementation of the Rabi pulsesequence.
    Will compute the optimal parameters for Rabi
    pulse sequence.
    """
    logging.info(f"Rabi mixing freq is {mixing_freq}")

    # MW pulse calculation
    t_step = (max_pulse - min_pulse) / (steps - 1)
    stepsize = 1 / samprate
    full_steps = round(t_step / stepsize) * stepsize
    max_pulse = min_pulse + full_steps * (steps - 1)
    max_pulse_buffer = math.ceil(max_pulse)

    # Readout and extra timing
    buffer_time = 0.5
    laser_separation = 1
    exp_time = math.ceil(
        max_pulse_buffer + 2 * buffer_time + readout_time + laser_separation / 2
    )

    time_half = exp_time + laser_separation / 2 + laserduration - readout_time
    time_half = max(time_half, 1 / max_framerate * 1e6)
    total_time = 2 * time_half

    pulses = np.linspace(min_pulse, max_pulse, steps)
    pulses = pulses.tolist()
    rabi = YamlSequence(duration=total_time)
    for i, pulse in enumerate(pulses):
        rabi.add_pulse(
            "MW_I",
            buffer_time + max_pulse_buffer - pulse,
            pulse,
            frequency=mixing_freq,
            phase=np.pi / 2,
            sequence_blocks=[f"block_{i}"],
        )
        rabi.add_pulse(
            "MW_Q",
            buffer_time + max_pulse_buffer - pulse,
            pulse,
            frequency=mixing_freq,
            sequence_blocks=[f"block_{i}"],
        )
        for j in range(2):
            rabi.add_pulse(
                "LASER",
                j * time_half + max_pulse_buffer + 2 * buffer_time,
                readout_time,
                sequence_blocks=[f"block_{i}"],
            )
            rabi.add_pulse(
                "LASER",
                j * time_half
                # + max_pulse_buffer
                # + 2 * buffer_time
                + exp_time + laser_separation / 2,
                laserduration - readout_time,
                sequence_blocks=[f"block_{i}"],
            )
            rabi.add_pulse("READ", j * time_half, 2, sequence_blocks=[f"block_{i}"])

    rabi.pulse_sequence["polarize"] = rabi.pulse_sequence[f"block_{i}"].copy()
    del rabi.pulse_sequence["polarize"]["READ"]

    rabi.sequencing_order = ["polarize", "block_0"]
    print(pulses)
    rabi.sequencing_order += [f"block_{i}" for i in range(len(pulses))]
    rabi.sequencing_repeats = [1, 10]
    rabi.sequencing_repeats += [1 for i in range(len(pulses))]
    rabi.write()
    return {
        "max_pulse_set": max_pulse,
        "sensor": {"config": {"exposure_time": exp_time}},
    }
