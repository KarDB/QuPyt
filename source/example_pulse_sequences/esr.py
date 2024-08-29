"""
Generate the pulse sequence for Tektronix AWG using the
Mikrotron 1.1 CXP camera.
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
        params.get("mw_duration", 10),
        params.get("laserduration", 10),
        params.get("readout_time", 1),
        params.get("referenced_measurements", 100),
        params.get("max_framerate", 1000),
    )


def gen_esr(
    mw_duration: float,
    laserduration: float,
    readout_time: float,
    referenced_measurements: int,
    max_framerate: float = 10000,
) -> dict:
    """
    Implementation of the ESR pulsesequence.
    Will compute the optimal parameters for ESR
    pulse sequence.
    """
    buffer_time = 0.5  # mus space between pulses.
    laser_separation = 0.4

    # 1.3 + 0.5(buffer time)
    # + 0.2(half laser sep) = 2 == min exposure EoSense
    camera_min_exp_guarantee = 1.3
    time_half = buffer_time * 3 + mw_duration + laserduration + laser_separation
    time_half = max(time_half, 1 / max_framerate * 1e6)
    total_time = 2 * time_half
    esr = YamlSequence(duration=total_time)
    esr.add_pulse("MW_I", buffer_time, mw_duration)
    for i in range(2):
        esr.add_pulse(
            "LASER",
            i * time_half + 2 * buffer_time + mw_duration,
            readout_time,
            sequence_blocks=["wait_loop", "block_0"],
        )
        esr.add_pulse(
            "LASER",
            i * time_half
            + 2 * buffer_time
            + mw_duration
            + readout_time
            + laser_separation,
            laserduration - readout_time,
            sequence_blocks=["wait_loop", "block_0"],
        )
        exp_time = readout_time + 2
        esr.add_pulse(
            "READ",
            i * time_half + buffer_time + mw_duration - camera_min_exp_guarantee,
            2,
        )

    esr.sequencing_order = ["wait_loop", "block_0"]
    esr.sequencing_repeats = [1, int(referenced_measurements / 2) + 10]
    esr.write()
    return {"sensor": {"config": {"exposure_time": exp_time}}}
