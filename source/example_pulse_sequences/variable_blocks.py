"""Generate unequal AWG blocks with equivalent flat and repeated-child playback."""

from __future__ import annotations

import logging
import math
from typing import Any

from qupyt.pulse_sequences.yaml_sequence import YamlSequence


def generate_sequence(params: dict[str, Any]) -> dict[str, Any]:
    """Write example AWG sequences and return their acquisition parameters.

    Parameters
    ----------
    params : dict[str, Any]
        ``sampling_rate`` is in samples per second (default 5 GHz).
        ``block_durations`` optionally overrides the prep, drive, readout, and
        done durations in microseconds. ``awg_sequence_mode`` selects ``flat``
        (default) or ``repeated_child``. ``pulse_sequence_steps`` is the number
        of numbered YAML files to write (default one). ``second_source`` and
        ``include_flags`` are booleans, both defaulting to True.

    Returns
    -------
    dict[str, Any]
        Measurement updates including two readouts per acquisition, the number
        of pulse-sequence steps, and ``body_duration_us`` calculated from rounded
        sample counts. The latter excludes the AWG's trigger gate and wait time.

    Raises
    ------
    ValueError
        If the sampling rate, step count, mode, or duration names are invalid.

    Notes
    -----
    Logical playback is prep, drive twice, readout, drive twice, readout, done.
    The repeated child contains the drive and readout steps and repeats twice.
    Flags describe entire sequencer steps; their pulse start and duration do
    not create timed flag edges. The done block is entirely idle.
    """
    sampling_rate = float(params.get("sampling_rate", 5e9))
    if not math.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive and finite")
    steps = int(params.get("pulse_sequence_steps", 1))
    if steps < 1:
        raise ValueError("pulse_sequence_steps must be positive")
    mode = params.get("awg_sequence_mode", "flat")
    if mode not in ("flat", "repeated_child"):
        raise ValueError("awg_sequence_mode must be 'flat' or 'repeated_child'")

    durations = {"prep": 1.2, "drive": 2.4, "readout": 0.96, "done": 1.2}
    overrides = params.get("block_durations", {})
    if not isinstance(overrides, dict) or set(overrides) - durations.keys():
        raise ValueError("block_durations may override prep, drive, readout, and done")
    durations.update({name: float(value) for name, value in overrides.items()})
    second_source = bool(params.get("second_source", True))
    include_flags = bool(params.get("include_flags", True))

    for step in range(steps):
        # total_duration remains the fallback for blocks without an override.
        sequence = YamlSequence(duration=1.2, block_durations=durations)
        sequence.add_pulse(
            "MW", 0.2, 0.8, amplitude=0.2, frequency=50e6, sequence_blocks=["drive"]
        )
        sequence.add_pulse("READ", 0.1, 0.1, sequence_blocks=["readout"])
        if second_source:
            sequence.add_pulse(
                "LASER", 0.0, 0.5, amplitude=0.2, sequence_blocks=["prep", "readout"]
            )
            sequence.add_pulse("SYNC", 0.1, 0.1, sequence_blocks=["prep"])
        if include_flags:
            sequence.add_pulse("DRIVE_ACTIVE", 0.0, 0.1, sequence_blocks=["drive"])
            if second_source:
                sequence.add_pulse(
                    "READOUT_ACTIVE", 0.0, 0.1, sequence_blocks=["readout"]
                )
        # This also creates a block with no pulses, retaining its full idle time.
        sequence.set_block_duration("done", durations["done"])
        sequence.sequencing_order = [
            "prep",
            "drive",
            "readout",
            "drive",
            "readout",
            "done",
        ]
        sequence.sequencing_repeats = [1, 2, 1, 2, 1, 1]
        sequence.pulse_sequence["awg_sequence_mode"] = mode
        if mode == "repeated_child":
            sequence.pulse_sequence["awg_repeated_child"] = {
                "prefix_steps": 1,
                "child_steps": 2,
                "repetitions": 2,
            }
        sequence.write(step)

    sample_counts = {
        name: int(round(duration * sampling_rate * 1e-6))
        for name, duration in durations.items()
    }
    body_samples = (
        sample_counts["prep"]
        + 4 * sample_counts["drive"]
        + 2 * sample_counts["readout"]
        + sample_counts["done"]
    )
    body_duration_us = body_samples / sampling_rate * 1e6
    logging.info(
        "Generated variable-block example: mode=%s, files=%s, body_duration_us=%s",
        mode,
        steps,
        body_duration_us,
    )
    return {
        "pulse_sequence_steps": steps,
        "body_duration_us": body_duration_us,
        "sensor": {"config": {"number_measurements": 2}},
        "data": {"ps_steps": steps},
    }
