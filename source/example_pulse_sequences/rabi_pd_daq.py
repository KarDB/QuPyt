from __future__ import annotations

from typing import Any

import numpy as np

from qupyt.pulse_sequences.yaml_sequence import YamlSequence


def linspace_discrete_by_intervals(
    length_us: float, n_intervals: int, sampling_rate: float, tol_samples: float = 1e-9
) -> list[float]:
    """Return positive pulse lengths spaced on the sampling grid.

    Parameters
    ----------
    length_us : float
        Maximum pulse length in microseconds.
    n_intervals : int
        Number of equal intervals from zero to length_us.
    sampling_rate : float
        Sampling rate in hertz.
    tol_samples : float, optional
        Tolerance for sample-grid alignment, defaulting to 1e-9 samples.

    Returns
    -------
    list[float]
        Pulse lengths in microseconds, excluding zero and including length_us.

    Raises
    ------
    ValueError
        If the interval is shorter than one sample or does not align with the
        sampling grid within the specified tolerance.
    """
    Ts_us = (1.0 / sampling_rate) * 1e6  # sample period in microseconds
    n_points = n_intervals + 1  # Number of points implied by "number of spacings"
    dt_us = length_us / n_intervals  # Intended spacing between points
    step_exact = dt_us / Ts_us
    step_rounded = int(round(step_exact))  # Samples per spacing

    if step_rounded < 1:
        raise ValueError(
            f"Requested spacing {dt_us*1000:.3f} ns is < 1 sample ({Ts_us*1000:.3f} ns)."
        )

    # If not aligned, suggest nearest valid n_intervals
    if abs(step_exact - step_rounded) > tol_samples:
        nearest_n_intervals = round(length_us / (step_rounded * Ts_us))
        nearest_spacing_us = step_rounded * Ts_us
        max_interval = step_rounded * Ts_us * nearest_n_intervals
        raise ValueError(
            f"Spacing {dt_us:.9f} µs is not on the sampling grid "
            f"({step_exact:.9f} samples/step, nearest integer {step_rounded}).\n"
            f"Nearest valid n_intervals: {nearest_n_intervals} "
            f"(spacing {nearest_spacing_us:.9f} µs)."
            f"Maximum interval: {max_interval:.9f} µs."
        )

    idx = np.arange(n_points, dtype=np.int64) * step_rounded
    t_us = idx * Ts_us

    return [float(value) for value in t_us[1:]]


def generate_sequence(params: dict[str, Any]) -> dict[str, Any]:
    """Generate Rabi sequence files and return dependent measurement parameters."""
    return gen_rabi(
        float(params["max_mw_duration"]),
        int(params["number_measurements"]),
        float(params["laser_duration"]),
        float(params["AWG_frequency"]),
        float(params["sampling_rate"]),
        float(params.get("MW_amplitude", 1.0)),
        float(params.get("AOM_frequency", 250e6)),
        float(params.get("readout_offset", 1.0)),
        float(params.get("laser_mw_offset", 0.5)),
        float(params.get("mw_laser_offset", 0.5)),
    )


def gen_rabi(
    max_mw_duration: float,
    number_measurements: int,
    laser_duration: float,
    AWG_frequency: float,
    sampling_rate: float,
    MW_amplitude: float,
    AOM_frequency: float,
    readout_offset: float,
    laser_mw_offset: float,
    mw_laser_offset: float,
) -> dict[str, Any]:
    """Write the Rabi pulse sequence and compute its dependent parameters."""
    # Readout and extra timing
    buffer_time = 0.5  # between lasers + laser_mw_offset and mw_laser_offset
    read_trigger_duration = 1

    time_half = (
        buffer_time
        + max_mw_duration
        + 2 * laser_duration
        + laser_mw_offset
        + mw_laser_offset
    )
    total_time = 2 * time_half

    mw_durations = linspace_discrete_by_intervals(
        max_mw_duration, int(number_measurements / 2), sampling_rate
    )

    rabi = YamlSequence(duration=total_time)

    rabi.add_pulse(
        "START",
        0.1,  # Starting time of the pulse.
        read_trigger_duration,  # Pulse duration.
        # This pulse appears in two sequence blocks.
        sequence_blocks=["start_block"],
    )

    for i, pulse in enumerate(mw_durations):
        rabi.add_pulse(
            "MW",
            buffer_time + laser_duration + laser_mw_offset,
            pulse,
            amplitude=MW_amplitude,
            frequency=AWG_frequency,
            sequence_blocks=[f"block_{i}"],
        )
        for j in range(2):
            rabi.add_pulse(  # initialization laser
                "LASER",
                j * time_half + buffer_time,
                laser_duration,
                frequency=AOM_frequency,
                amplitude=0.2,
                sequence_blocks=["wait_loop", "start_block", f"block_{i}"],
            )
            rabi.add_pulse(
                "LASER",
                j * time_half
                + buffer_time
                + laser_duration
                + laser_mw_offset
                + pulse
                + mw_laser_offset,
                laser_duration,
                frequency=AOM_frequency,
                amplitude=0.2,
                sequence_blocks=["wait_loop", "start_block", f"block_{i}"],
            )
            rabi.add_pulse(
                "READ",
                j * time_half
                + buffer_time
                + laser_duration
                + laser_mw_offset
                + pulse
                + mw_laser_offset
                + readout_offset,
                read_trigger_duration,
                sequence_blocks=[f"block_{i}"],
            )

    rabi.sequencing_order = ["wait_loop", "start_block", "wait_loop"] + [
        f"block_{p}" for p in range(len(mw_durations))
    ]  # add the first block at the end to ensure it is executed first
    rabi.sequencing_repeats = [1, 1, 1] + (
        [1] * (len(mw_durations))
    )  # execute each block once
    rabi.write()
    return {}
