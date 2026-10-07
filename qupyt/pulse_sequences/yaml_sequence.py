"""
Generation of yaml file based pulse sequences.
"""
import re
from typing import Dict, Any, Optional
import yaml
from qupyt import set_up
import numpy as np


class YamlSequence:
    def __init__(self, duration: float) -> None:
        self.pulse_sequence: Dict[str, Any] = {}
        self.counter: Dict[str, Any] = {}
        self.sequencing_order: list[str]
        self.sequencing_repeats: list[int]
        self.pulse_sequence["total_duration"] = duration

    def add_pulse(
        self,
        pulse_channel: str,
        start: float,
        duration: float,
        amplitude: float = 1.0,
        frequency: float = 0.0,
        phase: float = 0.0,
        sequence_blocks: list[str] = ["block_0"],
    ) -> None:
        for sequence_block in sequence_blocks:
            if sequence_block not in self.pulse_sequence:
                self.pulse_sequence[sequence_block] = {}
                self.counter[sequence_block] = {}
            if pulse_channel not in self.pulse_sequence[sequence_block]:
                self.pulse_sequence[sequence_block][pulse_channel] = {}
                self.counter[sequence_block][pulse_channel] = 1
            self.pulse_sequence[sequence_block][pulse_channel][
                "pulse{}".format(self.counter[sequence_block][pulse_channel])
            ] = {
                "start": start,
                "duration": duration,
                "amplitude": amplitude,
                "frequency": frequency,
                "phase": phase,
                # "sequence_block": sequence_block
            }
            self.counter[sequence_block][pulse_channel] += 1

    def write(self, sequence_number: int | None = None) -> None:
        sequence_number_string = "0" if sequence_number is None else str(sequence_number)
        file_path = set_up.get_seq_dir()
        file_name = f"sequence_{sequence_number_string}.yaml"
        self.pulse_sequence["sequencing_order"] = self.sequencing_order
        self.pulse_sequence["sequencing_repeats"] = self.sequencing_repeats
        with open(file_path / file_name, "w", encoding="utf-8") as file:
            yaml.dump(self.pulse_sequence, file)


class ComplexSequence:
    """Class that takes instance of
    YamlSequence and adds complex pulse sequences
    to requested channel in a macro like manner."""

    def __init__(
        self,
        # sequence_instance: yaml_squence_type,
        sequence_instance: YamlSequence,
        channel: str,
        tau: float,
        pi_half_pulse_dur: float,
        pi_pulse_dur: float,
        amplitude: float = 1,
        mixing_freq: float = 0,
        blocks: list[str] = ["block_0"],
        global_phase: float = 0,
        ts_start: float = 1,
        ts_end: float = 1,
    ) -> None:
        self.sequence = sequence_instance
        self.channel: str = channel
        self.tau: float = tau
        self.pi_half_pulse_dur: float = pi_half_pulse_dur
        self.pi_pulse_dur: float = pi_pulse_dur
        self.amplitude: float = amplitude
        self.mixing_freq: float = mixing_freq
        self.blocks: list[str] = blocks
        self.global_phase: float = global_phase
        self.tau_counter: int = 0
        self.phases: list[float] = []
        self.ts_start: float = ts_start
        self.ts_end: float = ts_end

        # Optional explicit pulse schedule for DROID60 and DIRAC2.
        # Start times are relative to the beginning of the full sequence.
        self.pulse_starts: Optional[list[float]] = None
        self.pulse_durations: list[float] = [] # Duration varies for DROID and DIRAC2

        # Total duration, including preparation and readout pulses.
        self.duration: float = 0.0 # Total sequence duration for scheduling the experiment

    def append_pulse(
        self,
        channel: str,
        start: float,
        duration: float,
        phase: float = 0,
        taushift: int = 2,
        hard_delay: float = 0,
    ) -> None:
        self.tau_counter += taushift
        self.sequence.add_pulse(
            channel,
            start + self.tau_counter * self.tau + hard_delay,
            duration,
            amplitude=self.amplitude,
            frequency=self.mixing_freq,
            phase=phase + self.global_phase,
            sequence_blocks=self.blocks,
        )

    def gen_phases(
        self,
        seq_type: str = "XY8",
        n: int = 8,
        readout_phase: float = np.pi / 2,
    ) -> None:
        """Generates a list of phases for a given sequence type.
        Args:
            seq_type (str, optional): Sequence type. Defaults to 'XY8'.
            n (int, optional): Number of repetitions of each single block of
            chosen sequence type. Defaults to 8.
            readout_phase (float, optional): Phase of readout pulse.
            Defaults to np.pi/2 (sine magnetometry).
            Set to 0 for cosine magnetometry.
        Returns:
            Nothing. "phases" attribute is set.
        Comments:
            Here N is the number of pi-pulses per block. It is also used to
            calculate the phases in the UD sequence.
        """

        # Clear any previously prepared sequence.
        self.pulse_starts = None
        self.pulse_durations = []
        self.phases = []
        self.duration = 0.0

        match = re.fullmatch(r"([A-Za-z]+)(\d+)", seq_type)
        if not match:
            raise ValueError(f"Invalid sequence type: {seq_type}")

        seq_name, N = match.groups()
        seq_name = seq_name.upper() # convert to uppercase letters
        N = int(N) # convert N to integer

        initial_phase = [0.0]
        final_phase = [readout_phase]

        if seq_name == "XY":

            # base: XY4 block
            phase_block = [0, np.pi / 2, 0, np.pi / 2]

            # build XY-N block: XY_{2N} = XY_N + reverse(XY_N)
            while len(phase_block) < N:
                phase_block = phase_block + phase_block[::-1]

            # XY only defined for N = 4 * 2^k
            if len(phase_block) != N:
                raise ValueError(f"Invalid XY lenght: {N}. N must be 4, 8, 16, 32, ...")


        elif seq_name == "UD": # From: Supplemental material to Arbitrarily accurate pulse sequences for robust dynamical decoupling

            phase_block = []

            # Determine phase increment
            if N % 4 == 0:
                phi_ud = np.pi / (N / 4)
            elif N % 4 == 2 and N != 2:
                phi_ud = (np.pi * (N - 2) / 2) / (((N - 2) / 2 ) + 1)
            else:
                raise ValueError(f"Invalid ud length. The value has to be even and four or higher.")

            # Generate quadratic phases
            if phi_ud is not None:
                for k in range(N):
                    phase_block.append(((k * (k + 1)) / 2 * phi_ud) % (2 * np.pi))

        elif seq_name in ("DROID", "DIRAC"):
            if (seq_name, N) not in (("DROID", 60), ("DIRAC", 2)): # N = sequence number
                raise ValueError(f"Unsupported sequence: {seq_type}")

            self._prepare_droid_dirac(
                seq_type=f"{seq_name}{N}",
                n=n,
                readout_phase=readout_phase,
            )
            return # Does not go into the other duration calcuation


        else:
            raise ValueError(f"Unsupported sequence family: {seq_type}")

        if phase_block is None:
            raise RuntimeError(f"phase_block was not initialized.")

        self.phases = initial_phase + list(phase_block) * n + final_phase
        number_inner_pulses = len(self.phases) - 2

        self.duration = ((self.ts_start + 2 * (number_inner_pulses - 1)+ self.ts_end) * self.tau + 2 * self.pi_half_pulse_dur)
        return None

    def _prepare_droid_dirac(
        self,
        seq_type: str,
        n: int,
        readout_phase: float,
    ) -> None:
        """Prepare an explicitly timed sequence, including outer pi/2 pulses."""
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer.")

        tau = self.tau
        pi = self.pi_pulse_dur
        pi_half = self.pi_half_pulse_dur

        if any(
            not np.isfinite(value) or value <= 0
            for value in (tau, pi, pi_half)
        ):
            raise ValueError("tau and pulse durations must be positive and finite.")

        if not all(
            np.isfinite(value)
            for value in (self.ts_start, self.ts_end, readout_phase)
        ):
            raise ValueError("Boundary shifts and readout phase must be finite.")


        # Reuse the existing definition without writing any pulses.
        definition = ArbitrarySequenceWriter(
            channel=self.channel,
            N=1,
            pi=pi,
            pi_half=pi_half,
            tau=tau,
            res_mix_freq=self.mixing_freq,
        )
        definition.prepare_sequence(seq_type)

        block_phases = definition.params["phases"]
        block_durations = definition.params["durations"]
        block_delays = definition.params["delays"]

        # Preserve the existing arbitrary writer's initial timing offset.
        initial_offset = pi

        if not (
            len(block_phases)
            == len(block_durations)
            == len(block_delays) - 1
        ):
            raise ValueError("Inconsistent pulse-block definition.")

        # Preparation pulse.
        starts = [0.0]
        durations = [pi_half]
        phases = [0.0]

        # Boundary convention:
        # ts_start=ts_end=1 preserves the default train timing.
        # Each extra unit adds tau at the corresponding boundary.
        cursor = initial_offset + (self.ts_start - 1) * tau

        for _ in range(n):
            cursor += block_delays[0]

            for k, phase in enumerate(block_phases):
                starts.append(cursor)
                durations.append(block_durations[k])
                phases.append(phase)

                # These are start-to-start increments, NOT empty gaps.
                cursor += block_delays[k + 1]

        # Readout placement: half-pulse duration after the train cursor,
        # plus the requested end-boundary shift.
        readout_start = (
            cursor + pi_half + (self.ts_end - 1) * tau
        )
        starts.append(readout_start)
        durations.append(pi_half)
        phases.append(readout_phase)

        # Validate before publishing the schedule to the writer.
        for k in range(1, len(starts)):
            previous_end = starts[k - 1] + durations[k - 1]
            if starts[k] < previous_end - 1e-12:
                raise ValueError(
                    "Pulses overlap. Check tau, pulse durations, "
                    "ts_start and ts_end."
                )

        self.pulse_starts = starts
        self.pulse_durations = durations
        self.phases = phases
        self.duration = starts[-1] + durations[-1]



    def write_sequence(self, start: float = 0) -> None:
        """Iterates over phases attibute and appends
        pulses to sequece instance.
        """
        if self.pulse_starts is not None: # pulse_starts contains a list: write each pulse with its specified start, duration and phase.
            if not (
                len(self.pulse_starts)
                == len(self.pulse_durations)
                == len(self.phases)
            ):
                raise ValueError(
                    "Pulse starts, durations and phases must have equal lengths."
                )

            for pulse_start, duration, phase in zip(
                self.pulse_starts,
                self.pulse_durations,
                self.phases,
            ):
                self.sequence.add_pulse(
                    self.channel,
                    start=start + pulse_start,
                    duration=duration,
                    amplitude=self.amplitude,
                    frequency=self.mixing_freq,
                    phase=phase + self.global_phase,
                    sequence_blocks=self.blocks,
                )

            return

        self.append_pulse(
            self.channel,
            start,
            self.pi_half_pulse_dur,
            self.phases[0],
            taushift=0,
        )
        for i, phase in enumerate(self.phases[1:-1]):
            if i == 0:
                self.append_pulse(
                    self.channel, start, self.pi_pulse_dur, phase, taushift=self.ts_start
                )
            else:
                self.append_pulse(self.channel, start, self.pi_pulse_dur, phase)
        self.append_pulse(
            self.channel, start, self.pi_half_pulse_dur, self.phases[-1], taushift=self.ts_end, hard_delay = self.pi_half_pulse_dur
        )


class ArbitrarySequenceWriter:
    def __init__(
        self,
        channel: str,
        N: int,
        pi: float,
        pi_half: float,
        tau: float,
        res_mix_freq: float,
        blocks: list[str] = ["block_0"],
        nLG4_per_tau: int = 0,
    ) -> None:
        self.channel = channel
        self.N = N
        self.pi = pi
        self.pi_half = pi_half
        self.tau = tau
        self.res_mix_freq = res_mix_freq
        self.blocks = blocks
        self.nLG4 = nLG4_per_tau

    def add_pulse(
        self,
        sequence_instance: YamlSequence,
        start: float,
        duration: float,
        amplitude: float,
        phase: float,
    ) -> None:
        sequence_instance.add_pulse(
            self.channel,
            start,
            duration,
            amplitude=amplitude,
            frequency=self.res_mix_freq,
            phase=phase,
            sequence_blocks=self.blocks,
        )

    def write_sequence(self, sequence_instance: YamlSequence, start: float) -> float:
        running_start: float = start
        num_pulses = len(self.params["phases"])

        assert (
            num_pulses == len(self.params["durations"])
            and num_pulses == len(self.params["delays"]) - 1
            and num_pulses == len(self.params["mixing_freqs"])
        ), "Uncompatible list lengths."

        if self.seq_type in ("DROID60", "DIRAC2", "LG4"):
            running_start += self.pi

        # Write sequence
        for _ in range(self.N):
            running_start += self.params["delays"][0]
            for k in range(num_pulses):
                sequence_instance.add_pulse(
                    self.channel,
                    running_start,
                    self.params["durations"][k],
                    amplitude=self.params["amplitudes"][k],
                    frequency=self.params["mixing_freqs"][k],
                    phase=self.params["phases"][k],
                    sequence_blocks=self.blocks,
                )

                running_start += self.params["delays"][k + 1]

        return running_start

    def prepare_sequence(
        self, seq_type: str, lock_scaling: float = 1.0
    ) -> Optional[float]:
        self.seq_type = seq_type
        supported_sequence_types = ["XY8", "DROID60", "DIRAC2", "LG4", "CPMG"]
        if seq_type not in supported_sequence_types:
            raise ValueError(
                "Sequence type {} not supported.\
                Currently only {} are supported".format(
                    seq_type, supported_sequence_types
                )
            )

        self.params = {}
        # delay1 pulse1 delay2 pulse2... pulseN delayN+1
        if seq_type == "XY8":
            self.params["delays"] = [self.tau] + [2 * self.tau] * 7 + [self.tau]
            self.params["durations"] = [self.pi] * 8
            self.params["amplitudes"] = [1] * 8
            self.params["mixing_freqs"] = [self.res_mix_freq] * 8
            self.params["phases"] = [
                0,
                np.pi / 2,
                0,
                np.pi / 2,
                np.pi / 2,
                0,
                np.pi / 2,
                0,
            ]

        elif seq_type == "CPMG":
            self.params["delays"] = [self.tau, self.tau]
            self.params["durations"] = [self.pi]
            self.params["amplitudes"] = [1]
            self.params["mixing_freqs"] = [self.res_mix_freq]
            self.params["phases"] = [0]

        elif seq_type == "DROID60":
            pi = self.pi
            pi2 = self.pi_half
            self.params["delays"] = (
                [2 * self.tau - pi, 2 * self.tau, pi2]
                + [2 * self.tau - pi2, 2 * self.tau, 2 * self.tau, 2 * self.tau, pi2]
                * 11
                + [2 * self.tau - pi2, 2 * self.tau, pi]
            )
            self.params["durations"] = (
                [pi, pi2, pi2] + [pi, pi, pi, pi2, pi2] * 11 + [pi, pi]
            )
            self.params["amplitudes"] = [1] * 60
            self.params["mixing_freqs"] = [self.res_mix_freq] * 60
            self.params["phases"] = [
                0,
                0,
                -np.pi / 2,
                np.pi,
                np.pi,
                0,
                0,
                -np.pi / 2,
                np.pi,
                np.pi,
                0,
                0,
                -np.pi / 2,
                np.pi,
                np.pi,
                -np.pi / 2,
                -np.pi / 2,
                0,
                np.pi / 2,
                np.pi / 2,
                -np.pi / 2,
                -np.pi / 2,
                0,
                np.pi / 2,
                np.pi / 2,
                -np.pi / 2,
                -np.pi / 2,
                0,
                np.pi / 2,
                np.pi / 2,
                -np.pi / 2,
                0,
                np.pi / 2,
                np.pi / 2,
                -np.pi / 2,
                -np.pi / 2,
                0,
                np.pi / 2,
                np.pi / 2,
                -np.pi / 2,
                -np.pi / 2,
                0,
                np.pi / 2,
                np.pi / 2,
                np.pi,
                np.pi,
                np.pi / 2,
                0,
                0,
                np.pi,
                np.pi,
                np.pi / 2,
                0,
                0,
                np.pi,
                np.pi,
                np.pi / 2,
                0,
                0,
                -np.pi / 2,
            ]
            return sum(self.params["delays"]) * self.N + self.pi + self.pi_half

        elif seq_type == "DIRAC2":
            pi = self.pi
            pi_half = self.pi_half
            tau = self.tau

            if not np.isclose(pi, 2 * pi_half, rtol=1e-9, atol=1e-12):
                raise ValueError(
                    "This DIRAC2 timing requires pi == 2 * pi_half."
                )

            # arXiv:2303.07374v1.
            # Each row: pi pulse, then two adjacent pi/2 pulses.
            # Phase units: pi/2; 0=+X, 1=+Y, 2=-X, -1=-Y.
            phase_triplets = [
                (-1, -1,  2),
                ( 1,  1,  2),
                ( 1,  1,  2),
                ( 1,  1,  0),
                (-1, -1,  2),
                (-1, -1,  2),
                (-1, -1,  0),
                (-1, -1,  2),
                ( 1,  1,  0),
                (-1, -1,  0),
                (-1, -1,  0),
                ( 1,  1, -1),
                (-1,  2,  1),
                ( 1,  2,  1),
                ( 1,  2, -1),
                (-1,  0,  1),
                ( 1,  2,  1),
                ( 1,  0,  1),
                ( 1,  0,  1),
                ( 1,  2, -1),
                (-1,  0, -1),
                (-1,  0, -1),
                (-1,  0,  1),
                ( 1,  0,  2),
            ]

            self.params["phases"] = [
                phase * np.pi / 2
                for triplet in phase_triplets
                for phase in triplet
            ]
            self.params["durations"] = [pi, pi_half, pi_half] * 24

            # Initial delay, followed by cursor increments after each pulse.
            self.params["delays"] = (
                [2 * tau - pi, 2 * tau, pi_half]
                + [2 * tau - pi_half, 2 * tau, pi_half] * 23
                + [pi_half]
            )

            self.params["amplitudes"] = [1.0] * 72
            self.params["mixing_freqs"] = [self.res_mix_freq] * 72

            # Same return-value convention as the existing DROID60 branch.
            return sum(self.params["delays"]) * self.N + pi + pi_half

        elif seq_type == "LG4":
            alpha = 55 * np.pi / 180
            t_cd = (self.tau - self.pi_half) / (4 * self.nLG4)
            rabi_cd = np.sqrt(2) / (np.sqrt(3) * t_cd)
            det_cd = round(rabi_cd / np.sqrt(2))
            amplitude_cd = float(2 * rabi_cd * self.pi) * lock_scaling
            assert amplitude_cd <= 1, "CD amplitude exceeding 1"

            LG4_freqs = [
                self.res_mix_freq - det_cd,
                self.res_mix_freq + det_cd,
                self.res_mix_freq + det_cd,
                self.res_mix_freq - det_cd,
            ]
            LG4_phases = [
                np.pi / 2 - alpha,
                np.pi + np.pi / 2 - alpha,
                np.pi + np.pi / 2 + alpha,
                np.pi / 2 + alpha,
            ]

            self.params["delays"] = (
                [0.0]
                + [t_cd] * 4 * self.nLG4
                + [self.pi]
                + [t_cd] * 8 * self.nLG4
                + [self.pi]
                + [t_cd] * 4 * self.nLG4
            )
            self.params["durations"] = (
                [t_cd] * 4 * self.nLG4
                + [self.pi]
                + [t_cd] * 8 * self.nLG4
                + [self.pi]
                + [t_cd] * 4 * self.nLG4
            )
            self.params["amplitudes"] = (
                [amplitude_cd] * 4 * self.nLG4
                + [1.0]
                + [amplitude_cd] * 8 * self.nLG4
                + [1.0]
                + [amplitude_cd] * 4 * self.nLG4
            )
            self.params["mixing_freqs"] = (
                LG4_freqs * self.nLG4
                + [self.res_mix_freq]
                + LG4_freqs * 2 * self.nLG4
                + [self.res_mix_freq]
                + LG4_freqs * self.nLG4
            )
            self.params["phases"] = (
                LG4_phases * self.nLG4
                + [0]
                + LG4_phases * 2 * self.nLG4
                + [0]
                + LG4_phases * self.nLG4
            )

            return sum(self.params["delays"]) * self.N + self.pi + self.pi_half

        return sum(self.params["delays"]) * self.N + self.pi_half