"""
Generation of yaml file based pulse sequences.
"""

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

    def write(self) -> None:
        file_path = set_up.get_seq_dir()
        self.pulse_sequence["sequencing_order"] = self.sequencing_order
        self.pulse_sequence["sequencing_repeats"] = self.sequencing_repeats
        with open(file_path / "sequence.yaml", "w", encoding="utf-8") as file:
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

    def append_pulse(
        self,
        channel: str,
        start: float,
        duration: float,
        phase: float = 0,
        taushift: int = 2,
        hard_delay: float = 0,
    ) -> None:
        self.sequence.add_pulse(
            channel,
            start + (self.tau_counter + taushift) * self.tau + hard_delay,
            duration,
            amplitude=self.amplitude,
            frequency=self.mixing_freq,
            phase=phase + self.global_phase,
            sequence_blocks=self.blocks,
        )
        self.tau_counter += taushift

    def gen_phases(
        self,
        seq_type: str = "XY8",
        n: int = 8,
        readout_phase: float = np.pi / 2,
    ) -> None:
        """Generates a list of phases for a given sequence type.
        Args:
            seq_type (str, optional): Sequence type. Defaults to 'XY8'.
            n (int, optional): Number of repetitions of single block of
            chosen sequence type. Defaults to 8.
            readout_phase (float, optional): Phase of readout pulse.
            Defaults to np.pi/2 (sine magnetometry).
            Set to 0 for cosine magnetometry.
        Returns:
            Nothing. "phases" attribute is set.
        """
        supported_seq_types = ["XY8"]
        if seq_type not in supported_seq_types:
            raise ValueError(
                "Sequence type {} not supported.\
                Currently only {} are supported".format(
                    seq_type, supported_seq_types
                )
            )
        if seq_type == "XY8":
            initial_phase = [0.0]
            final_phase = [readout_phase]
            xy8_phases = [0, np.pi / 2, 0, np.pi / 2, np.pi / 2, 0, np.pi / 2, 0]
            self.phases = initial_phase + xy8_phases * n + final_phase
            return None

    def write_sequence(self, start: float = 0) -> None:
        """Iterates over phases attibute and appends
        pulses to sequece instance.
        """
        self.append_pulse(
            self.channel,
            start,
            self.pi_half_pulse_dur,
            self.phases[0],
            hard_delay=self.pi_half_pulse_dur,
            taushift=0,
        )
        for i, phase in enumerate(self.phases[1:-1]):
            if i == 0:
                self.append_pulse(
                    self.channel, start, self.pi_pulse_dur, phase, taushift=1
                )
            else:
                self.append_pulse(self.channel, start, self.pi_pulse_dur, phase)
        self.append_pulse(
            self.channel, start, self.pi_half_pulse_dur, self.phases[-1], taushift=1
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

        if self.seq_type in ("DROID60", "LG4"):
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
        supported_sequence_types = ["XY8", "DROID60", "LG4", "CPMG"]
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
