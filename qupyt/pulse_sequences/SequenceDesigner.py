from __future__ import annotations

import hashlib
import logging
import pickle
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray
from termcolor import colored

from qupyt.set_up import get_seq_dir

# Bump when numeric compilation changes so older artifacts are regenerated.
NUMERIC_SEQUENCE_CACHE_VERSION = 1


def validate_sequence_instructions(instructions: object) -> None:
    """Reject incomplete metadata before a zip silently drops sequencing steps."""
    if not isinstance(instructions, dict):
        raise ValueError("Pulse-sequence instructions must contain a YAML mapping")
    order = instructions.get("sequencing_order")
    repeats = instructions.get("sequencing_repeats")
    if (
        not isinstance(order, list)
        or not isinstance(repeats, list)
        or not order
        or len(order) != len(repeats)
    ):
        raise ValueError(
            "sequencing_order and sequencing_repeats must be nonempty lists of equal length"
        )
    for block, repeat in zip(order, repeats):
        if not isinstance(block, str) or not isinstance(instructions.get(block), dict):
            raise ValueError(f"Missing or invalid pulse-sequence block {block!r}")
        if not isinstance(repeat, int) or repeat < 1:
            raise ValueError(
                f"Repeat count for block {block!r} must be a positive integer; got {repeat!r}"
            )


def parse_awg_flag(selector: str, channels: list[int]) -> tuple[str, int | None]:
    """Parse a flag selector; its suffix is a physical AWG source channel."""
    if (
        not isinstance(selector, str)
        or re.fullmatch(r"[ABCD](?:[12])?", selector) is None
    ):
        raise ValueError(
            f"Invalid AWG flag selector {selector!r}; expected A, B, C, D "
            "or a flag followed by physical AWG channel 1 or 2 (for example A2)."
        )
    channel = int(selector[1]) if len(selector) == 2 else None
    if channel is not None and channel not in channels:
        raise ValueError(
            f"AWG flag selector {selector!r} requests inactive AWG channel "
            f"{channel}; active channels are {channels}."
        )
    return selector[0], channel


class PulseSequenceYaml:
    def __init__(
        self,
        #  Channel list as you think about them
        #  marker 1, marker 2 and so on
        channel_mapping: dict[str, Any],
        #  Source as the AWG thinks about them
        #  e.g. source1 might include analog 1, marker 1, ...
        #  There is one analog channel per source but mulitple makers etc.
        awg_sources: list[int],
        samprate: float,  # = 5e9,
        yaml_file: str | Path | None = None,
    ) -> None:
        self.yaml_file = Path("sequence_0.yaml" if yaml_file is None else yaml_file)
        if not self.yaml_file.is_absolute():
            self.yaml_file = get_seq_dir() / self.yaml_file
        self.awg_sources = awg_sources
        self.channel_mapping = channel_mapping
        self.samp_rate = float(samprate)  # samples per second

    def _sequence_didnt_change(self) -> bool:
        with open(self.yaml_file, "r", encoding="utf-8") as file:
            sequence_instructions = yaml.safe_load(file)
        validate_sequence_instructions(sequence_instructions)
        # Configuration changes (including A -> A2) must regenerate flag metadata
        # even when the pulse YAML is unchanged.
        sequence_instructions = {
            "compiler_version": NUMERIC_SEQUENCE_CACHE_VERSION,
            "sequence": sequence_instructions,
            "channel_mapping": self.channel_mapping,
            "awg_sources": self.awg_sources,
            "sampling_rate": self.samp_rate,
        }
        self._cache_instructions = sequence_instructions
        # All YAML inputs compile into the same artifact. Keep its cache beside
        # that artifact so alternating YAML files cannot reuse another sequence.
        self._cache_path = get_seq_dir() / "sequence.npz.aux"
        try:
            artifact = (get_seq_dir() / "sequence.npz").stat()
            with self._cache_path.open("r", encoding="utf-8") as file:
                previous = yaml.safe_load(file)
        except FileNotFoundError:
            return False
        except yaml.YAMLError:
            logging.warning(
                "Invalid pulse-sequence cache %s; recompiling",
                self._cache_path,
                exc_info=True,
            )
            return False
        return isinstance(previous, dict) and previous == {
            "instructions": sequence_instructions,
            "artifact": [artifact.st_mtime_ns, artifact.st_size],
        }

    def _record_compiled_sequence(self) -> None:
        """Record a cache hit only after the numeric artifact was saved successfully."""
        artifact = (get_seq_dir() / "sequence.npz").stat()
        with self._cache_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                {
                    "instructions": self._cache_instructions,
                    "artifact": [artifact.st_mtime_ns, artifact.st_size],
                },
                file,
            )

    def translate_yaml_to_numeric_instructions(self) -> None:
        logging.info(
            "Translating pulse sequence from %s at sampling_rate=%s",
            self.yaml_file,
            self.samp_rate,
        )
        for selector in self.channel_mapping.values():
            if isinstance(selector, str):
                parse_awg_flag(selector, self.awg_sources)
        if self._sequence_didnt_change():
            logging.info(
                "Reusing cached numeric pulse sequence for unchanged configuration %s",
                self.yaml_file,
            )
            return
        with open(self.yaml_file, "r", encoding="utf-8") as file:
            sequence_instructions = yaml.safe_load(file)
        sequence_order = sequence_instructions["sequencing_order"]
        sequencing_repeats = sequence_instructions["sequencing_repeats"]
        duration = float(sequence_instructions["total_duration"])
        sorted_pulse_blocks = sorted(set(sequence_order))
        seq = PulseSequence(
            len(sorted_pulse_blocks),
            duration,
            self.awg_sources,
            samprate=self.samp_rate,
        )
        flag_channels: dict[str, list[str]] = {}
        for i, block in enumerate(sorted_pulse_blocks):
            flag_channels[block] = []
            for channel, pulses in sequence_instructions[block].items():
                mapped_channel = self.channel_mapping[channel]
                if isinstance(mapped_channel, str):
                    flag_channels[block].append(mapped_channel)
                    continue
                for pulse in pulses.values():
                    seq.add_pulse(
                        i,
                        float(pulse["start"]),
                        float(pulse["duration"]),
                        channel=mapped_channel,
                        inputtype="time",
                        amplitude=float(pulse["amplitude"]),
                        freq=(float(pulse["frequency"]), float(pulse["phase"])),
                    )
        seq.flag_channels = pickle.dumps(flag_channels)
        seq.sequencer = sequencing_repeats
        seq.sequencernames = sequence_order
        seq.make("sequence.npz")
        self._record_compiled_sequence()
        logging.info(
            "Translated pulse sequence from %s: blocks=%s, channels=%s",
            self.yaml_file,
            len(sorted_pulse_blocks),
            self.awg_sources,
        )


class PulseSequence:
    def __init__(
        self,
        numseqs: int,
        duration: float,
        awg_sources: list[int],
        samprate: float = 2.5e9,
    ) -> None:
        if (
            not np.isfinite(samprate)
            or samprate <= 0
            or not np.isfinite(duration)
            or duration <= 0
            or numseqs < 1
        ):
            raise ValueError(
                "Pulse sequence requires positive sampling rate, duration, and block count"
            )
        self.samp_rate = samprate  # samples per second
        self.min_time = 1 / samprate
        points = samprate * duration * 1e-6
        self.num_points = int(round(points))
        if self.num_points < 1:
            raise ValueError("Pulse sequence must contain at least one sampling point")
        self.time = np.arange(self.num_points) * self.min_time * 1e6  # microseconds
        self.numseqs = numseqs
        self.awg_sources = awg_sources

        if not np.isclose(points, self.num_points, rtol=0.0, atol=1e-9):
            print(
                colored(
                    "WARNING! the sequence duration is not an integer multiple of samples".ljust(
                        65, "."
                    )
                    + "! [WARNING]",
                    "red",
                )
            )
            logging.warning(
                "The sequence duration is not an integer multiple of samples".ljust(
                    65, "."
                )
                + "[WARNING]"
            )

        #  5 => 1 for analog, 4 for 8 bit marker.
        self.pulses = np.zeros((numseqs, 5 * len(self.awg_sources), self.num_points))
        self.sequencer: list[int] | None = None
        self.sequencernames: list[str] | None = None
        self.flag_channels: Any
        self.warning_counter = 0
        self.properties = {"Values": "None"}

    def time_to_index(self, time: float) -> int:
        points = self.samp_rate * time * 1e-6
        if not np.isclose(points, int(round(points))):
            print(
                colored(
                    "WARNING! the start or duration time is not an integer multiple of samples".ljust(
                        65, "."
                    )
                    + "! "
                    "[WARNING]",
                    "red",
                )
            )
            print(
                colored(
                    "WARNING! {} comp. to {}".format(points, int(round(points))).ljust(
                        65, "."
                    )
                    + "! [WARNING]",
                    "red",
                )
            )
            logging.warning(
                "Pulse start or duration %s us corresponds to %s samples; rounding to %s",
                time,
                points,
                int(round(points)),
            )
            self.warning_counter += 1
        return int(round(points))

    def add_pulse(
        self,
        numseq: int | tuple[int, int] | list[int],
        start: float,
        duration: float,
        channel: int = 0,
        inputtype: str = "time",
        amplitude: float = 1.0,
        freq: tuple[float, float] | None = None,
    ) -> None:
        """Add an analog pulse to one numeric sequence block or a range of blocks.

        Parameters
        ----------
        numseq : int or tuple[int, int] or list[int]
            Block index, or a two-element start/stop range with an exclusive stop.
        start : float
            Pulse start in microseconds when inputtype is time, otherwise in samples.
        duration : float
            Pulse duration in the same units as start.
        channel : int, optional
            Internal waveform-channel index. Defaults to 0.
        inputtype : str, optional
            time interprets start and duration in microseconds; other values use samples.
        amplitude : float, optional
            Pulse amplitude multiplier, defaulting to 1.0.
        freq : tuple[float, float] or None, optional
            Frequency in hertz and phase in radians for cosine modulation. None adds
            an unmodulated pulse.

        Raises
        ------
        ValueError
            If the pulse is outside the sequence bounds or a block range has other
            than two elements.
        """
        if inputtype == "time":
            start = self.time_to_index(start)
            duration = self.time_to_index(duration)
        else:
            logging.debug("Interpreting pulse input as a number of sampling points")

        if start < 0 or duration < 0 or start + duration > self.num_points:
            raise ValueError(
                f"Pulse interval [{start}, {start + duration}) exceeds sequence bounds [0, {self.num_points})"
            )
        fr: NDArray[np.float64] | int
        if freq is not None:
            frequency, phase = freq
            frequency *= 10**-6
            fr = np.cos(
                2 * np.pi * frequency * self.time[int(start) : int(start + duration)]
                + phase
            )
        else:
            fr = 1
        if isinstance(numseq, (tuple, list)):
            if len(numseq) != 2:
                raise ValueError(
                    "Pulse sequence range must contain a start and stop index"
                )
            self.pulses[
                numseq[0] : numseq[1], channel, int(start) : int(start + duration)
            ] = (amplitude * fr)
        else:
            self.pulses[numseq, channel, int(start) : int(start + duration)] = (
                amplitude * fr
            )

    def make(self, name: str) -> None:
        logging.info(
            "Pulse sequence generation produced %s timing warnings",
            self.warning_counter,
        )
        final = np.zeros((self.numseqs, 2 * len(self.awg_sources), self.num_points))
        for source_index, _source in enumerate(self.awg_sources):
            final[:, 2 * source_index, :] = self.pulses[:, 5 * source_index, :]
            final[:, 2 * source_index + 1, :] = (
                self.pulses[:, 5 * source_index + 1, :] * 2**7
                + self.pulses[:, 5 * source_index + 2, :] * 2**6
                + self.pulses[:, 5 * source_index + 3, :] * 2**5
                + self.pulses[:, 5 * source_index + 4, :] * 2**4
            )

        # These digests identify compiled waveforms for cache reuse.
        hash1 = hashlib.sha1(final.tobytes(), usedforsecurity=False).hexdigest()
        hash2 = hashlib.sha1(
            str(self.sequencer).encode("utf-8"), usedforsecurity=False
        ).hexdigest()
        hash3 = hashlib.sha1(
            str(self.sequencernames).encode("utf-8"), usedforsecurity=False
        ).hexdigest()
        hash4 = hash1 + hash2 + hash3
        finalhash = hashlib.sha1(
            hash4.encode("utf-8"), usedforsecurity=False
        ).hexdigest()

        file_dir = get_seq_dir()
        try:
            np.savez(
                file_dir / name,
                final,
                np.asarray(self.sequencer),
                np.asarray(self.sequencernames),
                finalhash,
                np.asarray(self.properties),
                self.flag_channels,
            )
        except Exception:
            logging.exception(
                "Failed to save numeric pulse sequence to %s", file_dir / name
            )
            raise
        logging.info(
            "Saved numeric pulse sequence to %s: shape=%s", file_dir / name, final.shape
        )


class PulseBlasterSequence:
    def __init__(
        self, channel_mapping: dict[str, Any], yaml_file: str | Path | None = None
    ) -> None:
        self.event_times: list[float] = []
        self.event_durations: list[float] = []
        self.event_channel: list[str] = []
        self.events: list[str] = []
        self.channel_bits: list[int] = []
        self.segment_durations: list[float] = []
        self.channel_mapping = channel_mapping
        self.ps: dict[str, Any] = {}
        sequence_path = (
            get_seq_dir() / "sequence_0.yaml" if yaml_file is None else yaml_file
        )
        self.yaml_sequence: dict[str, Any] = self._load_yaml_sequence(sequence_path)
        self.total_duration = self.yaml_sequence["total_duration"]

    def _load_yaml_sequence(self, path: str | Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as file:
            yaml_sequence = yaml.safe_load(file)
        validate_sequence_instructions(yaml_sequence)
        logging.debug("Loaded PulseBlaster pulse sequence from %s", path)
        return cast(dict[str, Any], yaml_sequence)

    def parse_pulse_sequence_file(self) -> None:
        for block in self.yaml_sequence["sequencing_order"]:
            self._parse_block(self.yaml_sequence[block])
            self._sort_pulses()
            self._get_event_durations()
            self._compute_channel_bits()
            self.ps[block] = {
                "channel_bits": self.channel_bits,
                "durations": self.event_durations,
            }
            self._reset_attributes()

    def compile(self) -> tuple[list[int], list[float]]:
        """Expand the logical blocks into one PulseBlaster instruction sequence.

        Returns
        -------
        list[int]
            Channel bit masks in playback order.
        list[float]
            Corresponding instruction durations in microseconds.
        """
        channel_bits = []
        bits_duration = []
        sequencing_info = zip(
            self.yaml_sequence["sequencing_order"],
            self.yaml_sequence["sequencing_repeats"],
        )
        for sequence_block, block_repeats in sequencing_info:
            channel_bits += self.ps[sequence_block]["channel_bits"] * block_repeats
            bits_duration += self.ps[sequence_block]["durations"] * block_repeats

        logging.info(
            "Compiled PulseBlaster pulse sequence: instructions=%s, duration=%s us",
            len(channel_bits),
            sum(bits_duration),
        )
        return channel_bits, bits_duration

    def _reset_attributes(self) -> None:
        self.event_times = []
        self.event_durations = []
        self.event_channel = []
        self.events = []
        self.channel_bits = []
        self.segment_durations = []

    def _append_event(self, channel: str, pulse: dict[str, float]) -> None:
        if (
            not np.isfinite(pulse["start"])
            or not np.isfinite(pulse["duration"])
            or pulse["start"] < 0
            or pulse["duration"] < 0
        ):
            raise ValueError(
                f"Pulse on {channel!r} requires finite, nonnegative start and duration"
            )
        self.event_times.append(pulse["start"])
        self.event_times.append(pulse["start"] + pulse["duration"])
        self.event_channel.append(channel)
        self.event_channel.append(channel)
        self.events.append("up")
        self.events.append("down")

    def _parse_channel(self, channel: str, channel_pulses: dict[str, Any]) -> None:
        for pulse in channel_pulses.values():
            self._append_event(channel, pulse)

    def _parse_block(self, ps_block: dict[str, Any]) -> None:
        for channel, channel_pulses in ps_block.items():
            self._parse_channel(channel, channel_pulses)

    def _sort_pulses(self) -> None:
        if not self.event_times:
            raise ValueError(
                "PulseBlaster sequence block must contain at least one pulse"
            )
        event_times, event_channels, event_states = zip(
            *sorted(zip(self.event_times, self.event_channel, self.events))
        )
        self.event_times = list(event_times)
        self.event_channel = list(event_channels)
        self.events = list(event_states)

    def _get_event_durations(self) -> None:
        self.event_durations = [
            i - j for i, j in zip(self.event_times[1:], self.event_times[:-1])
        ]
        if self.total_duration == "ignore":
            pass
        elif self.event_times[-1] > self.total_duration:
            raise ValueError(
                f"Pulse ends at {self.event_times[-1]} us, beyond total_duration={self.total_duration} us"
            )
        elif self.event_times[-1] != self.total_duration:
            self.event_durations.append(self.total_duration - self.event_times[-1])

    def _event_to_sign(self, event: str) -> int:
        if event == "up":
            return 1
        if event == "down":
            return -1
        raise ValueError(f'No event {event}, options are "up" or "down"')

    def _compute_channel_bits(self) -> None:
        for i, _ in enumerate(self.event_durations):
            prev_val = self.channel_bits[-1] if self.channel_bits else 0
            self.channel_bits.append(
                prev_val
                + 2 ** (self.channel_mapping[self.event_channel[i]])
                * self._event_to_sign(self.events[i])
            )
        if self.event_times[0] != 0:
            self.channel_bits.insert(0, 0)
            self.event_durations.insert(0, self.event_times[0])
        self._pop_uncessary_entries()

    def _pop_uncessary_entries(self) -> None:
        indices_to_pop = [
            index
            for index, duration in enumerate(self.event_durations)
            if duration == 0
        ]
        for pop_index in indices_to_pop:
            self.event_durations.pop(pop_index)
            self.channel_bits.pop(pop_index)
