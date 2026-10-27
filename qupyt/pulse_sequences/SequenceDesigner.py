from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable, Mapping
from numbers import Real
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray
from termcolor import colored

from qupyt.pulse_sequences.awg_sequence import (
    CompiledBlock,
    CompiledSequence,
    write_awg_archive,
)
from qupyt.pulse_sequences.awg_sequence import parse_awg_flag as parse_awg_flag
from qupyt.set_up import get_seq_dir
from qupyt.utils.errors import cleanup_after_error

# Bump when numeric compilation changes so older artifacts are regenerated.
NUMERIC_SEQUENCE_CACHE_VERSION = 2


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
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
            raise ValueError(
                f"Repeat count for block {block!r} must be a positive integer; got {repeat!r}"
            )


def validate_awg_duration(value: object, context: str) -> float:
    """Require a finite positive duration in microseconds."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"{context} must be a finite and positive duration in us; got {value!r}"
        )
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            f"{context} must be a finite and positive duration in us"
        ) from exc
    if not np.isfinite(numeric) or numeric <= 0:
        raise ValueError(
            f"{context} must be a finite and positive duration in us; got {value!r}"
        )
    return numeric


def resolve_awg_block_durations(instructions: Mapping[str, Any]) -> dict[str, float]:
    """Resolve AWG block durations using the scalar default and named overrides.

    Parameters
    ----------
    instructions : Mapping[str, Any]
        Validated logical pulse-sequence instructions, with durations in microseconds.

    Returns
    -------
    dict[str, float]
        Effective duration for each distinct referenced block in sorted order.

    Raises
    ------
    ValueError
        If a duration is invalid or an override references an unsequenced block.
    """
    default_duration = instructions.get("total_duration")
    # Preserve legacy numeric strings, including exponent literals parsed as
    # strings by PyYAML. The new overrides and setter require numeric values.
    if isinstance(default_duration, str):
        try:
            default_duration = float(default_duration)
        except ValueError as exc:
            raise ValueError(
                "total_duration must contain a positive numeric duration"
            ) from exc
    duration = validate_awg_duration(default_duration, "total_duration")
    overrides = instructions.get("block_durations", {})
    if not isinstance(overrides, dict):
        raise ValueError(
            "block_durations must be a mapping of block names to durations in us"
        )
    names = sorted(set(instructions["sequencing_order"]))
    for name, value in overrides.items():
        if not isinstance(name, str) or name not in names:
            raise ValueError(
                f"block_durations references an unsequenced block {name!r}"
            )
        validate_awg_duration(value, f"block_durations[{name!r}]")
    return {name: float(overrides.get(name, duration)) for name in names}


class PulseSequenceYaml:
    """Compile logical AWG YAML into a versioned collection of numeric blocks.

    Parameters
    ----------
    channel_mapping : dict[str, Any]
        Existing numeric channel indices or physical sequencer flag selectors.
    awg_sources : list[int]
        Active physical sources in the order used by numeric channel indices.
    samprate : float
        Sampling rate in samples per second.
    yaml_file : str or pathlib.Path, optional
        Source YAML; relative paths are resolved in the user's sequence directory.
    sample_count_validator : callable, optional
        Validate a block name and its rounded sample count before any allocation.
        This check also runs when the numeric cache is reused.
    """

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
        *,
        sample_count_validator: Callable[[str, int], None] | None = None,
    ) -> None:
        self.yaml_file = Path("sequence_0.yaml" if yaml_file is None else yaml_file)
        if not self.yaml_file.is_absolute():
            self.yaml_file = get_seq_dir() / self.yaml_file
        self.awg_sources = awg_sources
        self.channel_mapping = channel_mapping
        self.samp_rate = float(samprate)  # samples per second
        self.sample_count_validator = sample_count_validator

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
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._cache_path.parent,
                prefix=self._cache_path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = Path(file.name)
                yaml.safe_dump(
                    {
                        "instructions": self._cache_instructions,
                        "artifact": [artifact.st_mtime_ns, artifact.st_size],
                    },
                    file,
                )
            os.replace(temporary_path, self._cache_path)
        except BaseException:
            if temporary_path is not None:
                path_to_remove = temporary_path
                cleanup_after_error(
                    lambda: path_to_remove.unlink(missing_ok=True),
                    "pulse-sequence cache temporary file cleanup",
                )
            raise

    def translate_yaml_to_numeric_instructions(self) -> None:
        logging.info(
            "Translating pulse sequence from %s at sampling_rate=%s",
            self.yaml_file,
            self.samp_rate,
        )
        for selector in self.channel_mapping.values():
            if isinstance(selector, str):
                parse_awg_flag(selector, self.awg_sources)
        unchanged = self._sequence_didnt_change()
        # Compile the same parsed input that was used to establish cache identity.
        sequence_instructions = cast(
            dict[str, Any], self._cache_instructions["sequence"]
        )
        durations = resolve_awg_block_durations(sequence_instructions)
        if not np.isfinite(self.samp_rate) or self.samp_rate <= 0:
            raise ValueError("Pulse sequence requires a positive sampling rate")
        for block, duration in durations.items():
            points = self.samp_rate * duration * 1e-6
            if not np.isfinite(points) or round(points) < 1:
                raise ValueError(
                    f"Block {block!r} must contain at least one sampling point"
                )
            if self.sample_count_validator is not None:
                self.sample_count_validator(block, int(round(points)))
        if unchanged:
            logging.info(
                "Reusing cached numeric pulse sequence for unchanged configuration %s",
                self.yaml_file,
            )
            return
        sequence_order = sequence_instructions["sequencing_order"]
        sequencing_repeats = sequence_instructions["sequencing_repeats"]
        blocks: dict[str, CompiledBlock] = {}
        flag_channels: dict[str, list[str]] = {}
        for block, duration in durations.items():
            # Reuse the original sampler so floating-point evaluation, pulse
            # overwrites, and the established marker packing stay identical.
            seq = PulseSequence(1, duration, self.awg_sources, samprate=self.samp_rate)
            flag_channels[block] = []
            for channel, pulses in sequence_instructions[block].items():
                mapped_channel = self.channel_mapping[channel]
                if isinstance(mapped_channel, str):
                    flag_channels[block].append(mapped_channel)
                    continue
                for pulse in pulses.values():
                    seq.add_pulse(
                        0,
                        float(pulse["start"]),
                        float(pulse["duration"]),
                        channel=mapped_channel,
                        inputtype="time",
                        amplitude=float(pulse["amplitude"]),
                        freq=(float(pulse["frequency"]), float(pulse["phase"])),
                    )
            blocks[block] = CompiledBlock(block, duration, seq.pack_waveforms()[0])
            logging.debug(
                "Compiled AWG block %s: requested_duration_us=%s, samples=%s, timing_warnings=%s",
                block,
                duration,
                seq.num_points,
                seq.warning_counter,
            )
        compiled = CompiledSequence(
            sample_rate_hz=float(self.samp_rate),
            source_channels=tuple(self.awg_sources),
            blocks=blocks,
            step_block_names=tuple(sequence_order),
            repeat_counts=tuple(sequencing_repeats),
            flag_values=flag_channels,
            properties={"Values": "None"},
        )
        try:
            write_awg_archive(get_seq_dir() / "sequence.npz", compiled)
        except Exception:
            logging.exception(
                "Failed to save compiled AWG sequence from %s", self.yaml_file
            )
            raise
        self._record_compiled_sequence()
        logging.info(
            "Translated pulse sequence from %s: blocks=%s, channels=%s, body_duration_s=%s",
            self.yaml_file,
            len(blocks),
            self.awg_sources,
            compiled.body_duration_s,
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
        self.duration = duration
        self.min_time = 1 / samprate
        points = samprate * duration * 1e-6
        if not np.isfinite(points):
            raise ValueError("Pulse sequence sample count must be finite")
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
        self.flag_channels: dict[str, list[str]] = {}
        self.warning_counter = 0
        self.properties: dict[str, Any] = {"Values": "None"}

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
        if (
            not np.isfinite(start)
            or not np.isfinite(duration)
            or not np.isfinite(amplitude)
            or (freq is not None and not all(np.isfinite(value) for value in freq))
        ):
            raise ValueError(
                "Pulse timing, amplitude, frequency and phase must be finite"
            )
        if (
            isinstance(channel, bool)
            or not isinstance(channel, int)
            or not 0 <= channel < 5 * len(self.awg_sources)
        ):
            raise ValueError(f"Invalid numeric AWG channel index {channel!r}")
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

    def pack_waveforms(self) -> NDArray[np.float64]:
        """Pack uniform numeric blocks using the established marker bit ordering."""
        final = np.zeros((self.numseqs, 2 * len(self.awg_sources), self.num_points))
        for source_index, _source in enumerate(self.awg_sources):
            final[:, 2 * source_index, :] = self.pulses[:, 5 * source_index, :]
            final[:, 2 * source_index + 1, :] = (
                self.pulses[:, 5 * source_index + 1, :] * 2**7
                + self.pulses[:, 5 * source_index + 2, :] * 2**6
                + self.pulses[:, 5 * source_index + 3, :] * 2**5
                + self.pulses[:, 5 * source_index + 4, :] * 2**4
            )

        return final

    def make(self, name: str) -> None:
        """Save uniform blocks in the versioned, pickle-free AWG archive format."""
        if self.sequencernames is None or self.sequencer is None:
            raise ValueError(
                "Set sequencernames and sequencer before saving an AWG sequence"
            )
        if not isinstance(self.flag_channels, dict):
            raise ValueError(
                "flag_channels must be a mapping of block names to selector lists; "
                "replace legacy pickled bytes with the original mapping"
            )
        names = sorted(set(self.sequencernames))
        if len(names) != self.numseqs:
            raise ValueError("Unique waveform names must match the numeric block count")
        if any(
            isinstance(count, bool) or not isinstance(count, (int, np.integer))
            for count in self.sequencer
        ):
            raise ValueError("sequencer repeat counts must be positive integers")
        final = self.pack_waveforms()
        compiled = CompiledSequence(
            sample_rate_hz=float(self.samp_rate),
            source_channels=tuple(self.awg_sources),
            blocks={
                block_name: CompiledBlock(
                    block_name, float(self.duration), final[index]
                )
                for index, block_name in enumerate(names)
            },
            step_block_names=tuple(self.sequencernames),
            repeat_counts=tuple(int(count) for count in self.sequencer),
            flag_values=self.flag_channels or {block_name: [] for block_name in names},
            properties=self.properties,
        )
        output_path = get_seq_dir() / (name if name.endswith(".npz") else name + ".npz")
        try:
            write_awg_archive(output_path, compiled)
        except Exception:
            logging.exception(
                "Failed to save numeric pulse sequence to %s", output_path
            )
            raise
        logging.info(
            "Saved numeric pulse sequence to %s: shape=%s, timing_warnings=%s",
            output_path,
            final.shape,
            self.warning_counter,
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
        if self.yaml_sequence.get("block_durations"):
            raise ValueError(
                "block_durations is supported by the AWG path, not PulseBlaster"
            )
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
