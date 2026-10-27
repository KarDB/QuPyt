# pylint: disable=logging-format-interpolation
# pylint: disable=logging-not-lazy
"""Handle AWG input and output."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep, time
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import yaml
from numpy.typing import NDArray
from pulsestreamer import (
    OutputState,
    PulseStreamer,
    Sequence,
    TriggerRearm,
    TriggerStart,
    findPulseStreamers,
)
from termcolor import colored
from tqdm import tqdm

from qupyt import set_up
from qupyt.hardware.visa_handler import VisaObject
from qupyt.mixins import ConfigurationMixin, PulseSequenceError, UpdateConfigurationType
from qupyt.pulse_sequences.awg_sequence import (
    DEFAULT_MAX_SEQUENCE_STEPS,
    DEFAULT_MAX_STEP_REPEAT,
    CompiledSequence,
    build_logical_awg_steps,
    get_awg_sequence_options,
    infer_child_repetitions,
    read_awg_archive,
    validate_awg_limit,
    validate_awg_metadata_lengths,
    validate_awg_repeat_count,
    validate_awg_sequence_plan,
    validate_repeated_child_region,
)
from qupyt.pulse_sequences.SequenceDesigner import (
    PulseBlasterSequence,
    PulseSequenceYaml,
    parse_awg_flag,
    resolve_awg_block_durations,
    validate_sequence_instructions,
)
from qupyt.set_up import get_seq_dir
from qupyt.utils.errors import cleanup_after_error, cleanup_on_exit

try:
    import qupyt.hardware.wrappers.spinapi_adapted as _spinapi

    spapi: Any = _spinapi
except ImportError:
    spapi = None
    logging.warning(
        "Could not load spinapi library".ljust(65, ".")
        + "[failed]\nIf you are not using a PulseBlaster you do not need this!"
    )


def _validate_synchroniser_sequence_mode(
    sequence_instructions: dict[str, Any],
    device_type: str,
    supported_modes: tuple[str, ...],
) -> str:
    """Reject sequence upload modes unsupported by the selected synchroniser.

    awg_sequence_mode is not part of the logical pulse program. It only
    describes how TekAWG should upload that pulse program. Non-AWG devices
    reject unsupported modes explicitly so a Tek-only YAML option is not
    silently ignored.
    """
    mode = str(sequence_instructions.get("awg_sequence_mode", "flat"))
    if mode not in supported_modes:
        message = (
            f"awg_sequence_mode: {mode} is not supported by {device_type}. "
            f"Supported modes for {device_type}: {', '.join(supported_modes)}."
        )
        logging.error(message)
        raise ValueError(message)
    logging.info("%s accepted awg_sequence_mode=%s", device_type, mode)
    return mode


class SynchroniserFactory:
    """Factory for devices that implement the common Synchroniser interface.

    Call create_synchroniser directly without constructing a factory instance.

    Examples
    --------
    >>> sync = SynchroniserFactory.create_synchroniser(
    ...     "MockSynchroniser", {}, {"LASER": 1, "READ": 2}
    ... )
    """

    @staticmethod
    def create_synchroniser(
        sync_type: str, configuration: dict[str, Any], channel_mapping: dict[str, Any]
    ) -> Synchroniser:
        """Create and configure the requested synchroniser.

        Parameters
        ----------
        sync_type : str
            Synchroniser model: SwabInstPS, TekAWG, MockSynchroniser, or PulseBlaster.
        configuration : dict[str, Any]
            Parameters supported by the selected synchroniser.
        channel_mapping : dict[str, Any]
            Mapping from logical channel names to device ports.

        Returns
        -------
        Synchroniser
            Configured instance of the requested synchroniser.

        Raises
        ------
        ValueError
            If the synchroniser type is unknown.
        """
        logging.info("Creating synchroniser type=%s", sync_type)
        try:
            synchroniser_types: dict[
                str, Callable[[dict[str, Any], dict[str, Any]], Synchroniser]
            ] = {
                "SwabInstPS": PStreamer,
                "TekAWG": AWGenerator,
                "MockSynchroniser": MockGenerator,
                "PulseBlaster": PulseBlaster,
            }
            if sync_type not in synchroniser_types:
                raise ValueError(f"Unknown synchroniser type {sync_type}")
            synchroniser = synchroniser_types[sync_type](configuration, channel_mapping)
        except Exception:
            logging.exception("Could not create synchroniser type=%s", sync_type)
            raise
        logging.info("Created synchroniser type=%s", sync_type)
        return synchroniser


class Synchroniser(ABC, ConfigurationMixin):
    """Abstract interface for coordinating a measurement through a pulse sequence.

    Concrete devices parse and play the pulse sequence using their own hardware
    instructions. The default sequence file is ~/.qupyt/sequences/sequence_0.yaml.

    Attributes
    ----------
    address : str
        Device address, supplied through the concrete synchroniser's configuration.

    Notes
    -----
    Configure concrete synchronisers through their configuration dictionaries.
    VISA addresses can use a form such as TCPIP::<ipaddress>::INSTR; available
    settings depend on the selected device.
    """

    attribute_map: UpdateConfigurationType

    def __init__(self) -> None:
        self.address: str
        self.attribute_map = {"address": self._set_address}

    def _set_address(self, address: str) -> None:
        self.address = address

    @abstractmethod
    def load_sequence(self, ps_yaml_file: str | Path) -> None:
        """Read, compile, and upload a YAML pulse sequence.

        Parameters
        ----------
        ps_yaml_file : str or pathlib.Path
            Pulse-sequence file to load. Concrete devices resolve relative names in
            the default sequence directory.
        """

    @abstractmethod
    def run(self) -> None:
        """Enable the outputs and prepare playback.

        Some devices need this step to avoid slow output ramp-up or startup pulses.
        Concrete devices define whether run also plays the sequence or waits for a
        subsequent trigger.
        """

    @abstractmethod
    def open(self) -> None:
        """Establish the synchroniser connection and apply its configuration."""

    @abstractmethod
    def trigger(self) -> None:
        """Start one playback of the measurement sequence.

        One sequence can contain multiple readouts or repeated measurement blocks.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop pulse-sequence playback and deactivate the outputs."""

    @abstractmethod
    def close(self) -> None:
        """Close the synchroniser connection."""


class AWGenerator(VisaObject, Synchroniser):
    """Synchroniser implementation for Tektronix AWGs.

    Flat mode preserves the existing Tek AWG behavior. repeated_child mode uses
    Tek AWG70000-series sequence-asset SCPI and was tested on AWG70001B.
    """

    @dataclass(frozen=True)
    class AWGSequenceStep:
        """One expanded logical AWG sequencer step.

        Repeated-child compression must compare all hardware-visible state, not just
        waveform names; repeat counts and flag states are part of the timing contract.
        """

        waveform: str
        repeat: int
        flags: tuple[str, ...]

    def __init__(
        self, configuration: dict[str, Any], channel_mapping: dict[str, Any]
    ) -> None:
        self.device_type: str = "TekAWG"
        self.samprate: float = 5e9
        self.channel_mapping = channel_mapping
        self.flag_channels = self._extract_flags(channel_mapping)

        self.wavenames: list[str]
        self.seqrepeats: list[int]
        self.compiled_sequence: CompiledSequence
        self.sequence_options: dict[str, Any] = {}
        self.analog_amplitude: float = 1.0
        self.marker_amplitude: float = 1.75
        self.dac_resolution: int = 12
        self.channels: list[int] = [1, 2]
        self.marker_channels: list[int] = [1, 2, 3, 4]
        self.awg_max_sequence_steps: int = DEFAULT_MAX_SEQUENCE_STEPS
        self.awg_max_step_repeat: int = DEFAULT_MAX_STEP_REPEAT

        Synchroniser.__init__(self)
        self.attribute_map["device_type"] = self._set_device_type
        self.attribute_map["sampling_rate"] = self._set_sampling_rate_attribute
        self.attribute_map["channels"] = self._set_channels_attribute
        self.attribute_map["awg_max_sequence_steps"] = self._set_awg_max_sequence_steps
        self.attribute_map["awg_max_step_repeat"] = self._set_awg_max_step_repeat
        if configuration is not None:
            self._update_from_configuration(configuration)
        VisaObject.__init__(self, self.address, self.device_type)
        try:
            self.instance.timeout = 20000
            self._limit_channels_to_available_outputs()
        except BaseException:
            logging.exception("AWG initialization failed at %s", self.address)
            cleanup_after_error(
                lambda: VisaObject.close(self), "AWG initialization cleanup"
            )
            raise

    def _extract_flags(self, channel_mapping: dict[str, int | str]) -> list[str]:
        selectors = []
        for value in channel_mapping.values():
            if isinstance(value, int):
                continue
            # Validate syntax here; active channels are configured afterwards.
            parse_awg_flag(value, [1, 2])
            if value not in selectors:
                selectors.append(value)
        return selectors

    def _validate_flag_channels(self) -> None:
        for selector in self.flag_channels:
            parse_awg_flag(selector, self.channels)

    def open(self) -> None:
        self._limit_channels_to_available_outputs()
        self._configure()

    def close(self) -> None:
        """Keep the VISA connection available for reuse between measurements."""
        logging.debug("AWG cleanup leaves the VISA connection open for reuse")

    def run(self) -> None:
        self.instance.write("awgcontrol:run:immediate")
        logging.info("Turned on AWG output to RUN immediate".ljust(65, ".") + "[done]")
        self.opc_wait()

    def stop(self) -> None:
        self.instance.write("awgcontrol:stop:immediate")
        logging.info("Turned on AWG output to STOP immediate".ljust(65, ".") + "[done]")
        self.opc_wait()

    def trigger(self) -> None:
        self.instance.write("trigger:immediate atrigger")
        logging.debug("Sent trigger to AWG".ljust(65, ".") + "[done]")

    def load_sequence(self, ps_yaml_file: str | Path = "sequence_0.yaml") -> None:
        logging.info("Loading AWG sequence from %s", ps_yaml_file)
        self._limit_channels_to_available_outputs()
        self._validate_flag_channels()
        missing = object()
        previous = {
            name: getattr(self, name, missing)
            for name in (
                "sequence_options",
                "compiled_sequence",
                "wavenames",
                "seqrepeats",
                "flag_values",
            )
        }
        try:
            instructions = self._load_awg_sequence_input(ps_yaml_file)
            self.sequence_options = get_awg_sequence_options(instructions)
            sample_count_validator = (
                self._get_waveform_sample_count_validator(ps_yaml_file)
                if instructions.get("block_durations")
                else None
            )
            sequence_translator = PulseSequenceYaml(
                self.channel_mapping,
                self.channels,
                samprate=self.samprate,
                yaml_file=ps_yaml_file,
                sample_count_validator=sample_count_validator,
            )
            sequence_translator.translate_yaml_to_numeric_instructions()
            self._load_sequence_block(get_seq_dir() / "sequence.npz")
            # Preflight every asset before stopping the current program.
            self._validate_awg_sequence_plan(self._get_logical_awg_steps())
        except BaseException:
            for name, value in previous.items():
                if value is missing:
                    if hasattr(self, name):
                        delattr(self, name)
                else:
                    setattr(self, name, value)
            logging.exception("Could not prepare AWG sequence from %s", ps_yaml_file)
            raise
        self.stop()
        self.instance.write("*CLS")
        self._clear_awg()
        self._upload_waveforms()
        self._raise_on_awg_errors("waveform upload")
        self._sequence("autoseq", nongatereps=1)
        self._raise_on_awg_errors("sequencer construction")
        logging.info("Loaded and sequenced AWG pulse sequence from %s", ps_yaml_file)
        self.opc_wait()

    def _configure(self) -> None:
        self._set_sampling_rate()
        for channel in self.channels:
            self._set_daq_resolution(channel, self.dac_resolution)
            self._set_analog_amplitude(channel, self.analog_amplitude)
            self._set_output_on(channel)
            for marker in self.marker_channels:
                self._set_marker_amplitude(channel, marker, self.marker_amplitude)
        print("\nConfiguring AWG".ljust(65, ".") + colored(" [done]", "green"))
        logging.info(
            "Configured AWG at %s: channels=%s, sampling_rate=%s",
            self.address,
            self.channels,
            self.samprate,
        )

    def _upload_waveform(self, wavename: str, waveform: NDArray[Any]) -> None:
        """Upload one source's analog and packed-marker rows without repacking."""
        if (
            waveform.ndim != 2
            or waveform.shape[0] != 2
            or waveform.shape[1] < 1
            or waveform.dtype.kind not in "fiu"
            or not np.all(np.isfinite(waveform))
        ):
            raise ValueError(
                f"AWG waveform {wavename!r} requires two finite numeric rows "
                "and at least one sample."
            )
        marker_row = waveform[1]
        if (
            np.any(marker_row < 0)
            or np.any(marker_row > 255)
            or np.any(marker_row != np.floor(marker_row))
        ):
            raise ValueError(
                f"AWG waveform {wavename!r} marker samples must be integers in [0, 255]."
            )
        # Preserve the existing float32 wire format and legacy marker byte values.
        # Only markers require conversion; converting the full array wastes memory.
        markers = marker_row.astype(np.uint8)
        self.instance.write('wlist:waveform:delete "' + wavename + '"')
        self.instance.write(
            'wlist:waveform:new "' + wavename + '",' + str(waveform.shape[1]) + ",real"
        )
        self.instance.write_binary_values(
            'wlist:waveform:data "' + wavename + '",',
            waveform[0, :],
            datatype="f",
            is_big_endian=False,
            header_fmt="ieee",
        )
        self.instance.write_binary_values(
            'wlist:waveform:marker:data "' + wavename + '",',
            markers,
            datatype="B",
            is_big_endian=False,
            header_fmt="ieee",
        )
        logging.info(f"Uploaded waveform {wavename}".ljust(65, ".") + "[done]")
        self.opc_wait()

    def _sequence(self, seqname: str, nongatereps: int = 1) -> None:
        mode = self._get_awg_sequence_mode()
        logical_steps = self._get_logical_awg_steps()
        self._validate_awg_sequence_plan(logical_steps, nongatereps=nongatereps)
        if mode == "flat":
            self._sequence_flat(seqname, logical_steps, nongatereps=nongatereps)
            return
        if mode == "repeated_child":
            self._sequence_repeated_child(
                seqname, logical_steps, nongatereps=nongatereps
            )
            return
        raise ValueError(
            f"Unsupported awg_sequence_mode: {mode!r}. Supported values are "
            "'flat' and 'repeated_child'."
        )

    def _sequence_flat(
        self,
        seqname: str,
        logical_steps: list[AWGenerator.AWGSequenceStep],
        nongatereps: int = 1,
    ) -> None:
        if len(logical_steps) > self.awg_max_sequence_steps:
            raise ValueError(
                f"Expanded AWG sequence has {len(logical_steps)} steps, exceeding "
                f"the configured AWG limit of {self.awg_max_sequence_steps}. Enable "
                "awg_sequence_mode: repeated_child or reduce the sequence."
            )
        summary = f"Tek AWG sequence mode: flat\nLogical steps: {len(logical_steps)}"
        print(summary)
        logging.info(
            "Tek AWG sequence mode: flat; logical_steps=%s", len(logical_steps)
        )
        print("Setting up sequencer".ljust(65, "."), end="")
        # AWG70K sequence assets have one or more tracks. Each active AWG output
        # is assigned one track of the same sequence asset via SOURCE:CASS.
        tracks = len(self.channels)
        self.instance.write('slist:sequence:delete "sub"')
        self.instance.write(f'slist:sequence:new "sub",{len(logical_steps)},{tracks}')
        self.instance.write('slist:sequence:event:jtiming "sub", immediate')
        for i, step in enumerate(logical_steps, start=1):
            for track, channel in enumerate(self.channels, start=1):
                self._configure_waveform_step("sub", i, step, channel, track)

        self.instance.write(f'slist:sequence:delete "{seqname}"')
        self.instance.write(f'slist:sequence:new "{seqname}",2,{tracks}')
        self.instance.write(f'slist:sequence:step2:goto "{seqname}",first')
        self.instance.write(f'slist:sequence:event:jtiming "{seqname}", immediate')

        # for gating pulse
        self.instance.write(f'slist:sequence:step1:goto "{seqname}", first')
        self.instance.write(f'slist:sequence:step1:ejinput "{seqname}", ATR')
        self.instance.write(f'slist:sequence:step1:ejump "{seqname}", 2')
        for track, channel in enumerate(self.channels, start=1):
            self.instance.write(
                f'slist:sequence:step1:tasset{track}:waveform "{seqname}","{self.wavenames[0]}_{channel}"'
            )

        # for actual seq
        self.instance.write(f'slist:sequence:step2:rcount "{seqname}", {nongatereps}')
        for track in range(1, tracks + 1):
            self.instance.write(f'SLIS:SEQ:STEP2:TASS{track}:SEQ "{seqname}","sub"')

        for track, channel in enumerate(self.channels, start=1):
            self.instance.write(f'SOURCE{channel}:CASS:SEQUENCE "{seqname}",{track}')
        self.opc_wait()
        print(colored(" [done]", "green"))
        logging.info(
            "Configured flat AWG sequencer %s: logical_steps=%s, tracks=%s",
            seqname,
            len(logical_steps),
            tracks,
        )

    def _sequence_repeated_child(
        self,
        seqname: str,
        logical_steps: list[AWGenerator.AWGSequenceStep],
        nongatereps: int = 1,
    ) -> None:
        # This parent/child sequencer upload uses Tek AWG70000-series SCPI and
        # was tested on TEKTRONIX AWG70001B firmware FV:8.1.0266.0. Similar Tek
        # models should be verified before enabling repeated_child in the lab.
        prefix, child, repetitions, suffix = self._validate_repeated_child_region(
            logical_steps
        )
        compact_body_steps = len(prefix) + 1 + len(suffix)
        top_level_steps = 1 + compact_body_steps
        summary = (
            "Tek AWG sequence mode: repeated_child\n"
            f"Expanded logical steps: {len(logical_steps)}\n"
            f"Prefix steps: {len(prefix)}\n"
            f"Child steps: {len(child)}\n"
            f"Child repetitions: {repetitions}\n"
            f"Suffix steps: {len(suffix)}\n"
            f"Compact measurement steps: {compact_body_steps}\n"
            f"Top-level steps: {top_level_steps}\n"
            f"Unique waveforms: {len(set(self.wavenames))}"
        )
        print(summary)
        logging.info(
            "Tek AWG sequence mode: repeated_child; expanded_logical_steps=%s; "
            "prefix_steps=%s; child_steps=%s; child_repetitions=%s; "
            "suffix_steps=%s; compact_measurement_steps=%s; top_level_steps=%s; "
            "unique_waveforms=%s",
            len(logical_steps),
            len(prefix),
            len(child),
            repetitions,
            len(suffix),
            compact_body_steps,
            top_level_steps,
            len(set(self.wavenames)),
        )
        print("Setting up sequencer".ljust(65, "."), end="")
        # Keep waveform memory unchanged: the child is a sequence asset that
        # references already-uploaded waveforms, not a concatenated waveform.
        tracks = len(self.channels)
        child_sequence = "child"
        top_sequence = seqname

        self.instance.write(f'slist:sequence:delete "{child_sequence}"')
        self.instance.write(
            f'slist:sequence:new "{child_sequence}",{len(child)},{tracks}'
        )
        self.instance.write(
            f'slist:sequence:event:jtiming "{child_sequence}", immediate'
        )
        for step_index, step in enumerate(child, start=1):
            for track, channel in enumerate(self.channels, start=1):
                self._configure_waveform_step(
                    child_sequence, step_index, step, channel, track
                )

        self.instance.write(f'slist:sequence:delete "{top_sequence}"')
        self.instance.write(
            f'slist:sequence:new "{top_sequence}",{top_level_steps},{tracks}'
        )
        self.instance.write(f'slist:sequence:event:jtiming "{top_sequence}", immediate')
        self.instance.write(f'slist:sequence:step1:goto "{top_sequence}", first')
        self.instance.write(f'slist:sequence:step1:ejinput "{top_sequence}", ATR')
        self.instance.write(f'slist:sequence:step1:ejump "{top_sequence}", 2')
        for track, channel in enumerate(self.channels, start=1):
            self.instance.write(
                f'slist:sequence:step1:tasset{track}:waveform "{top_sequence}","{self.wavenames[0]}_{channel}"'
            )

        body_step = 2
        # The top-level sequence preserves the existing trigger-wait step and
        # then runs prefix + repeated child + suffix directly. Avoiding
        # wrapper -> body -> child nesting keeps this compatible with AWG70000B.
        for step in prefix:
            for track, channel in enumerate(self.channels, start=1):
                self._configure_waveform_step(
                    top_sequence, body_step, step, channel, track
                )
            body_step += 1
        self.instance.write(
            f'slist:sequence:step{body_step}:rcount "{top_sequence}",{repetitions}'
        )
        for track in range(1, tracks + 1):
            self.instance.write(
                f'SLIS:SEQ:STEP{body_step}:TASS{track}:SEQ "{top_sequence}","{child_sequence}"'
            )
        body_step += 1
        for step in suffix:
            for track, channel in enumerate(self.channels, start=1):
                self._configure_waveform_step(
                    top_sequence, body_step, step, channel, track
                )
            body_step += 1
        self.instance.write(
            f'slist:sequence:step{top_level_steps}:goto "{top_sequence}", first'
        )
        for track, channel in enumerate(self.channels, start=1):
            self.instance.write(
                f'SOURCE{channel}:CASS:SEQUENCE "{top_sequence}",{track}'
            )
        self.opc_wait()
        print(colored(" [done]", "green"))
        logging.info(
            "Configured repeated-child AWG sequencer %s: top_level_steps=%s, child_steps=%s",
            seqname,
            top_level_steps,
            len(child),
        )

    def _configure_waveform_step(
        self,
        sequence_name: str,
        step_number: int,
        step: AWGenerator.AWGSequenceStep,
        channel: int,
        track: int,
    ) -> None:
        self.instance.write(
            f'slist:sequence:step{step_number}:rcount "{sequence_name}",{step.repeat}'
        )
        self.instance.write(
            f'slist:sequence:step{step_number}:tasset{track}:waveform "{sequence_name}","{step.waveform}_{channel}"'
        )
        # Resolve physical channels to tracks through the configured ordering.
        # Combine overlapping legacy/qualified selectors before writing each flag.
        targeted_flags = {
            flag
            for selector in self.flag_channels
            for flag, source in [parse_awg_flag(selector, self.channels)]
            if source is None or source == channel
        }
        present_flags = {
            flag
            for selector in step.flags
            for flag, source in [parse_awg_flag(selector, self.channels)]
            if source is None or source == channel
        }
        for flag in sorted(targeted_flags):
            flag_value = "HIGH" if flag in present_flags else "LOW"
            self.instance.write(
                f'slist:sequence:step{step_number}:tflag{track}:{flag}flag "{sequence_name}",{flag_value}'
            )

    def _get_awg_sequence_mode(self) -> str:
        return str(self.sequence_options.get("awg_sequence_mode", "flat"))

    def _validate_awg_sequence_plan(
        self, steps: list[AWGenerator.AWGSequenceStep], nongatereps: int = 1
    ) -> None:
        """Check mode, complete resource use, and child layout before upload."""
        mode = self._get_awg_sequence_mode()
        validate_awg_sequence_plan(
            steps,
            mode,
            self.sequence_options.get("awg_repeated_child", {}),
            max_sequence_steps=self.awg_max_sequence_steps,
            max_step_repeat=self.awg_max_step_repeat,
            device_type="Tek AWG",
            nongatereps=nongatereps,
        )
        if mode == "repeated_child":
            self._raise_if_repeated_child_is_unverified_model()

    def _get_logical_awg_steps(self) -> list[AWGenerator.AWGSequenceStep]:
        validate_awg_metadata_lengths(self.wavenames, self.seqrepeats, "Tek AWG")
        self._validate_flag_channels()
        for selectors in self.flag_values.values():
            for selector in selectors:
                parse_awg_flag(selector, self.channels)
        return build_logical_awg_steps(
            self.wavenames,
            self.seqrepeats,
            AWGenerator.AWGSequenceStep,
            max_step_repeat=self.awg_max_step_repeat,
            device_type="Tek AWG",
            flag_values=self.flag_values,
        )

    def _validate_awg_repeat_count(self, repeat: int, context: str) -> None:
        validate_awg_repeat_count(repeat, self.awg_max_step_repeat, context, "Tek AWG")

    def _validate_repeated_child_region(
        self, steps: list[AWGenerator.AWGSequenceStep]
    ) -> tuple[
        list[AWGenerator.AWGSequenceStep],
        list[AWGenerator.AWGSequenceStep],
        int,
        list[AWGenerator.AWGSequenceStep],
    ]:
        return validate_repeated_child_region(
            steps,
            self.sequence_options.get("awg_repeated_child", {}),
            max_sequence_steps=self.awg_max_sequence_steps,
            max_step_repeat=self.awg_max_step_repeat,
            device_type="Tek AWG",
        )

    def _infer_child_repetitions(
        self,
        steps: list[AWGenerator.AWGSequenceStep],
        prefix_steps: int,
        child: list[AWGenerator.AWGSequenceStep],
    ) -> int:
        return infer_child_repetitions(steps, prefix_steps, child)

    def _load_awg_sequence_options(self, ps_yaml_file: str | Path) -> dict[str, Any]:
        return get_awg_sequence_options(self._load_awg_sequence_input(ps_yaml_file))

    def _load_awg_sequence_input(self, ps_yaml_file: str | Path) -> dict[str, Any]:
        """Read validated YAML for upload options and explicit duration detection."""
        yaml_file = Path(ps_yaml_file)
        if not yaml_file.is_absolute():
            yaml_file = get_seq_dir() / yaml_file
        with open(yaml_file, "r", encoding="utf-8") as file:
            sequence_instructions = yaml.safe_load(file)
        validate_sequence_instructions(sequence_instructions)
        return cast(dict[str, Any], sequence_instructions)

    def _get_waveform_sample_count_validator(
        self, ps_yaml_file: str | Path
    ) -> Callable[[str, int], None]:
        """Query connected waveform limits for explicitly sized AWG blocks.

        Read-only capability queries run before numeric allocation or stopping
        playback. Legacy scalar-duration sequences retain their existing device
        call sequence. Returned limits reflect the current instrument mode;
        unsuitable lengths are rejected without padding or changing durations.
        """
        limits = {}
        for key in ("LMINIMUM", "GRANULARITY", "LMAXIMUM"):
            command = f"WLIST:WAVEFORM:{key}?"
            try:
                response = str(self.instance.query(command)).strip()
            except Exception as exc:
                raise ValueError(
                    f"Cannot validate AWG block lengths for {ps_yaml_file}: "
                    f"query {command} failed."
                ) from exc
            if re.fullmatch(r"\+?[0-9]+", response) is None or int(response) <= 0:
                raise ValueError(
                    f"Cannot validate AWG block lengths for {ps_yaml_file}: "
                    f"query {command} returned {response!r}; expected a positive integer."
                )
            limits[key] = int(response)
        minimum = limits["LMINIMUM"]
        maximum = limits["LMAXIMUM"]
        granularity = limits["GRANULARITY"]
        if minimum > maximum:
            raise ValueError(
                f"Cannot validate AWG block lengths for {ps_yaml_file}: "
                f"reported minimum {minimum} exceeds maximum {maximum}."
            )
        logging.info(
            "AWG waveform limits for %s: minimum=%s, maximum=%s, granularity=%s samples",
            ps_yaml_file,
            minimum,
            maximum,
            granularity,
        )

        def validate_sample_count(block_name: str, sample_count: int) -> None:
            # One REAL sample occupies four bytes. IEEE definite-length binary
            # blocks have at most nine payload-length digits; chunking requires
            # an explicit offset-based transfer implementation.
            single_transfer_maximum = 249_999_999
            if sample_count > single_transfer_maximum:
                raise ValueError(
                    f"AWG block {block_name!r} in {ps_yaml_file} has {sample_count} "
                    f"samples, exceeding the single-transfer limit of "
                    f"{single_transfer_maximum} float32 samples. Connected limits: "
                    f"minimum={minimum}, maximum={maximum}, "
                    f"granularity={granularity} samples. Chunked waveform uploads "
                    "are not implemented; reduce the block duration or sampling rate."
                )
            if not minimum <= sample_count <= maximum or sample_count % granularity:
                raise ValueError(
                    f"AWG block {block_name!r} in {ps_yaml_file} has {sample_count} "
                    f"samples, which violates connected waveform limits: "
                    f"minimum={minimum}, maximum={maximum}, "
                    f"granularity={granularity} samples. Adjust the block duration "
                    "or sampling rate; automatic padding is disabled."
                )

        return validate_sample_count

    def _read_awg_errors(self) -> list[str]:
        errors = []
        for _ in range(100):
            response = str(self.instance.query("SYSTEM:ERROR?")).strip()
            code_text, _, message = response.partition(",")
            try:
                code = int(code_text)
            except ValueError:
                errors.append(response)
                break
            if code == 0:
                break
            errors.append(f"{code},{message}".rstrip(","))
        else:
            errors.append("AWG error queue did not drain after 100 reads")
        return errors

    def _raise_on_awg_errors(self, context: str) -> None:
        errors = self._read_awg_errors()
        if errors:
            logging.error(
                "Tektronix AWG reported errors during %s: %s",
                context,
                "; ".join(errors),
            )
            raise RuntimeError(
                f"Tektronix AWG reported errors during {context}: " + "; ".join(errors)
            )

    def _load_sequence_block(self, seqname: Path) -> None:
        """Validate a pickle-free archive before replacing the loaded state."""
        try:
            sequence = read_awg_archive(
                seqname,
                expected_source_channels=self.channels,
                expected_sample_rate_hz=self.samprate,
            )
            self._validate_flag_channels()
            for block_name, selectors in sequence.flag_values.items():
                for selector in selectors:
                    try:
                        parse_awg_flag(selector, self.channels)
                    except ValueError as exc:
                        raise ValueError(
                            f"AWG archive {seqname}, block {block_name!r}: {exc}"
                        ) from exc
            build_logical_awg_steps(
                sequence.step_block_names,
                sequence.repeat_counts,
                AWGenerator.AWGSequenceStep,
                max_step_repeat=self.awg_max_step_repeat,
                device_type="Tek AWG",
                flag_values=sequence.flag_values,
            )
        except Exception:
            logging.exception("Could not load AWG archive %s", seqname)
            raise
        self.compiled_sequence = sequence
        self.seqrepeats = list(sequence.repeat_counts)
        self.wavenames = list(sequence.step_block_names)
        self.flag_values = sequence.flag_values
        logging.info(
            "Loaded AWG archive %s: unique_blocks=%s, body_duration_s=%s",
            seqname,
            len(sequence.blocks),
            sequence.body_duration_s,
        )

    @property
    def waveform_block(self) -> NDArray[Any]:
        """Return the legacy dense view when all blocks have equal sample counts.

        Unequal blocks remain separate arrays in ``compiled_sequence.blocks``.
        The dense view is constructed on demand rather than kept as a second
        in-memory copy of every uploaded sample.
        """
        blocks = self.compiled_sequence.blocks
        samples = [blocks[name].samples for name in sorted(blocks)]
        if len({array.shape[1] for array in samples}) != 1:
            raise ValueError(
                "AWG blocks have different sample counts; inspect "
                "compiled_sequence.blocks[name].samples instead of waveform_block."
            )
        return np.stack(samples)

    def _clear_awg(self) -> None:
        self.instance.write("slist:sequence:delete all")
        self.instance.write("wlist:waveform:delete all")
        logging.info("Clear all AWG slist and wlist".ljust(65, ".") + "[done]")
        self.opc_wait()

    def _upload_waveforms(self) -> None:
        time_1 = time()
        sorted_wavenames = sorted(set(self.wavenames))
        for wavename in tqdm(
            sorted_wavenames,
            total=len(sorted_wavenames),
            ascii=True,
            desc="uploading waveforms",
        ):
            for channel_index, channel in enumerate(self.channels):
                self._upload_waveform(
                    f"{wavename}_{channel}",
                    self.compiled_sequence.blocks[wavename].samples[
                        channel_index * 2 : (channel_index * 2) + 2
                    ],
                )
                self.opc_wait()
        logging.info(
            f"Uploaded Tektronix AWG waveforms in {time() - time_1} seconds".ljust(
                65, "."
            )
            + "[done]"
        )

    def _set_output_on(self, channel: int) -> None:
        self.instance.write(f"outp{channel} on")
        logging.info(f"Set channel{channel} output to on".ljust(65, ".") + "[done]")

    def _set_daq_resolution(self, channel: int, dac_resolution: int) -> None:
        self.instance.write(f"source{channel}:dac:resolution {dac_resolution}")
        logging.info("Set AWG channel%s resolution to %s bits", channel, dac_resolution)

    def _set_sampling_rate(self) -> None:
        self.instance.write(f"source:frequency {self.samprate}")
        logging.info("AWG sampling rate".ljust(65, ".") + f"{self.samprate}")
        self.opc_wait()

    def _set_marker_amplitude(
        self, channel: int, marker: int, voltage: float = 1.75
    ) -> None:
        self.instance.write(
            f"SOURCE{channel}:MARKER{marker}:VOLTAGE:LEVEL:IMMEDIATE:HIGH {voltage}"
        )
        self.opc_wait()
        logging.info(
            f"Set AWG channel{channel} marker{marker} to / V".ljust(65, ".")
            + f"{voltage}"
        )

    def _set_analog_amplitude(self, channel: int, amplitude: float = 1.0) -> None:
        # amplitude is given in fractions of the max amplitude.
        self.instance.write(
            f"source{channel}:voltage:level:immediate:amplitude {amplitude}"
        )
        self.opc_wait()
        logging.info(
            f"Set AWG channel{channel} analog amplitude to".ljust(65, ".")
            + f"{amplitude}"
        )

    def _set_device_type(self, device_type: str) -> None:
        self.device_type = device_type

    def _set_sampling_rate_attribute(self, sampling_rate: float) -> None:
        # YAML spellings such as 5.0e9 can arrive as strings; the compiler has
        # always accepted them by converting to float before numeric sampling.
        self.samprate = float(sampling_rate)

    def _set_channels_attribute(self, channels: list[int]) -> None:
        self.channels = channels

    def _set_awg_max_sequence_steps(self, max_sequence_steps: int) -> None:
        self.awg_max_sequence_steps = validate_awg_limit(
            max_sequence_steps, "awg_max_sequence_steps", "Tek AWG"
        )

    def _set_awg_max_step_repeat(self, max_step_repeat: int) -> None:
        self.awg_max_step_repeat = validate_awg_limit(
            max_step_repeat, "awg_max_step_repeat", "Tek AWG"
        )

    def _limit_channels_to_available_outputs(self) -> None:
        try:
            identity = self._query_awg_identity()
        except Exception:
            logging.warning(
                "Could not query AWG identity for output channel detection",
                exc_info=True,
            )
            raise
        available_channels = self._detect_awg70000_available_channels(identity)
        if available_channels is None:
            return
        usable_channels = [
            channel for channel in self.channels if channel in available_channels
        ]
        if not usable_channels:
            raise ValueError(
                "Configured Tek AWG channels do not match the detected "
                f"AWG70000-series outputs. channels={self.channels}, "
                f"available_channels={available_channels}, idn={identity.strip()!r}."
            )
        if usable_channels != self.channels:
            message = (
                "Restricting Tek AWG channels to detected AWG70000-series "
                f"outputs: requested {self.channels}, available "
                f"{available_channels}, using {usable_channels}, "
                f"idn={identity.strip()!r}."
            )
            print(message)
            logging.warning(message)
            self.channels = usable_channels

    def _detect_awg70000_available_channels(self, identity: str) -> list[int] | None:
        """Return available output channels for known Tek AWG70000-series models.

        The AWG70001 is single-output, while AWG70002 models have two outputs.
        For unknown models, return None and leave the user-configured channels
        unchanged instead of guessing hardware capabilities.
        """
        if "AWG70001" in identity:
            return [1]
        if "AWG70002" in identity:
            return [1, 2]
        return None

    def _query_awg_identity(self) -> str:
        return str(self.instance.query("*IDN?"))

    def _raise_if_repeated_child_is_unverified_model(self) -> None:
        try:
            identity = self._query_awg_identity()
        except Exception:
            logging.warning(
                "Could not query AWG identity for repeated-child model verification",
                exc_info=True,
            )
            raise
        if "AWG5202" in identity:
            message = (
                "awg_sequence_mode: repeated_child is not enabled for detected "
                f"AWG model {identity.strip()!r}. The current repeated_child "
                "SCPI path was tested on TEKTRONIX AWG70001B and uses "
                "AWG70000-series sequence asset/track commands. AWG5202 may be "
                "similar, but it must be verified with the programming manual "
                "or a hardware test before this guard is removed. Use flat mode "
                "on AWG5202 for now."
            )
            logging.error(message)
            raise ValueError(message)

    def plot_waveform(self, wavename: str) -> None:
        """Query and plot an uploaded AWG waveform and its markers.

        Parameters
        ----------
        wavename : str
            Identifier of the uploaded waveform block to inspect.
        """
        marker = self.instance.query_binary_values(
            'wlist:waveform:marker:data? "' + wavename + '"', datatype="B"
        )
        analog = self.instance.query_binary_values(
            'wlist:waveform:data? "' + wavename + '"'
        )
        markers = np.zeros((4, len(marker)))
        print(np.shape(markers), np.shape(analog))
        for i, mark in enumerate(marker):
            num = bin(mark)[2:]
            if not len(num) < 5:
                for j in range(1, 5):
                    try:
                        markers[-j, i] = int(num[-(j + 4)])
                    except (IndexError, ValueError):
                        break

        plt.plot(np.asarray((analog)) * 0.5 + 4)
        plt.plot(markers[0, :] * 0.5 + 3)
        plt.plot(markers[1, :] * 0.5 + 2)
        plt.plot(markers[2, :] * 0.5 + 1)
        plt.plot(markers[3, :] * 0.5 + 0)
        plt.show()


class PStreamer(Synchroniser):
    def __init__(
        self, configuration: dict[str, Any], channel_mapping: dict[str, Any]
    ) -> None:
        super().__init__()
        self.channel_mapping = channel_mapping
        self._update_from_configuration(configuration)
        self.initial_configuration_dict = configuration
        if configuration["address"] == "None":
            self._find_pulse_streamers()

    def __repr__(self) -> str:
        return f"PStreamer(configuration: {self.initial_configuration_dict}, channel_mapping: {self.channel_mapping})"

    def __str__(self) -> str:
        return f"Synchronizer of type PStreamer(configuration: {self.initial_configuration_dict}, channel_mapping: {self.channel_mapping})"

    def _find_pulse_streamers(self) -> None:
        devices = findPulseStreamers()
        if devices:
            logging.info(
                "Detected PulseStreamer devices: %s; selecting %s",
                devices,
                devices[0][0],
            )
            print("Detected PulseStreamer: ")
            print(devices)
            print("----------------------------------------------------\n")
            self.address = devices[0][0]
        else:
            print("No PulseStreamers found by autodetect.\n-------------------\n")
            raise ConnectionError("No PulseStreamer devices found by autodetect")

    def open(self) -> None:
        """Connect to the PulseStreamer at its configured address."""
        # if no IP adress is provided, try to detect one:
        try:
            self.pulser = PulseStreamer(self.address)
            logging.info(
                f"Pulse Streamer connected at {self.address}".ljust(65, ".") + "[done]"
            )
            print("Connected.\n-------------------------------------------\n")
        except Exception:
            logging.exception("Failed to connect PulseStreamer at %s", self.address)

            raise

    def close(self) -> None:
        """Record cleanup without closing a PulseStreamer connection.

        Notes
        -----
        The driver exposes no explicit connection-close operation.
        """
        logging.debug("PulseStreamer cleanup requires no explicit connection close")

    def stop(self) -> None:
        """Stop playback and force the configured final output state."""
        try:
            # define the final state of the Pulsestreamer
            _ = OutputState.ZERO()
            # force the final state.
            self.pulser.forceFinal()
            # print a text if the program has successfully ended
            if not self.pulser.isStreaming():
                print("Stop PulseStreamer: The sequence has been stoped.")
                logging.info(
                    "Pulse Streamer: stopped pulse sequence execution".ljust(65, ".")
                    + "[done]"
                )
            else:
                raise RuntimeError(
                    "PulseStreamer is still streaming after forceFinal()"
                )

        except Exception:
            logging.exception(
                "Pulse Streamer: error stopping pulse sequence".ljust(65, ".")
                + "[failed]"
            )

            raise

    def writeDigSeq(self, channel_key: str) -> list[tuple[int, int]]:
        """Compile the pulses for one PulseStreamer channel.

        Parameters
        ----------
        channel_key : str
            Logical channel name, such as LASER, MW, or READ.

        Returns
        -------
        list[tuple[int, int]]
            Pairs of duration in nanoseconds and digital output level.

        Raises
        ------
        KeyError
            If the requested channel is absent from the pulse list.
        PulseSequenceError
            If the pulses exceed the configured total sequence duration.
        """
        self.channel_key = channel_key
        self.seq = []  # sequence of the channel
        pointer_i = 0  # pointer in time
        # Check if the asked channel_key exists in the file
        if self.channel_key not in list(self.pulse_list.keys()):
            logging.error("KeyError: No element named " + str(self.channel_key) + ".")
            raise KeyError

        for i in range(len(self.pulse_list[self.channel_key])):
            # Start and length of the pulse turned into
            # float in case it has str format.
            start_pulse_i = (
                float(
                    self.pulse_list.get(self.channel_key)
                    .get("pulse" + str(i + 1))
                    .get("start")
                )
                * 1e3
            )
            len_pulse_i = (
                float(
                    self.pulse_list.get(self.channel_key)
                    .get("pulse" + str(i + 1))
                    .get("duration")
                )
                * 1e3
            )
            # Check if pulses are well defined
            if pointer_i > start_pulse_i:
                logging.error(
                    "Error: "
                    + self.channel_key
                    + " sequence definition makes no sense, pulses are overlaping!"
                )
                raise PulseSequenceError

            # Check for unsupported analog signals
            freq_pulse_i = (
                self.pulse_list.get(self.channel_key)
                .get("pulse" + str(i + 1))
                .get("frequency")
            )
            if freq_pulse_i != 0:
                logging.warning(
                    "Warning: Frequency different than 0. This programm does not support analog signals. Set frequency to 0."
                )
                raise PulseSequenceError

            ampl_pulse_i = (
                self.pulse_list.get(self.channel_key)
                .get("pulse" + str(i + 1))
                .get("amplitude")
            )
            if ampl_pulse_i != 1 or ampl_pulse_i is None:
                logging.warning(
                    "Warning: Amplitude of the pulse different than 1 (can only be 0 or 1). This programm does not support analog signals. Set amplitude to 1."
                )
                raise PulseSequenceError

            phase_pulse_i = (
                self.pulse_list.get(self.channel_key)
                .get("pulse" + str(i + 1))
                .get("phase")
            )
            if phase_pulse_i != 0:
                logging.warning(
                    "Warning: Phase of the pulse different than 0. This programm does not support analog signals. Set amplitude to 1."
                )
                raise PulseSequenceError

            # Check for non-multples of the sampling time.
            if (start_pulse_i - round(start_pulse_i) != 0) or (
                len_pulse_i - round(len_pulse_i) != 0
            ):
                logging.warning(
                    "Warning: Sampling unit is 1ns. Time values are being rounded."
                )
                # Round the values and turn into integers
                start_pulse_i = int(round(start_pulse_i))
                len_pulse_i = int(round(len_pulse_i))
                # Append a low and a high for each pulse: ___----
                self.seq.append((start_pulse_i - pointer_i, 0))
                self.seq.append((len_pulse_i, 1))
                # Update pointer
                pointer_i = start_pulse_i + len_pulse_i

            else:
                # Turn into integers
                start_pulse_i = int(start_pulse_i)
                len_pulse_i = int(len_pulse_i)
                # Append a low and a high for each pulse: ___----
                self.seq.append((start_pulse_i - pointer_i, 0))
                self.seq.append((len_pulse_i, 1))
                # Update pointer
                pointer_i = start_pulse_i + len_pulse_i

        # Check if the sequence is longer than the defined total time.
        if pointer_i > self.total_duration and self.total_duration_unparsed != "ignore":
            logging.error(
                "Error: %s duration exceeds the defined total time.", self.channel_key
            )
            raise PulseSequenceError
        # Add the final low to make the sequence last its length
        if self.total_duration_unparsed != "ignore":
            self.seq.append((int(self.total_duration - pointer_i), 0))
        return self.seq

    def plot_sequence(self) -> None:
        """Plot the loaded sequence if the PulseStreamer is streaming."""
        # check if the sequence has been loaded.
        if self.pulser.isStreaming() is True:
            self.sequence.plot()
        else:
            print(
                "Warning (plot_sequence()): No sequence is streaming in the PulseStreamer."
            )

    def load_sequence(self, ps_yaml_file: str | Path = "sequence_0.yaml") -> None:
        """Compile the YAML pulse sequence into the PulseStreamer channel program.

        Parameters
        ----------
        ps_yaml_file : str or pathlib.Path, optional
            Pulse-sequence path, defaulting to sequence_0.yaml in the sequence directory.
        """
        try:
            # Selected folder:
            self.yaml_file = set_up.get_seq_dir() / ps_yaml_file
            logging.info("Loading PulseStreamer sequence from %s", self.yaml_file)
            with open(self.yaml_file, "r", encoding="utf-8") as file:
                full_pulse_list = yaml.safe_load(file)
            validate_sequence_instructions(full_pulse_list)
            if full_pulse_list.get("block_durations"):
                raise ValueError(
                    "PulseStreamer does not support AWG block_durations overrides."
                )
            # PulseStreamer consumes the expanded pulse program directly; it
            # does not implement Tek AWG parent/child sequencer compression.
            _validate_synchroniser_sequence_mode(
                full_pulse_list, "PulseStreamer", ("flat",)
            )
            sequence_order = full_pulse_list["sequencing_order"]
            sequencing_repeats = full_pulse_list["sequencing_repeats"]

            total_duration_unparsed = full_pulse_list["total_duration"]
            self.total_duration_unparsed = total_duration_unparsed
            if total_duration_unparsed == "ignore":
                self.total_duration = np.inf
            if total_duration_unparsed != "ignore":
                total_duration = float(total_duration_unparsed) * 1e3  # convert to ns
                if total_duration - round(total_duration) != 0:
                    logging.warning(
                        "Warning: The total duration is not multiple of the\
                                sampling time and is being rounded!".ljust(
                            65, "."
                        )
                        + "[WARNING]"
                    )
                    self.total_duration = int(round(total_duration))
                else:
                    self.total_duration = int(total_duration)

            # Generate Sequence objects for all pulseseqeunce blocks.
            # These will be sequenced together later.
            sequences_to_write = {}
            for block in set(sequence_order):
                sequences_to_write[block] = Sequence()
                self.pulse_list = full_pulse_list[block]
                self.check_types(self.pulse_list)
                for channel in self.pulse_list:
                    sequences_to_write[block].setDigital(
                        self.channel_mapping[channel], self.writeDigSeq(channel)
                    )
            self.sequence = Sequence()
            for block, repetitions in zip(sequence_order, sequencing_repeats):
                self.sequence += repetitions * sequences_to_write[block]
            logging.info(
                "Prepared PulseStreamer sequence from %s: blocks=%s, sequencing_steps=%s",
                self.yaml_file,
                len(sequences_to_write),
                len(sequence_order),
            )

        except Exception:
            logging.exception(
                "Failed to load PulseStreamer sequence from %s", ps_yaml_file
            )

            raise

    def check_types(self, pulse_list: dict[str, dict[str, dict[str, Any]]]) -> None:
        for channel in pulse_list:
            for pulse, pulse_params in pulse_list[channel].items():
                for par, par_value in pulse_params.items():
                    pulse_list[channel][pulse][par] = float(par_value)

    def run(self) -> None:
        """Stream the sequence once and wait for playback to finish.

        Raises
        ------
        TimeoutError
            If playback exceeds the sequence duration plus ten seconds.
        """
        try:
            # never runs the seq
            n_runs = 1

            # reset the device - all outputs 0V
            self.pulser.reset()
            self.pulser.constant(OutputState.ZERO())  # all outputs 0V
            final = OutputState.ZERO()

            # Start the sequence after the upload and disable
            # the retrigger-function
            start = TriggerStart.IMMEDIATE
            rearm = TriggerRearm.MANUAL
            self.pulser.setTrigger(start=start, rearm=rearm)

            # upload the sequence and arm the device
            self.pulser.stream(self.sequence, n_runs, final)
            deadline = monotonic() + self.sequence.getDuration() * 1e-9 + 10
            while self.pulser.isStreaming():
                if monotonic() >= deadline:
                    raise TimeoutError(
                        "PulseStreamer did not finish within the sequence duration plus 10 seconds"
                    )
                sleep(0.01)
            # check if the sequence has been started correctly.
            logging.info("Pulse Strearmer: sent run signal".ljust(65, ".") + "[done]")

        except Exception:
            logging.exception(
                "Pulse Strearmer: problem setting to run".ljust(65, ".") + "[failed]"
            )
            cleanup_after_error(self.stop, "PulseStreamer run cleanup")
            raise

    def trigger(self) -> None:
        """Rearm the PulseStreamer and start the uploaded sequence immediately."""
        self.pulser.rearm()
        self.pulser.startNow()
        logging.debug(
            "Pulse Strearmer: Sent signal to play sequence".ljust(65, ".") + "[done]"
        )


class MockGenerator(Synchroniser):
    def __init__(
        self, configuration: dict[str, Any], channel_mapping: dict[str, Any]
    ) -> None:
        self.channel_mapping = channel_mapping
        self.device_type: str = "MockGenerator"
        self.max_sequence_steps: int = DEFAULT_MAX_SEQUENCE_STEPS
        self.max_step_repeat: int = DEFAULT_MAX_STEP_REPEAT
        self.samprate: float = 5e9
        self.channels: list[int] = [1, 2]
        self.compiled_sequence: CompiledSequence | None = None

        Synchroniser.__init__(self)
        # Retain TekAWG YAML keys so the mock can dry-run the same config.
        self.attribute_map["awg_max_sequence_steps"] = self._set_max_sequence_steps
        self.attribute_map["awg_max_step_repeat"] = self._set_max_step_repeat
        self.attribute_map["sampling_rate"] = self._set_sampling_rate_attribute
        self.attribute_map["channels"] = self._set_channels_attribute
        self.initial_configuration_dict = configuration
        if configuration is not None:
            self._update_from_configuration(configuration)
        logging.info("MockSynchroniser instance created".ljust(65, ".") + "[done]")

    def __repr__(self) -> str:
        return f"MockGenerator(configuration: {self.initial_configuration_dict}, channel_mapping: {self.channel_mapping})"

    def __str__(self) -> str:
        return f"Synchronizer of type MockGenerator(configuration: {self.initial_configuration_dict}, channel_mapping: {self.channel_mapping})"

    def open(self) -> None:
        logging.info("Opened MockSynchroniser".ljust(65, ".") + "[done]")

    def load_sequence(self, ps_yaml_file: str | Path = "sequence_0.yaml") -> None:
        try:
            self.yaml_file = set_up.get_seq_dir() / ps_yaml_file
            logging.info("Loading MockSynchroniser sequence from %s", self.yaml_file)
            with open(self.yaml_file, "r", encoding="utf-8") as file:
                full_pulse_list = yaml.safe_load(file)
            validate_sequence_instructions(full_pulse_list)
            total_duration = (
                float(full_pulse_list["total_duration"]) * 1e3
            )  # convert to ns
            sequence_order = full_pulse_list["sequencing_order"]
            sequencing_repeats = full_pulse_list["sequencing_repeats"]
            # Mock accepts repeated_child as a dry run of the Tek AWG upload
            # strategy. It validates the metadata but still has no hardware
            # sequencer to program.
            _validate_synchroniser_sequence_mode(
                full_pulse_list, "MockSynchroniser", ("flat", "repeated_child")
            )
            compiled_sequence = None
            block_durations = (
                resolve_awg_block_durations(full_pulse_list)
                if "block_durations" in full_pulse_list
                else {
                    name: float(full_pulse_list["total_duration"])
                    for name in sorted(set(sequence_order))
                }
            )
            if full_pulse_list.get("block_durations"):
                compiler = PulseSequenceYaml(
                    self.channel_mapping,
                    self.channels,
                    samprate=self.samprate,
                    yaml_file=self.yaml_file,
                )
                compiler.translate_yaml_to_numeric_instructions()
                compiled_sequence = read_awg_archive(
                    get_seq_dir() / "sequence.npz",
                    expected_source_channels=self.channels,
                    expected_sample_rate_hz=self.samprate,
                )
                body_duration_s = compiled_sequence.body_duration_s
            else:
                body_duration_s = sum(
                    block_durations[name] * count * 1e-6
                    for name, count in zip(
                        sequence_order, sequencing_repeats, strict=True
                    )
                )
            self._validate_awg_sequence_options(
                full_pulse_list,
                sequence_order,
                sequencing_repeats,
                flag_values=(
                    compiled_sequence.flag_values
                    if compiled_sequence is not None
                    else None
                ),
            )
            if total_duration - round(total_duration) != 0:
                logging.warning(
                    "Warning: The total duration is not multiple of the sampling time and is being rounded!"
                )
                self.total_duration = int(round(total_duration))
            else:
                self.total_duration = int(total_duration)

            self.compiled_sequence = compiled_sequence
            self.block_durations = block_durations
            self.body_duration_s = body_duration_s

            logging.info(
                "Loaded sequence for MockSynchroniser from %s: body_duration_s=%s",
                self.yaml_file,
                self.body_duration_s,
            )

        except AttributeError:
            logging.exception(
                "Caught attribute error when writing loading from yaml pulse sequence"
            )

            raise

    def _validate_awg_sequence_options(
        self,
        full_pulse_list: dict[str, Any],
        sequence_order: list[str],
        sequencing_repeats: list[int],
        *,
        flag_values: dict[str, list[str]] | None = None,
    ) -> None:
        """Dry-run TekAWG sequence-mode metadata for mock measurements.

        The mock does not send SCPI commands. It checks
        that the expanded logical sequence can be represented by the configured
        repeated_child parent/child layout, so mock measurements catch the same
        configuration mistakes before the YAML is sent to a real AWG.
        """
        mode = str(full_pulse_list.get("awg_sequence_mode", "flat"))
        if mode != "repeated_child":
            if mode != "flat":
                raise ValueError(
                    f"Unsupported awg_sequence_mode for MockSynchroniser: {mode!r}. "
                    "Supported values are 'flat' and 'repeated_child'."
                )

        previous_options = getattr(self, "sequence_options", {})
        self.sequence_options = get_awg_sequence_options(full_pulse_list)
        try:
            logical_steps = build_logical_awg_steps(
                sequence_order,
                sequencing_repeats,
                AWGenerator.AWGSequenceStep,
                max_step_repeat=self.max_step_repeat,
                device_type="MockSynchroniser AWG",
                flag_values=flag_values,
            )
            validate_awg_sequence_plan(
                logical_steps,
                mode,
                self.sequence_options.get("awg_repeated_child", {}),
                max_sequence_steps=self.max_sequence_steps,
                max_step_repeat=self.max_step_repeat,
                device_type="MockSynchroniser AWG",
            )
            if mode == "flat":
                logging.info("MockSynchroniser AWG sequence mode: flat")
                return
            prefix, child, repetitions, suffix = self._validate_repeated_child_region(
                logical_steps
            )
        finally:
            self.sequence_options = previous_options

        summary = (
            "MockSynchroniser AWG sequence mode: repeated_child\n"
            f"Expanded logical steps: {len(logical_steps)}\n"
            f"Prefix steps: {len(prefix)}\n"
            f"Child steps: {len(child)}\n"
            f"Child repetitions: {repetitions}\n"
            f"Suffix steps: {len(suffix)}"
        )
        print(summary)
        logging.info(
            "MockSynchroniser AWG sequence mode: repeated_child; "
            "expanded_logical_steps=%s; prefix_steps=%s; child_steps=%s; "
            "child_repetitions=%s; suffix_steps=%s",
            len(logical_steps),
            len(prefix),
            len(child),
            repetitions,
            len(suffix),
        )

    def _validate_awg_repeat_count(self, repeat: int, context: str) -> None:
        validate_awg_repeat_count(
            repeat, self.max_step_repeat, context, "MockSynchroniser AWG"
        )

    def _set_max_sequence_steps(self, max_sequence_steps: int) -> None:
        self.max_sequence_steps = validate_awg_limit(
            max_sequence_steps, "awg_max_sequence_steps", "MockSynchroniser"
        )

    def _set_max_step_repeat(self, max_step_repeat: int) -> None:
        self.max_step_repeat = validate_awg_limit(
            max_step_repeat, "awg_max_step_repeat", "MockSynchroniser"
        )

    def _set_sampling_rate_attribute(self, sampling_rate: float) -> None:
        self.samprate = float(sampling_rate)

    def _set_channels_attribute(self, channels: list[int]) -> None:
        self.channels = channels

    def _validate_repeated_child_region(
        self, steps: list[AWGenerator.AWGSequenceStep]
    ) -> tuple[
        list[AWGenerator.AWGSequenceStep],
        list[AWGenerator.AWGSequenceStep],
        int,
        list[AWGenerator.AWGSequenceStep],
    ]:
        return validate_repeated_child_region(
            steps,
            self.sequence_options.get("awg_repeated_child", {}),
            max_sequence_steps=self.max_sequence_steps,
            max_step_repeat=self.max_step_repeat,
            device_type="MockSynchroniser AWG",
        )

    def _infer_child_repetitions(
        self,
        steps: list[AWGenerator.AWGSequenceStep],
        prefix_steps: int,
        child: list[AWGenerator.AWGSequenceStep],
    ) -> int:
        return infer_child_repetitions(steps, prefix_steps, child)

    def run(self) -> None:
        logging.info("Sent run to MockSynchroniser".ljust(65, ".") + "[done]")

    def trigger(self) -> None:
        logging.debug("Sent trigger from MockSynchroniser".ljust(65, ".") + "[done]")

    def stop(self) -> None:
        logging.info("Stopped MockSynchroniser".ljust(65, ".") + "[done]")

    def close(self) -> None:
        logging.info("Closed MockSynchroniser".ljust(65, ".") + "[done]")


class PulseBlaster(Synchroniser):
    """PulseBlaster card synchroniser controlled through SpinAPI.

    Parameters
    ----------
    configuration : dict[str, Any]
        Board settings. sampling_rate selects the core clock in MHz;
        min_instr_clk_cycles selects the minimum instruction duration in clock cycles.
    channel_mapping : dict[str, Any]
        Mapping from logical pulse-channel names to board output bit positions.

    Attributes
    ----------
    samprate : float
        Board core clock frequency in MHz, defaulting to 500.
    pb_min_instr_clk_cycles : int
        Minimum instruction duration in core-clock cycles, defaulting to 5.

    Notes
    -----
    load_sequence compiles a YAML pulse sequence into bit masks and durations.
    program_pb uploads these instructions; run starts playback and stop halts it.
    close releases communication with the board.
    """

    def __init__(
        self, configuration: dict[str, Any], channel_mapping: dict[str, Any]
    ) -> None:
        """Initialize the board clock, instruction limits, and channel mapping.

        Parameters
        ----------
        configuration : dict[str, Any]
            Board configuration, including sampling_rate and min_instr_clk_cycles.
        channel_mapping : dict[str, Any]
            Mapping from logical channel names to output bit positions.

        Raises
        ------
        RuntimeError
            If SpinAPI is unavailable.
        """
        if spapi is None:
            raise RuntimeError(
                "This class requires 'spinapi' by SpinCore to be installed and functional"
            )
        self._is_open = False
        self.device_type: str = "PulseBlaster"
        self.samprate: float = 500  # MHz
        self.pb_min_instr_clk_cycles = 5
        self.channel_mapping = channel_mapping
        Synchroniser.__init__(self)
        self.attribute_map["sampling_rate"] = self._set_sampling_rate_attribute
        self.attribute_map["min_instr_clk_cycles"] = self._set_min_instr_clk_cycles
        self.initial_configuration_dict = configuration
        if configuration is not None:
            self._update_from_configuration(configuration)

    def __repr__(self) -> str:
        return f"PulseBlaster(configuration: {self.initial_configuration_dict}, channel_mapping: {self.channel_mapping})"

    def __str__(self) -> str:
        return f"Synchronizer of type PulseBlaster(configuration: {self.initial_configuration_dict}, channel_mapping: {self.channel_mapping})"

    def open(self) -> None:
        self.configure_pb()

    def load_sequence(self, ps_yaml_file: str | Path = "sequence_0.yaml") -> None:
        yaml_file = Path(ps_yaml_file)
        if not yaml_file.is_absolute():
            yaml_file = get_seq_dir() / yaml_file
        logging.info("Loading PulseBlaster sequence from %s", yaml_file)
        with open(yaml_file, "r", encoding="utf-8") as file:
            # PulseBlaster also consumes the expanded program directly. Keep
            # Tek-only upload modes out of this path with a clear error.
            _validate_synchroniser_sequence_mode(
                yaml.safe_load(file) or {}, "PulseBlaster", ("flat",)
            )
        yaml_sequence_transpiler = PulseBlasterSequence(self.channel_mapping, yaml_file)
        yaml_sequence_transpiler.parse_pulse_sequence_file()
        channel_bit_mask, pulse_duration_list = yaml_sequence_transpiler.compile()
        self.program_pb(channel_bit_mask, pulse_duration_list)
        logging.info("Loaded PulseBlaster pulse sequence from %s", yaml_file)

    def close(self) -> None:
        """Release communication with the PulseBlaster board and check its status."""
        # Close the communication with the PB board
        if not self._is_open:
            return
        status = spapi.pb_close()
        self.error_catcher(status)
        self._is_open = False
        logging.info("Closed PulseBlaster board connection")

    def run(self) -> None:
        """Start the pulse program and close communication with the board.

        Notes
        -----
        Program execution continues after the communication channel is closed.
        """
        # Returns a 0 on success or a negative number on an error.
        status = spapi.pb_start()
        self.error_catcher(status)
        logging.info("Started execution of PulseBlaster pulse program")

    def trigger(self) -> None:
        status = spapi.pb_start()
        self.error_catcher(status)
        logging.debug("Triggered PulseBlaster pulse program")

    def stop(self) -> None:
        """Stop the pulse program."""
        # Returns a 0 on success or a negative number on an error.
        status = spapi.pb_stop()
        self.error_catcher(status)
        logging.info("Stopped execution of PulseBlaster pulse program")

    def error_catcher(self, status: int) -> None:
        """Check a SpinAPI status and report board errors.

        Parameters
        ----------
        status : int
            SpinAPI result: a negative value indicates an error; zero or a positive
            instruction address indicates success.

        Raises
        ------
        RuntimeError
            If status is negative; the error includes the available board diagnostics.
        """
        if status < 0:
            try:
                detail = spapi.pb_get_error()
            except Exception:
                logging.warning(
                    "Could not retrieve PulseBlaster error details", exc_info=True
                )
                detail = "error details unavailable"
            message = f"PulseBlaster returned status={status}: {detail}"
            logging.error(message)
            raise RuntimeError(message)

    def configure_pb(self) -> None:
        """Initialize the board connection and set its core clock frequency."""
        # 1 - Enable the spincore log file, appears as log.txt in current working directory
        spapi.pb_set_debug(0)
        # Initializing communication with the PulseBlasterESR-PRO board
        if not self._is_open:
            status = spapi.pb_init()
            self.error_catcher(status)
            self._is_open = True
        # Configure the core clock frequency (in MHz)
        spapi.pb_core_clock(self.samprate)
        logging.info("Initialized PulseBlaster board: core_clock=%s MHz", self.samprate)

    @staticmethod
    def pb_inst_pbonly(
        flags: int, instruction: int, instruction_data: int, pulse_length: float
    ) -> int:
        """Write one instruction to the PulseBlaster pulse program.

        Parameters
        ----------
        flags : int
            Bit mask defining the state of each TTL output.
        instruction : int
            SpinAPI instruction opcode.
        instruction_data : int
            Data associated with the instruction opcode.
        pulse_length : float
            Instruction duration in nanoseconds.

        Returns
        -------
        int
            Instruction address on success or a negative error code. -99 indicates
            an invalid parameter.
        """
        return cast(
            int,
            spapi.pb_inst_pbonly(flags, instruction, instruction_data, pulse_length),
        )

    def check_pulse_length_short(
        self, pulse_duration: float, current_option: float
    ) -> float:
        if pulse_duration != current_option:
            logging.warning(
                "pulse duration of %s not possible with PB card. Setting to %s",
                pulse_duration,
                current_option,
            )
            pulse_duration = current_option
            return pulse_duration
        return pulse_duration

    def program_pb(
        self, channel_bit_masks: list[int], pulse_duration_list: list[float]
    ) -> None:
        """Upload the channel bit masks and their corresponding pulse durations.

        Parameters
        ----------
        channel_bit_masks : list[int]
            TTL output bit masks for the instruction sequence.
        pulse_duration_list : list[float]
            Matching instruction durations in microseconds.

        Raises
        ------
        ValueError
            If the lists are empty, have different lengths, or contain durations that
            are not finite and positive.
        """
        if not channel_bit_masks or len(channel_bit_masks) != len(pulse_duration_list):
            raise ValueError(
                "PulseBlaster requires nonempty, equally sized bit masks and durations"
            )
        if any(
            not np.isfinite(duration) or duration <= 0
            for duration in pulse_duration_list
        ):
            raise ValueError("PulseBlaster pulse durations must be finite and positive")
        self.start_programming()
        with cleanup_on_exit(self.stop_programming, "PulseBlaster programming stop"):

            # Send instructions to the pulse program
            # Instruction format:
            # int status pb_inst_pbonly(int bit flags, int instruction,
            # int instruction_data, int pulse_length)
            for i, pulse_duration in zip(
                range(len(channel_bit_masks)), pulse_duration_list
            ):
                # All pulse duration checks in mus.
                if pulse_duration >= 0.01:
                    channel_bit_mask = channel_bit_masks[i]
                else:
                    # Short Pulse Feature:
                    # bits 23-21 controls the number of clock periods
                    if 0 < pulse_duration <= 0.003:
                        # 001 for 1 clock period, 2ns for 500 MHz
                        pulse_duration = self.check_pulse_length_short(
                            pulse_duration, 0.002
                        )
                        channel_bit_mask = channel_bit_masks[i] + 2**21
                    elif 0.003 < pulse_duration <= 0.005:
                        # 010 for 2 clock periods
                        pulse_duration = self.check_pulse_length_short(
                            pulse_duration, 0.004
                        )
                        channel_bit_mask = channel_bit_masks[i] + 2**22
                    elif 0.005 < pulse_duration <= 0.007:
                        # 011 for 3 clock periods
                        pulse_duration = self.check_pulse_length_short(
                            pulse_duration, 0.006
                        )
                        channel_bit_mask = channel_bit_masks[i] + 2**21 + 2**22
                    elif 0.007 < pulse_duration <= 0.009:
                        # 100 for 4 clock periods
                        pulse_duration = self.check_pulse_length_short(
                            pulse_duration, 0.008
                        )
                        channel_bit_mask = channel_bit_masks[i] + 2**23
                    elif 0.009 < pulse_duration < 0.01:
                        # 100 for 4 clock periods
                        pulse_duration = self.check_pulse_length_short(
                            pulse_duration, 0.01
                        )
                        channel_bit_mask = channel_bit_masks[i]

                    # Shortest minimum instruction time is 5 clock periods
                    # i.e. 10 ns for 500 MHz
                    pulse_duration = self.pb_min_instr_clk_cycles

                # Time resolution of PulseBlaster, given by 1/(clock frequency):
                # t_min = 1e3/self.samprate  # in ns
                # upload times as is
                t_min = 1

                # Instructions for pulse sequence
                if i == 0:
                    start_instr_num = spapi.pb_inst_pbonly(
                        channel_bit_mask,
                        spapi.Inst.CONTINUE,
                        0,
                        pulse_duration * t_min * spapi.us,
                    )
                    self.error_catcher(start_instr_num)
                elif i != len(channel_bit_masks) - 1:
                    status = spapi.pb_inst_pbonly(
                        channel_bit_mask,
                        spapi.Inst.CONTINUE,
                        0,
                        pulse_duration * t_min * spapi.us,
                    )
                    self.error_catcher(status)
                else:
                    status = spapi.pb_inst_pbonly(
                        channel_bit_mask,
                        spapi.Inst.BRANCH,
                        start_instr_num,
                        pulse_duration * t_min * spapi.us,
                    )
                    self.error_catcher(status)

            status = spapi.pb_inst_pbonly(
                0, spapi.Inst.STOP, 0, self.pb_min_instr_clk_cycles * t_min * spapi.us
            )
            self.error_catcher(status)

        logging.info(
            "Programmed PulseBlaster board: pulse_instructions=%s",
            len(channel_bit_masks),
        )

    def start_programming(self) -> None:
        """Start programming the PulseBlaster pulse memory."""
        # Returns a 0 on success or a negative number on an error.
        status = spapi.pb_start_programming(spapi.PULSE_PROGRAM)
        self.error_catcher(status)
        logging.debug("Entered PulseBlaster programming mode")

    def stop_programming(self) -> None:
        """Stop programming the PulseBlaster pulse memory."""
        status = (
            spapi.pb_stop_programming()
        )  # Returns a 0 on success or a negative number on an error.
        self.error_catcher(status)

        logging.debug("Exited PulseBlaster programming mode")

    def _set_sampling_rate_attribute(self, samprate: int) -> None:
        self.samprate = samprate

    def _set_min_instr_clk_cycles(self, clk_cycles: int) -> None:
        # Minimum instruction time, in clock periods
        # (5 clk cycles or 10 ns for 500 MHz)
        self.pb_min_instr_clk_cycles = clk_cycles
