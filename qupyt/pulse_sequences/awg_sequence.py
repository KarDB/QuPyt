"""Represent, store, and validate compiled AWG pulse sequences.

This module connects numeric pulse-sequence compilation with real and mock AWG
uploaders. CompiledBlock and CompiledSequence describe independently sized
waveform blocks, their playback order, repeat counts, flags, and timing. Each
physical source retains one analog sample row followed by one packed-marker row.

The archive reader and writer validate this representation and store it as a
versioned NPZ file with numeric arrays and JSON metadata, without pickle. Archive
validation checks the data contract independently of an instrument's limits.

The sequence-planning helpers build logical steps, validate flat and repeated-child
layouts, and check configured repeat and sequencer-memory limits for both real
and mock uploaders. These checks are applied when preparing an upload; they do
not restrict which otherwise valid compiled sequences can be read or written.

Instrument discovery, hardware queries, SCPI commands, and waveform transfers
are handled by the synchronisers in qupyt.hardware.synchronisers.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray

from qupyt.utils.errors import cleanup_on_exit

Step = TypeVar("Step")

DEFAULT_MAX_SEQUENCE_STEPS = 16_383
DEFAULT_MAX_STEP_REPEAT = 1_048_576

AWG_ARCHIVE_FORMAT_VERSION = 2
_METADATA_FIELDS = {
    "format_version",
    "sample_rate_hz",
    "source_channels",
    "block_names",
    "sample_counts",
    "requested_durations_us",
    "step_block_names",
    "repeat_counts",
    "flags_json",
    "properties_json",
}
_INT64_MAX = int(np.iinfo(np.int64).max)


@dataclass(frozen=True)
class CompiledBlock:
    """One named waveform with analog/packed-marker pairs of rows.

    Parameters
    ----------
    name : str
        Logical block name, also used to construct instrument waveform names.
    requested_duration_us : float
        Requested duration before rounding to the nearest sample.
    samples : numpy.ndarray
        Float64 samples of shape ``(2 * number_of_sources, sample_count)``.
    """

    name: str
    requested_duration_us: float
    samples: NDArray[np.float64]

    @property
    def sample_count(self) -> int:
        """Return the actual number of samples in this block."""
        return int(self.samples.shape[1])


@dataclass(frozen=True)
class CompiledSequence:
    """Numeric blocks and logical execution metadata, independent of child tables.

    Parameters
    ----------
    sample_rate_hz : float
        Sampling rate used to compile every block.
    source_channels : tuple of int
        Physical source numbers in the order of the interleaved sample rows.
    blocks : dict of str to CompiledBlock
        Distinct named blocks; their sample counts may differ.
    step_block_names : tuple of str
        Logical execution order, with repeated references retained.
    repeat_counts : tuple of int
        Positive repeat counts parallel to ``step_block_names``.
    flag_values : dict of str to list of str
        Flag selectors for every block, including blocks with no flags.
    properties : dict of str to object
        Additional recursively JSON-compatible metadata.
    """

    sample_rate_hz: float
    source_channels: tuple[int, ...]
    blocks: dict[str, CompiledBlock]
    step_block_names: tuple[str, ...]
    repeat_counts: tuple[int, ...]
    flag_values: dict[str, list[str]]
    properties: dict[str, Any]

    @property
    def body_sample_count(self) -> int:
        """Return logical body length, including each step's repeat count."""
        return sum(
            self.blocks[name].sample_count * repeat
            for name, repeat in zip(self.step_block_names, self.repeat_counts)
        )

    @property
    def body_duration_s(self) -> float:
        """Return one complete expanded body duration, excluding the gate."""
        return self.body_sample_count / self.sample_rate_hz


def parse_awg_flag(selector: str, channels: Sequence[int]) -> tuple[str, int | None]:
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


def _positive_finite(value: object, description: str) -> float:
    """Validate a Python numeric scalar without accepting booleans."""
    if type(value) not in (int, float):
        raise ValueError(f"{description} must be a finite positive number")
    try:
        numeric = float(cast("int | float", value))
    except OverflowError as error:
        raise ValueError(f"{description} must be a finite positive number") from error
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{description} must be a finite positive number")
    return numeric


def _validate_channels(channels: Sequence[int]) -> None:
    """Validate the physical source order without changing its meaning."""
    if any(type(channel) is not int for channel in channels) or tuple(channels) not in (
        (1,),
        (2,),
        (1, 2),
        (2, 1),
    ):
        raise ValueError("source_channels must be [1], [2], [1, 2], or [2, 1]")


def _validate_name(name: object) -> None:
    """Reject names that cannot safely be placed in quoted SCPI commands."""
    if (
        not isinstance(name, str)
        or not name
        or any(
            character in "\"'" or ord(character) < 32 or ord(character) == 127
            for character in name
        )
    ):
        raise ValueError(
            f"Invalid AWG block name {name!r}: use a nonempty name without quotes or control characters"
        )


def _validate_json_value(value: object, description: str) -> None:
    """Require primitive JSON values and finite numbers, without coercion."""
    if value is None or type(value) in (bool, str, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{description} contains a non-finite JSON number")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, description)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{description} must have string JSON object keys")
            _validate_json_value(item, description)
        return
    raise ValueError(
        f"{description} contains a value unsupported by JSON: {type(value).__name__}"
    )


def _validate_sequence(sequence: CompiledSequence) -> None:
    """Check complete numeric and metadata invariants before publication/use."""
    rate = _positive_finite(sequence.sample_rate_hz, "sample_rate_hz")
    _validate_channels(sequence.source_channels)
    if not isinstance(sequence.blocks, dict) or not sequence.blocks:
        raise ValueError("blocks must be a nonempty mapping")
    for name, block in sequence.blocks.items():
        _validate_name(name)
        if not isinstance(block, CompiledBlock) or block.name != name:
            raise ValueError(f"Block key {name!r} must match its CompiledBlock name")
        duration = _positive_finite(
            block.requested_duration_us, f"Requested duration for block {name!r}"
        )
        samples = block.samples
        if (
            not isinstance(samples, np.ndarray)
            or samples.dtype != np.dtype(np.float64)
            or samples.ndim != 2
            or samples.shape[0] != 2 * len(sequence.source_channels)
            or samples.shape[1] < 1
        ):
            raise ValueError(
                f"Block {name!r} must be float64 with shape (2 * sources, positive sample count)"
            )
        requested_samples = rate * duration * 1e-6
        if (
            not math.isfinite(requested_samples)
            or round(requested_samples) != block.sample_count
        ):
            raise ValueError(
                f"Block {name!r} sample count disagrees with its requested duration and sample rate"
            )
        if not np.isfinite(samples).all():
            raise ValueError(f"Block {name!r} contains non-finite samples")
        # Existing sequences can deliberately exceed the normalized analog
        # range. Preserve their finite values; do not clip or normalize them
        # while changing the archive representation.
        markers = samples[1::2]
        if (
            np.any(markers < 0)
            or np.any(markers > 255)
            or np.any(markers != np.floor(markers))
        ):
            raise ValueError(
                f"Block {name!r} packed markers must be integral bytes in [0, 255]"
            )
    if (
        not isinstance(sequence.step_block_names, tuple)
        or not sequence.step_block_names
        or not isinstance(sequence.repeat_counts, tuple)
        or len(sequence.step_block_names) != len(sequence.repeat_counts)
    ):
        raise ValueError(
            "step_block_names and repeat_counts must be nonempty tuples of equal length"
        )
    for name, repeat in zip(sequence.step_block_names, sequence.repeat_counts):
        if not isinstance(name, str) or name not in sequence.blocks:
            raise ValueError(f"Logical step references unknown block {name!r}")
        if type(repeat) is not int or not 1 <= repeat <= _INT64_MAX:
            raise ValueError(
                f"Repeat count for block {name!r} must be a positive int64 integer"
            )
    if not isinstance(sequence.flag_values, dict) or set(sequence.flag_values) != set(
        sequence.blocks
    ):
        raise ValueError(
            "flags_json must contain exactly one entry for every block, including empty flag lists"
        )
    for name, selectors in sequence.flag_values.items():
        if not isinstance(selectors, list):
            raise ValueError(f"Flag selectors for block {name!r} must be a list")
        for selector in selectors:
            parse_awg_flag(selector, sequence.source_channels)
    if not isinstance(sequence.properties, dict):
        raise ValueError("properties_json must contain a JSON object")
    _validate_json_value(sequence.properties, "properties_json")


def write_awg_archive(path: Path, sequence: CompiledSequence) -> None:
    """Validate and atomically publish a version-2 AWG archive.

    Parameters
    ----------
    path : pathlib.Path
        Destination archive. The parent directory must already exist.
    sequence : CompiledSequence
        Complete validated waveform and logical execution data.

    Raises
    ------
    ValueError
        Numeric samples or metadata violate the archive contract.
    OSError
        Writing or publishing fails; an existing destination is retained.
    """
    _validate_sequence(sequence)
    names = sorted(sequence.blocks)
    fields: dict[str, Any] = {
        "format_version": np.asarray(AWG_ARCHIVE_FORMAT_VERSION, dtype=np.int64),
        "sample_rate_hz": np.asarray(sequence.sample_rate_hz, dtype=np.float64),
        "source_channels": np.asarray(sequence.source_channels, dtype=np.int64),
        "block_names": np.asarray(names, dtype=np.str_),
        "sample_counts": np.asarray(
            [sequence.blocks[name].sample_count for name in names], dtype=np.int64
        ),
        "requested_durations_us": np.asarray(
            [sequence.blocks[name].requested_duration_us for name in names],
            dtype=np.float64,
        ),
        "step_block_names": np.asarray(sequence.step_block_names, dtype=np.str_),
        "repeat_counts": np.asarray(sequence.repeat_counts, dtype=np.int64),
        "flags_json": np.asarray(
            json.dumps(sequence.flag_values, sort_keys=True, allow_nan=False),
            dtype=np.str_,
        ),
        "properties_json": np.asarray(
            json.dumps(sequence.properties, sort_keys=True, allow_nan=False),
            dtype=np.str_,
        ),
    }
    for index, name in enumerate(names):
        fields[f"block_{index:06d}"] = sequence.blocks[name].samples
    temporary = tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary.name)
    with cleanup_on_exit(
        lambda: temporary_path.unlink(missing_ok=True), "temporary AWG archive removal"
    ):
        with cleanup_on_exit(temporary.close, "temporary AWG archive close"):
            # Explicit dtypes above prevent object serialization on all supported
            # NumPy versions, including versions without a writer allow_pickle option.
            np.savez(temporary, **fields)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    logging.debug("Published AWG archive %s with %s distinct blocks", path, len(names))


def _require_array(
    fields: dict[str, NDArray[Any]], name: str, kind: str, ndim: int
) -> NDArray[Any]:
    """Check a primitive array's dimensionality and dtype category."""
    value = fields[name]
    valid_dtype = value.dtype.kind in kind
    if kind == "f":
        valid_dtype = value.dtype == np.dtype(np.float64)
    if value.ndim != ndim or not valid_dtype:
        raise ValueError(f"AWG archive field {name!r} has invalid shape or dtype")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of silently discarding metadata."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _read_json(fields: dict[str, NDArray[Any]], name: str) -> Any:
    """Decode a scalar Unicode JSON field without nonstandard numbers."""
    value = _require_array(fields, name, "U", 0)
    decoded: Any = json.loads(str(value.item()), object_pairs_hook=_unique_json_object)
    _validate_json_value(decoded, name)
    return decoded


def _decode_archive(fields: dict[str, NDArray[Any]]) -> CompiledSequence:
    """Decode an already materialized archive and validate its full schema."""
    version = _require_array(fields, "format_version", "iu", 0)
    if int(version.item()) != AWG_ARCHIVE_FORMAT_VERSION:
        raise ValueError(f"Unsupported AWG archive format version {version.item()!r}")
    rate = float(_require_array(fields, "sample_rate_hz", "f", 0).item())
    channels = tuple(
        int(value) for value in _require_array(fields, "source_channels", "iu", 1)
    )
    names = tuple(str(value) for value in _require_array(fields, "block_names", "U", 1))
    if not names or len(set(names)) != len(names):
        raise ValueError("block_names must be nonempty and unique")
    if names != tuple(sorted(names)):
        raise ValueError("block_names must be sorted")
    counts = _require_array(fields, "sample_counts", "iu", 1)
    durations = _require_array(fields, "requested_durations_us", "f", 1)
    if len(counts) != len(names) or len(durations) != len(names):
        raise ValueError(
            "block_names, sample_counts, and requested_durations_us must have equal lengths"
        )
    expected_fields = _METADATA_FIELDS | {
        f"block_{index:06d}" for index in range(len(names))
    }
    if set(fields) != expected_fields:
        raise ValueError(
            f"AWG archive members differ from the schema: missing={sorted(expected_fields - set(fields))}, extra={sorted(set(fields) - expected_fields)}"
        )
    blocks: dict[str, CompiledBlock] = {}
    for index, name in enumerate(names):
        samples = _require_array(fields, f"block_{index:06d}", "f", 2)
        if int(counts[index]) < 1 or samples.shape[1] != int(counts[index]):
            raise ValueError(
                f"Block {name!r} sample_counts metadata does not match its samples"
            )
        blocks[name] = CompiledBlock(name, float(durations[index]), samples)
    steps = tuple(
        str(value) for value in _require_array(fields, "step_block_names", "U", 1)
    )
    repeats = tuple(
        int(value) for value in _require_array(fields, "repeat_counts", "iu", 1)
    )
    sequence = CompiledSequence(
        rate,
        channels,
        blocks,
        steps,
        repeats,
        _read_json(fields, "flags_json"),
        _read_json(fields, "properties_json"),
    )
    _validate_sequence(sequence)
    return sequence


def read_awg_archive(
    path: Path,
    *,
    expected_source_channels: Sequence[int] | None = None,
    expected_sample_rate_hz: float | None = None,
) -> CompiledSequence:
    """Load a version-2 archive without pickle and validate it completely.

    Parameters
    ----------
    path : pathlib.Path
        Numeric AWG archive to read.
    expected_source_channels : sequence of int, optional
        Required physical source order, if configured by the caller.
    expected_sample_rate_hz : float, optional
        Required compilation sampling rate, if configured by the caller.

    Returns
    -------
    CompiledSequence
        Independent read-only arrays and decoded primitive metadata.

    Raises
    ------
    ValueError
        The file is legacy, unsupported, malformed, or incompatible with the
        configured source order/sampling rate. Regenerate legacy files from YAML.
    """
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, NpzFile):
        raise ValueError("An AWG archive must be an NPZ file")
    with cleanup_on_exit(archive.close, "numeric AWG archive close"):
        if "format_version" not in archive.files:
            raise ValueError(
                "Legacy AWG archive has no format_version; regenerate it from YAML"
            )
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("AWG archive contains duplicate member names")
        missing = _METADATA_FIELDS - set(archive.files)
        if missing:
            raise ValueError(
                f"AWG archive is missing required fields: {sorted(missing)}"
            )
        fields = {name: archive[name] for name in archive.files}
        sequence = _decode_archive(fields)
        if expected_source_channels is not None:
            _validate_channels(expected_source_channels)
            if sequence.source_channels != tuple(expected_source_channels):
                raise ValueError(
                    "AWG archive source_channels disagree with configured physical source order"
                )
        if expected_sample_rate_hz is not None:
            expected_rate = _positive_finite(
                expected_sample_rate_hz, "Expected sample rate"
            )
            if sequence.sample_rate_hz != expected_rate:
                raise ValueError(
                    "AWG archive sample_rate_hz disagrees with configured sampling rate"
                )
    for block in sequence.blocks.values():
        block.samples.setflags(write=False)
    logging.debug(
        "Loaded AWG archive %s with %s distinct blocks", path, len(sequence.blocks)
    )
    return sequence


def validate_awg_limit(value: Any, key: str, device_type: str) -> int:
    """Validate a positive AWG hardware limit provided by the configuration."""
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{key} must be an integer AWG limit for {device_type}, got {value!r}."
        ) from exc
    if limit <= 0:
        raise ValueError(
            f"{key} must be a positive AWG limit for {device_type}, got {limit}."
        )
    logging.info("%s configured %s=%s", device_type, key, limit)
    return limit


def validate_awg_repeat_count(
    repeat: int, max_step_repeat: int, context: str, device_type: str
) -> None:
    """Reject repeat counts that cannot be represented by the configured AWG."""
    if repeat <= 0:
        raise ValueError(f"{device_type} {context} repeat count must be positive.")
    if repeat > max_step_repeat:
        raise ValueError(
            f"{device_type} {context} repeat count {repeat} exceeds the "
            f"configured AWG limit of {max_step_repeat}."
        )


def validate_awg_metadata_lengths(
    waveform_names: Sequence[str], repeat_counts: Sequence[int], device_type: str
) -> None:
    """Require a repeat count for every logical waveform step."""
    if len(waveform_names) != len(repeat_counts):
        raise ValueError(
            f"{device_type} sequence metadata mismatch: "
            f"{len(waveform_names)} waveform names but "
            f"{len(repeat_counts)} repeat counts."
        )


def build_logical_awg_steps(
    waveform_names: Sequence[str],
    repeat_counts: Sequence[int],
    step_factory: Callable[[str, int, tuple[str, ...]], Step],
    *,
    max_step_repeat: int,
    device_type: str,
    flag_values: Mapping[str, Sequence[str]] | None = None,
) -> list[Step]:
    """Build logical steps using the same repeat and flag rules for both devices."""
    validate_awg_metadata_lengths(waveform_names, repeat_counts, device_type)
    flags = flag_values or {}
    steps: list[Step] = []
    for wavename, count in zip(waveform_names, repeat_counts, strict=True):
        repeat = int(count)
        validate_awg_repeat_count(
            repeat, max_step_repeat, f"step {len(steps) + 1}", device_type
        )
        steps.append(
            step_factory(str(wavename), repeat, tuple(sorted(flags.get(wavename, ()))))
        )
    return steps


def get_awg_sequence_options(instructions: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the upload options from an expanded logical pulse sequence."""
    return {
        key: instructions[key]
        for key in ("awg_sequence_mode", "awg_repeated_child")
        if key in instructions
    }


def validate_awg_sequence_plan(
    steps: list[Step],
    mode: str,
    config: object,
    *,
    max_sequence_steps: int,
    max_step_repeat: int,
    device_type: str,
    nongatereps: int = 1,
) -> None:
    """Validate every stored sequence asset before programming the instrument.

    The shared AWG70000 step memory includes the trigger gate and all child
    assets. Child repetitions consume playback time, but do not allocate more
    sequence steps. Tracks of one asset share the same stored steps.
    """
    if not steps:
        raise ValueError(f"{device_type} requires at least one logical sequence step.")
    validate_awg_repeat_count(
        nongatereps, max_step_repeat, "measurement-body", device_type
    )
    if mode == "flat":
        if len(steps) > max_sequence_steps:
            raise ValueError(
                f"Expanded AWG sequence has {len(steps)} steps, exceeding "
                f"the configured AWG limit of {max_sequence_steps}. Enable "
                "awg_sequence_mode: repeated_child or reduce the sequence."
            )
        stored_steps = len(steps) + 2
    elif mode == "repeated_child":
        if nongatereps != 1:
            raise ValueError(
                "repeated_child mode supports nongatereps=1 only; outer body "
                "repetition is not implemented for this upload layout."
            )
        prefix, child, _, suffix = validate_repeated_child_region(
            steps,
            config,
            max_sequence_steps=max_sequence_steps,
            max_step_repeat=max_step_repeat,
            device_type=device_type,
        )
        stored_steps = 1 + len(prefix) + 1 + len(suffix) + len(child)
    else:
        raise ValueError(
            f"Unsupported awg_sequence_mode: {mode!r}. Supported values are "
            "'flat' and 'repeated_child'."
        )
    if stored_steps > max_sequence_steps:
        raise ValueError(
            f"{device_type} aggregate stored sequence steps {stored_steps}, including "
            f"the trigger gate and child assets, exceed the configured AWG limit "
            f"of {max_sequence_steps}."
        )


def infer_child_repetitions(
    steps: list[Step], prefix_steps: int, child: list[Step]
) -> int:
    """Count consecutive identical child blocks following the prefix."""
    child_steps = len(child)
    if child_steps == 0:
        raise ValueError("Cannot infer repetitions of an empty child sequence.")
    repetitions = 0
    start = prefix_steps
    while steps[start : start + child_steps] == child:
        repetitions += 1
        start += child_steps
    if repetitions < 2:
        raise ValueError(
            "Could not infer awg_repeated_child repetitions: at least two "
            "identical child occurrences are required."
        )
    return repetitions


def validate_repeated_child_region(
    steps: list[Step],
    config: object,
    *,
    max_sequence_steps: int,
    max_step_repeat: int,
    device_type: str,
) -> tuple[list[Step], list[Step], int, list[Step]]:
    """Validate and split a repeated-child plan, preserving complete step equality."""
    if not isinstance(config, dict):
        raise ValueError(
            "awg_repeated_child must be a mapping with prefix_steps, "
            "child_steps, and optionally repetitions."
        )
    try:
        prefix_steps = int(config.get("prefix_steps", 0))
        child_steps = int(config["child_steps"])
    except KeyError as exc:
        raise ValueError(
            "awg_repeated_child requires child_steps for repeated_child mode."
        ) from exc
    repetitions_config = config.get("repetitions")
    if prefix_steps < 0:
        raise ValueError(
            f"awg_repeated_child prefix_steps must be >= 0, got {prefix_steps}."
        )
    if child_steps <= 0:
        raise ValueError(
            f"awg_repeated_child child_steps must be > 0, got {child_steps}."
        )
    if child_steps > max_sequence_steps:
        raise ValueError(
            f"Repeated child sequence has {child_steps} steps, exceeding "
            f"the configured AWG limit of {max_sequence_steps}."
        )
    if prefix_steps + child_steps > len(steps):
        raise ValueError(
            "awg_repeated_child prefix_steps + child_steps exceeds the "
            f"logical sequence length: {prefix_steps} + {child_steps} > {len(steps)}."
        )
    child = steps[prefix_steps : prefix_steps + child_steps]
    if repetitions_config is None:
        repetitions = infer_child_repetitions(steps, prefix_steps, child)
    else:
        repetitions = int(repetitions_config)
        if repetitions <= 0:
            raise ValueError(
                f"awg_repeated_child repetitions must be > 0, got {repetitions}."
            )
        expected_end = prefix_steps + repetitions * child_steps
        if expected_end > len(steps):
            available_steps = len(steps) - prefix_steps
            max_complete_repetitions = available_steps // child_steps
            raise ValueError(
                "awg_repeated_child does not match the expanded logical "
                "sequence. The configured repeated region would end after "
                f"{prefix_steps} + {repetitions} * {child_steps} = "
                f"{expected_end} steps, but the generated sequence has only "
                f"{len(steps)} steps. After the {prefix_steps} prefix steps, "
                f"only {available_steps} steps remain, which can contain at "
                f"most {max_complete_repetitions} complete child repetitions "
                f"of {child_steps} steps. Check that n_meas/repetitions and "
                "len(cycle_order)/child_steps are calculated from the same "
                "sequence that is written to sequencing_order."
            )
        for repetition_index in range(repetitions):
            start = prefix_steps + repetition_index * child_steps
            current_child = steps[start : start + child_steps]
            if current_child != child:
                raise ValueError(
                    "Repeated child block mismatch at repetition "
                    f"{repetition_index + 1}. Waveform names, repeat counts, "
                    "and flags must match exactly for repeated_child mode."
                )
    validate_awg_repeat_count(
        repetitions, max_step_repeat, "child-sequence parent", device_type
    )
    suffix_start = prefix_steps + repetitions * child_steps
    suffix = steps[suffix_start:]
    compact_body_steps = prefix_steps + 1 + len(suffix)
    if compact_body_steps > max_sequence_steps:
        raise ValueError(
            f"Compact AWG parent/body sequence has {compact_body_steps} steps, "
            f"exceeding the configured AWG limit of {max_sequence_steps}."
        )
    return steps[:prefix_steps], child, repetitions, suffix
