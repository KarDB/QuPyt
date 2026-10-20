"""Shared validation of real and mock AWG sequence plans."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

Step = TypeVar("Step")

DEFAULT_MAX_SEQUENCE_STEPS = 16_383
DEFAULT_MAX_STEP_REPEAT = 1_048_576


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
