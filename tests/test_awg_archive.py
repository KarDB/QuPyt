"""Exercise the safe numeric AWG archive contract and failed publication paths."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.typing import NDArray

from qupyt.pulse_sequences.awg_sequence import (
    CompiledBlock,
    CompiledSequence,
    parse_awg_flag,
    read_awg_archive,
    write_awg_archive,
)


@pytest.fixture
def compiled_sequence() -> CompiledSequence:
    short = np.asarray(
        [[0.0, 0.25, -0.5], [16, 32, 48], [1.0, -1.0, 0], [64, 128, 255]],
        dtype=np.float64,
    )
    long = np.zeros((4, 5), dtype=np.float64)
    long[1] = [0, 1, 16, 32, 255]
    return CompiledSequence(
        sample_rate_hz=1e6,
        source_channels=(2, 1),
        blocks={
            "prep": CompiledBlock("prep", 3.0, short),
            "drive": CompiledBlock("drive", 5.0, long),
        },
        step_block_names=("prep", "drive", "prep"),
        repeat_counts=(2, 3, 1),
        flag_values={"prep": ["A", "A2", "A", "D1"], "drive": []},
        properties={
            "description": "µs experiment",
            "nested": [None, True, {"n": 3, "f": 0.5}],
        },
    )


def _archive_fields(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def test_ragged_archive_round_trips_without_objects(
    tmp_path: Path, compiled_sequence: CompiledSequence
) -> None:
    path = tmp_path / "sequence.npz"
    write_awg_archive(path, compiled_sequence)
    fields = _archive_fields(path)
    assert all(value.dtype.kind != "O" for value in fields.values())
    assert "arr_0" not in fields
    assert fields["format_version"].item() == 2
    assert fields["block_names"].tolist() == ["drive", "prep"]
    assert fields["sample_counts"].tolist() == [5, 3]
    assert fields["block_000000"].shape == (4, 5)
    assert fields["block_000001"].shape == (4, 3)
    assert fields["flags_json"].shape == ()
    assert json.loads(fields["flags_json"].item()) == compiled_sequence.flag_values
    loaded = read_awg_archive(
        path, expected_source_channels=[2, 1], expected_sample_rate_hz=1e6
    )
    assert loaded.source_channels == (2, 1)
    assert loaded.step_block_names == compiled_sequence.step_block_names
    assert loaded.repeat_counts == compiled_sequence.repeat_counts
    assert loaded.flag_values == compiled_sequence.flag_values
    assert loaded.properties == compiled_sequence.properties
    assert loaded.body_sample_count == 24
    assert loaded.body_duration_s == pytest.approx(24e-6)
    for name, block in compiled_sequence.blocks.items():
        assert loaded.blocks[name].requested_duration_us == block.requested_duration_us
        np.testing.assert_array_equal(loaded.blocks[name].samples, block.samples)
        assert not loaded.blocks[name].samples.flags.writeable
        assert block.samples.flags.writeable


@pytest.mark.parametrize("channels", [(1,), (2,), (1, 2), (2, 1)])
def test_all_existing_source_orders_are_preserved(
    tmp_path: Path, channels: tuple[int, ...]
) -> None:
    samples = np.zeros((2 * len(channels), 3), dtype=np.float64)
    sequence = CompiledSequence(
        1e6,
        channels,
        {"b": CompiledBlock("b", 3, samples)},
        ("b",),
        (1,),
        {"b": []},
        {},
    )
    path = tmp_path / "sequence.npz"
    write_awg_archive(path, sequence)
    assert read_awg_archive(path).source_channels == channels


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format_version", np.asarray(3), "Unsupported"),
        ("format_version", np.asarray(True), "shape or dtype"),
        ("format_version", np.asarray([2]), "shape or dtype"),
        ("sample_rate_hz", np.asarray(0.0), "finite positive"),
        ("sample_rate_hz", np.asarray(float("nan")), "finite positive"),
        ("sample_rate_hz", np.asarray(1_000_000), "shape or dtype"),
        ("source_channels", np.asarray([1, 1]), "source_channels"),
        ("source_channels", np.asarray([True]), "shape or dtype"),
        ("source_channels", np.asarray([3]), "source_channels"),
        ("source_channels", np.asarray([1.0, 2.0]), "shape or dtype"),
        ("block_names", np.asarray(["drive", "drive"]), "unique"),
        ("block_names", np.asarray(["prep", "drive"]), "sorted"),
        ("block_names", np.asarray([], dtype=np.str_), "nonempty"),
        ("sample_counts", np.asarray([5, 2]), "does not match"),
        ("sample_counts", np.asarray([5]), "equal lengths"),
        ("sample_counts", np.asarray([5.0, 3.0]), "shape or dtype"),
        ("requested_durations_us", np.asarray([5.0, 4.0]), "disagrees"),
        ("requested_durations_us", np.asarray([5.0, -3.0]), "finite positive"),
        ("requested_durations_us", np.asarray([5.0]), "equal lengths"),
        ("step_block_names", np.asarray(["missing", "drive", "prep"]), "unknown block"),
        ("step_block_names", np.asarray([], dtype=np.str_), "nonempty"),
        ("repeat_counts", np.asarray([2, 0, 1]), "positive int64"),
        ("repeat_counts", np.asarray([2, -1, 1]), "positive int64"),
        ("repeat_counts", np.asarray([2.0, 3.0, 1.0]), "shape or dtype"),
        ("repeat_counts", np.asarray([True, True, True]), "shape or dtype"),
        ("repeat_counts", np.asarray([2, 3]), "equal length"),
        ("flags_json", np.asarray('{"prep": []}'), "exactly one entry"),
        (
            "flags_json",
            np.asarray('{"prep": [], "drive": [], "extra": []}'),
            "exactly one entry",
        ),
        ("flags_json", np.asarray('{"prep": null, "drive": []}'), "must be a list"),
        ("flags_json", np.asarray('{"prep": [1], "drive": []}'), "Invalid AWG flag"),
        ("flags_json", np.asarray('{"prep": ["A3"], "drive": []}'), "Invalid AWG flag"),
        (
            "flags_json",
            np.asarray('{"prep": [], "prep": [], "drive": []}'),
            "Duplicate JSON",
        ),
        ("flags_json", np.asarray(['{"prep": [], "drive": []}']), "shape or dtype"),
        ("flags_json", np.asarray(b'{"prep": [], "drive": []}'), "shape or dtype"),
        ("properties_json", np.asarray("[]"), "JSON object"),
        ("properties_json", np.asarray('{"x": NaN}'), "non-finite"),
        ("properties_json", np.asarray('{"x": Infinity}'), "non-finite"),
        ("properties_json", np.asarray('{"x": 1, "x": 2}'), "Duplicate JSON"),
        ("properties_json", np.asarray("invalid JSON"), "Expecting"),
        ("block_000000", np.zeros((3, 5), dtype=np.float64), "shape"),
        ("block_000000", np.zeros((4, 5), dtype=np.float32), "shape or dtype"),
        ("block_000000", np.zeros((4, 4), dtype=np.float64), "does not match"),
    ],
)
def test_reader_rejects_malformed_fields(
    tmp_path: Path,
    compiled_sequence: CompiledSequence,
    field: str,
    value: NDArray[Any],
    message: str,
) -> None:
    path = tmp_path / "sequence.npz"
    write_awg_archive(path, compiled_sequence)
    fields = _archive_fields(path)
    fields[field] = value
    np.savez(path, **fields)
    with pytest.raises(ValueError, match=message):
        read_awg_archive(path)


@pytest.mark.parametrize("name", ["block_000000", "properties_json", "repeat_counts"])
def test_missing_fields_are_rejected(
    tmp_path: Path, compiled_sequence: CompiledSequence, name: str
) -> None:
    path = tmp_path / "sequence.npz"
    write_awg_archive(path, compiled_sequence)
    fields = _archive_fields(path)
    del fields[name]
    np.savez(path, **fields)
    with pytest.raises(ValueError, match="missing"):
        read_awg_archive(path)


def test_extra_fields_and_object_arrays_are_rejected(
    tmp_path: Path, compiled_sequence: CompiledSequence
) -> None:
    path = tmp_path / "sequence.npz"
    write_awg_archive(path, compiled_sequence)
    fields = _archive_fields(path)
    np.savez(path, **fields, unexpected=np.asarray(1))
    with pytest.raises(ValueError, match="extra"):
        read_awg_archive(path)
    fields["properties_json"] = np.asarray({"unexpected": True}, dtype=object)
    np.savez(path, **fields)
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        read_awg_archive(path)


def test_legacy_archives_require_regeneration(tmp_path: Path) -> None:
    path = tmp_path / "sequence.npz"
    np.savez(path, np.zeros((1, 2, 3)), np.asarray({"old": "object"}, dtype=object))
    with pytest.raises(ValueError, match="Legacy.*regenerate.*YAML"):
        read_awg_archive(path)


@pytest.mark.parametrize(
    "row,value", [(1, -1), (1, 256), (3, 1.5), (0, float("nan")), (1, float("inf"))]
)
def test_writer_rejects_invalid_samples_before_creating_a_file(
    tmp_path: Path, compiled_sequence: CompiledSequence, row: int, value: float
) -> None:
    path = tmp_path / "sequence.npz"
    compiled_sequence.blocks["prep"].samples[row, 0] = value
    with pytest.raises(ValueError):
        write_awg_archive(path, compiled_sequence)
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_analog_samples_beyond_normalized_range_round_trip_without_changes(
    tmp_path: Path, compiled_sequence: CompiledSequence
) -> None:
    values = np.asarray(
        [-1.3, 1.3, np.nextafter(-1.0, -np.inf), np.nextafter(1.0, np.inf), -0.0]
    )
    samples = compiled_sequence.blocks["drive"].samples
    samples[0] = values
    samples[2] = values[::-1]
    original = samples.tobytes()
    path = tmp_path / "sequence.npz"
    write_awg_archive(path, compiled_sequence)
    loaded = read_awg_archive(path)
    assert loaded.blocks["drive"].samples.tobytes() == original


@pytest.mark.parametrize("repeat", [True, 1.0, 0, -1, 2**63])
def test_writer_rejects_non_integer_or_invalid_repeats(
    tmp_path: Path, compiled_sequence: CompiledSequence, repeat: object
) -> None:
    invalid = replace(compiled_sequence, repeat_counts=(cast(int, repeat), 3, 1))
    with pytest.raises(ValueError, match="positive int64"):
        write_awg_archive(tmp_path / "sequence.npz", invalid)


@pytest.mark.parametrize(
    "name", ["", 'bad"name', "bad'name", "bad\nname", "bad\rname", "bad\x00name"]
)
def test_writer_rejects_unsafe_scpi_names(
    tmp_path: Path, compiled_sequence: CompiledSequence, name: str
) -> None:
    block = replace(compiled_sequence.blocks["prep"], name=name)
    invalid = replace(
        compiled_sequence,
        blocks={name: block},
        flag_values={name: []},
        step_block_names=(name,),
        repeat_counts=(1,),
    )
    with pytest.raises(ValueError, match="Invalid AWG block name"):
        write_awg_archive(tmp_path / "sequence.npz", invalid)


@pytest.mark.parametrize(
    "properties",
    [
        {"n": np.int64(1)},
        {"array": np.zeros(2)},
        {"tuple": (1, 2)},
        {1: "value"},
        {"nan": float("nan")},
    ],
)
def test_writer_rejects_non_json_properties(
    tmp_path: Path, compiled_sequence: CompiledSequence, properties: dict[Any, Any]
) -> None:
    with pytest.raises(ValueError, match="JSON"):
        write_awg_archive(
            tmp_path / "sequence.npz", replace(compiled_sequence, properties=properties)
        )


def test_expected_configuration_requires_exact_source_order_and_rate(
    tmp_path: Path, compiled_sequence: CompiledSequence
) -> None:
    path = tmp_path / "sequence.npz"
    write_awg_archive(path, compiled_sequence)
    with pytest.raises(ValueError, match="physical source order"):
        read_awg_archive(path, expected_source_channels=[1, 2])
    with pytest.raises(ValueError, match="sampling rate"):
        read_awg_archive(path, expected_sample_rate_hz=2e6)


@pytest.mark.parametrize("failing_operation", ["savez", "replace"])
def test_publication_failure_retains_previous_archive_and_removes_temporary_file(
    tmp_path: Path,
    compiled_sequence: CompiledSequence,
    monkeypatch: pytest.MonkeyPatch,
    failing_operation: str,
) -> None:
    path = tmp_path / "sequence.npz"
    write_awg_archive(path, compiled_sequence)
    original = path.read_bytes()
    target = np if failing_operation == "savez" else os
    monkeypatch.setattr(
        target, failing_operation, Mock(side_effect=OSError("publication failed"))
    )
    with pytest.raises(OSError, match="publication failed"):
        write_awg_archive(path, compiled_sequence)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_primary_validation_failure_survives_archive_close_failure(
    tmp_path: Path,
    compiled_sequence: CompiledSequence,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "sequence.npz"
    write_awg_archive(path, compiled_sequence)
    fields = _archive_fields(path)
    fields["repeat_counts"] = np.asarray([1, 0, 1])
    np.savez(path, **fields)
    archive = np.load(path, allow_pickle=False)
    original_close = archive.close

    def fail_after_close() -> None:
        original_close()
        raise OSError("close failed")

    monkeypatch.setattr(archive, "close", fail_after_close)
    monkeypatch.setattr(np, "load", Mock(return_value=archive))
    with pytest.raises(ValueError, match="positive int64"):
        read_awg_archive(path)
    assert "original failure retained" in caplog.text
    assert "close failed" in caplog.text


@pytest.mark.parametrize(
    "selector,channels,expected",
    [("A", [2], ("A", None)), ("B2", [2, 1], ("B", 2)), ("D1", [1], ("D", 1))],
)
def test_flag_suffix_remains_a_physical_source(
    selector: str, channels: list[int], expected: tuple[str, int | None]
) -> None:
    assert parse_awg_flag(selector, channels) == expected


def test_inactive_physical_flag_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="inactive AWG channel 2"):
        parse_awg_flag("A2", [1])
