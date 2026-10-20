"""Exercise variable-length YAML compilation through the original numeric sampler."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from qupyt.pulse_sequences import SequenceDesigner as designer
from qupyt.pulse_sequences.awg_archive import read_awg_archive
from qupyt.pulse_sequences.yaml_sequence import YamlSequence
from qupyt.set_up import get_seq_dir, make_userdirs


@pytest.fixture
def sequence_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    make_userdirs()
    return get_seq_dir()


@pytest.mark.parametrize("sources", [[1], [1, 2], [2, 1]])
def test_variable_blocks_preserve_the_uniform_samplers_exact_samples(
    sequence_directory: Path, sources: list[int]
) -> None:
    sequence = YamlSequence(
        4.4, block_durations={"short": 2.4, "long": 7.6, "idle": 1.1}
    )
    mapping: dict[str, Any] = {"analog": 0, "marker": 1, "flag": "A1"}
    for name in ("default", "short", "long"):
        sequence.add_pulse(
            "analog",
            0.4,
            1.4,
            amplitude=0.375,
            frequency=125000,
            phase=0.3,
            sequence_blocks=[name],
        )
        sequence.add_pulse("marker", 0, 1, sequence_blocks=[name])
    sequence.add_pulse("flag", 0, 1, sequence_blocks=["long"])
    sequence.sequencing_order = ["idle", "default", "short", "long", "short"]
    sequence.sequencing_repeats = [1, 2, 1, 3, 1]
    sequence.write()
    compiler = designer.PulseSequenceYaml(mapping, sources, 1e6)
    compiler.translate_yaml_to_numeric_instructions()
    compiled = read_awg_archive(sequence_directory / "sequence.npz")
    assert compiled.source_channels == tuple(sources)
    assert compiled.step_block_names == tuple(sequence.sequencing_order)
    assert {name: block.sample_count for name, block in compiled.blocks.items()} == {
        "default": 4,
        "short": 2,
        "long": 8,
        "idle": 1,
    }
    for name, duration in (("default", 4.4), ("short", 2.4), ("long", 7.6)):
        original_sampler = designer.PulseSequence(1, duration, sources, samprate=1e6)
        original_sampler.add_pulse(0, 0.4, 1.4, amplitude=0.375, freq=(125000, 0.3))
        original_sampler.add_pulse(0, 0, 1, channel=1, freq=(0.0, 0.0))
        np.testing.assert_array_equal(
            compiled.blocks[name].samples, original_sampler.pack_waveforms()[0]
        )
    assert not np.any(compiled.blocks["idle"].samples)
    assert compiled.flag_values == {
        "default": [],
        "short": [],
        "long": ["A1"],
        "idle": [],
    }
    assert compiled.body_sample_count == 37
    assert compiled.body_duration_s == pytest.approx(37e-6)


def test_all_four_established_marker_slots_remain_unchanged(
    sequence_directory: Path,
) -> None:
    sequence = YamlSequence(4)
    for index in range(4):
        sequence.add_pulse(f"marker{index}", index, 1)
    sequence.sequencing_order = ["block_0"]
    sequence.sequencing_repeats = [1]
    sequence.write()
    designer.PulseSequenceYaml(
        {f"marker{i}": i + 1 for i in range(4)}, [1], 1e6
    ).translate_yaml_to_numeric_instructions()
    compiled = read_awg_archive(sequence_directory / "sequence.npz")
    np.testing.assert_array_equal(
        compiled.blocks["block_0"].samples[1], [128, 64, 32, 16]
    )


def test_legacy_numeric_string_duration_remains_supported(
    sequence_directory: Path,
) -> None:
    source = sequence_directory / "sequence_0.yaml"
    source.write_text(
        "total_duration: '4.0'\nblock_0: {}\nsequencing_order: [block_0]\n"
        "sequencing_repeats: [1]\n",
        encoding="utf-8",
    )
    designer.PulseSequenceYaml({}, [1], 1e6).translate_yaml_to_numeric_instructions()
    assert (
        read_awg_archive(sequence_directory / "sequence.npz")
        .blocks["block_0"]
        .sample_count
        == 4
    )


@pytest.mark.parametrize(
    "duration", [0, -1, float("nan"), float("inf"), True, "2", 10**400]
)
def test_invalid_override_duration_is_rejected(duration: Any) -> None:
    sequence = YamlSequence(4)
    with pytest.raises(ValueError, match="finite and positive"):
        sequence.set_block_duration("short", duration)
    assert sequence.pulse_sequence == {"total_duration": 4}


@pytest.mark.parametrize(
    "name", ["", "total_duration", "block_durations", "sequencing_order"]
)
def test_override_cannot_replace_sequence_metadata(name: str) -> None:
    sequence = YamlSequence(4)
    with pytest.raises(ValueError, match="reserved"):
        sequence.set_block_duration(name, 1)


def test_unused_override_is_not_silently_ignored(sequence_directory: Path) -> None:
    sequence = YamlSequence(4, block_durations={"typo": 2})
    sequence.add_pulse("analog", 0, 1)
    sequence.sequencing_order = ["block_0"]
    sequence.sequencing_repeats = [1]
    with pytest.raises(ValueError, match="unsequenced block 'typo'"):
        sequence.write()
    assert not (sequence_directory / "sequence_0.yaml").exists()


def test_short_block_failure_keeps_previous_archive_and_cache(
    sequence_directory: Path,
) -> None:
    sequence = YamlSequence(4, block_durations={"long": 6, "short": 2})
    sequence.add_pulse("analog", 0, 1, sequence_blocks=["long", "short"])
    sequence.sequencing_order = ["long", "short"]
    sequence.sequencing_repeats = [1, 1]
    sequence.write()
    compiler = designer.PulseSequenceYaml({"analog": 0}, [1], 1e6)
    compiler.translate_yaml_to_numeric_instructions()
    archive = sequence_directory / "sequence.npz"
    cache = sequence_directory / "sequence.npz.aux"
    previous = archive.read_bytes(), cache.read_bytes()
    sequence.add_pulse("analog", 1, 2, sequence_blocks=["long", "short"])
    sequence.write()
    with pytest.raises(ValueError, match="exceeds sequence bounds"):
        compiler.translate_yaml_to_numeric_instructions()
    assert (archive.read_bytes(), cache.read_bytes()) == previous


def test_override_change_invalidates_cache_and_unchanged_input_reuses_it(
    sequence_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequence = YamlSequence(4, block_durations={"idle": 2})
    sequence.sequencing_order = ["idle"]
    sequence.sequencing_repeats = [1]
    sequence.write()
    compiler = designer.PulseSequenceYaml({}, [1], 1e6)
    compiler.translate_yaml_to_numeric_instructions()
    sequence.set_block_duration("idle", 3)
    sequence.write()
    compiler.translate_yaml_to_numeric_instructions()
    assert (
        read_awg_archive(sequence_directory / "sequence.npz")
        .blocks["idle"]
        .sample_count
        == 3
    )
    save = Mock(
        side_effect=AssertionError("Unchanged input must reuse its compiled archive")
    )
    monkeypatch.setattr(np, "savez", save)
    compiler.translate_yaml_to_numeric_instructions()
    save.assert_not_called()


def test_legacy_cache_regenerates_without_unpickling(
    sequence_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequence = YamlSequence(4)
    sequence.add_pulse("analog", 0, 1)
    sequence.sequencing_order = ["block_0"]
    sequence.sequencing_repeats = [1]
    sequence.write()
    compiler = designer.PulseSequenceYaml({"analog": 0}, [1], 1e6)
    compiler.translate_yaml_to_numeric_instructions()
    path = sequence_directory / "sequence.npz"
    np.savez(
        path, np.zeros((1, 2, 4)), arr_5=np.asarray(b"legacy metadata must not be read")
    )
    cache_path = sequence_directory / "sequence.npz.aux"
    cache = yaml.safe_load(cache_path.read_text(encoding="utf-8"))
    cache["instructions"]["compiler_version"] = 1
    cache["artifact"] = [path.stat().st_mtime_ns, path.stat().st_size]
    cache_path.write_text(yaml.safe_dump(cache), encoding="utf-8")
    unpickle = Mock(side_effect=AssertionError("Legacy caches must never be unpickled"))
    monkeypatch.setattr(pickle, "loads", unpickle)
    compiler.translate_yaml_to_numeric_instructions()
    assert read_awg_archive(path).blocks["block_0"].sample_count == 4
    unpickle.assert_not_called()


def test_uniform_numeric_builder_retains_arrays_and_writes_supported_archive(
    sequence_directory: Path,
) -> None:
    sequence = designer.PulseSequence(2, 4, [1], samprate=1e6)
    sequence.add_pulse((0, 2), 1, 2, amplitude=0.25)
    sequence.sequencernames = ["z", "a", "z"]
    sequence.sequencer = [1, 2, 1]
    sequence.make("direct")
    assert isinstance(sequence.pulses, np.ndarray)
    assert sequence.pulses.shape == (2, 5, 4)
    assert sequence.num_points == 4
    assert sequence.time.shape == (4,)
    compiled = read_awg_archive(sequence_directory / "direct.npz")
    assert sorted(compiled.blocks) == ["a", "z"]
    assert compiled.flag_values == {"a": [], "z": []}
    np.testing.assert_array_equal(compiled.blocks["a"].samples[0], [0, 0.25, 0.25, 0])


def test_direct_builder_rejects_legacy_pickled_flags(sequence_directory: Path) -> None:
    sequence = designer.PulseSequence(1, 4, [1], samprate=1e6)
    sequence.sequencernames = ["block_0"]
    sequence.sequencer = [1]
    setattr(sequence, "flag_channels", b"legacy metadata")
    with pytest.raises(ValueError, match="replace legacy pickled bytes"):
        sequence.make("rejected.npz")
    assert not (sequence_directory / "rejected.npz").exists()


def test_public_builder_accepts_numpy_scalar_metadata(sequence_directory: Path) -> None:
    sequence = designer.PulseSequence(
        1, cast(float, np.float64(4)), [1], samprate=cast(float, np.float64(1e6))
    )
    sequence.add_pulse(0, 1, 2, amplitude=0.25)
    sequence.sequencernames = ["block_0"]
    setattr(sequence, "sequencer", [np.int64(2)])
    original_samples = sequence.pack_waveforms().copy()
    sequence.make("numpy-scalars.npz")
    compiled = read_awg_archive(sequence_directory / "numpy-scalars.npz")
    assert compiled.repeat_counts == (2,)
    np.testing.assert_array_equal(
        compiled.blocks["block_0"].samples, original_samples[0]
    )


def test_length_preflight_runs_before_allocation_and_on_cache_hits(
    sequence_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequence = YamlSequence(4, block_durations={"idle": 2})
    sequence.sequencing_order = ["idle"]
    sequence.sequencing_repeats = [1]
    sequence.write()
    validator = Mock()
    compiler = designer.PulseSequenceYaml(
        {}, [1], 1e6, sample_count_validator=validator
    )
    compiler.translate_yaml_to_numeric_instructions()
    validator.assert_called_once_with("idle", 2)
    previous = (sequence_directory / "sequence.npz").read_bytes()
    validator.reset_mock()
    validator.side_effect = ValueError("changed device length limit")
    allocate = Mock(
        side_effect=AssertionError("Must validate lengths before allocation")
    )
    monkeypatch.setattr(designer, "PulseSequence", allocate)
    with pytest.raises(ValueError, match="changed device length limit"):
        compiler.translate_yaml_to_numeric_instructions()
    validator.assert_called_once_with("idle", 2)
    allocate.assert_not_called()
    assert (sequence_directory / "sequence.npz").read_bytes() == previous
