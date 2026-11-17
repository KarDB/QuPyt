"""Test the original YAML generators, compilers, cache, and numeric file loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import pyvisa
import yaml

from qupyt.hardware import synchronisers
from qupyt.hardware.synchronisers import AWGenerator, PStreamer
from qupyt.pulse_sequences import SequenceDesigner as designer
from qupyt.pulse_sequences.pulse_sequence_handler import (
    update_params_dict,
    write_user_ps,
)
from qupyt.pulse_sequences.yaml_sequence import (
    ArbitrarySequenceWriter,
    ComplexSequence,
    YamlSequence,
)
from qupyt.set_up import get_seq_dir, make_userdirs


@pytest.fixture
def sequence_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    make_userdirs()
    sequence = YamlSequence(4)
    sequence.add_pulse("analog", 1, 2, amplitude=0.25)
    sequence.add_pulse("marker", 2, 1)
    sequence.add_pulse("flag", 0, 1)
    sequence.sequencing_order = ["block_0"]
    sequence.sequencing_repeats = [2]
    sequence.write()
    return get_seq_dir() / "sequence_0.yaml"


@pytest.fixture
def awg_loader(monkeypatch: pytest.MonkeyPatch) -> AWGenerator:
    resource = Mock()
    resource.query.return_value = "TEKTRONIX,AWG70002B,test,1"
    manager = Mock()
    manager.open_resource.return_value = resource
    monkeypatch.setattr(pyvisa, "ResourceManager", Mock(return_value=manager))
    return AWGenerator({"address": "test"}, {})


def test_yaml_to_numeric_archive_to_awg_loader_preserves_values(
    sequence_files: Path, awg_loader: AWGenerator
) -> None:
    compiler = designer.PulseSequenceYaml(
        {"analog": 0, "marker": 1, "flag": "A1"}, [1, 2], 1e6, sequence_files
    )
    compiler.translate_yaml_to_numeric_instructions()
    loader = awg_loader
    loader._load_sequence_block(get_seq_dir() / "sequence.npz")
    assert loader.waveform_block.shape == (1, 4, 4)
    np.testing.assert_array_equal(
        loader.waveform_block[0],
        [[0, 0.25, 0.25, 0], [0, 0, 128, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    )
    assert loader.seqrepeats == [2]
    assert loader.wavenames == ["block_0"]
    assert loader.flag_values == {"block_0": ["A1"]}


def test_pulseblaster_relative_path_uses_sequence_directory(
    sequence_files: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Keep this path-resolution check independent of the deferred edge-removal bug.
    sequence = YamlSequence(4)
    sequence.add_pulse("analog", 1, 2)
    sequence.sequencing_order = ["block_0"]
    sequence.sequencing_repeats = [2]
    sequence.write()
    api = Mock()
    api.us = 1000
    api.PULSE_PROGRAM = 0
    api.pb_start_programming.return_value = 0
    api.pb_stop_programming.return_value = 0
    api.pb_inst_pbonly.return_value = 42
    monkeypatch.setattr(synchronisers, "spapi", api)
    monkeypatch.chdir(tmp_path)
    assert not Path(sequence_files.name).exists()
    sync = synchronisers.PulseBlaster({}, {"analog": 0, "marker": 1, "flag": 2})
    sync.load_sequence(sequence_files.name)
    api.pb_start_programming.assert_called_once_with(0)
    assert api.pb_inst_pbonly.call_count > 1
    api.pb_stop_programming.assert_called_once_with()


def test_awg_default_sequence_path_uses_current_user_directory(
    sequence_files: Path,
) -> None:
    compiler = designer.PulseSequenceYaml({}, [1], 1e6)
    assert compiler.yaml_file == sequence_files


def test_pulseblaster_default_sequence_path_uses_current_user_directory(
    sequence_files: Path,
) -> None:
    compiler = designer.PulseBlasterSequence({"analog": 0, "marker": 1, "flag": 2})
    assert compiler.yaml_sequence == yaml.safe_load(
        sequence_files.read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("sync_type", ["pulsestreamer", "mock"])
def test_sequence_loaders_reject_python_specific_yaml_tags(
    sync_type: str, tmp_path: Path
) -> None:
    sequence_file = tmp_path / "sequence.yaml"
    sequence_file.write_text(
        "sequencing_order: [block_0]\n"
        "sequencing_repeats: [1]\n"
        "total_duration: 4\n"
        "block_0: {}\n"
        "metadata: !!python/tuple [1, 2]\n",
        encoding="utf-8",
    )
    sync = (
        PStreamer({"address": "192.0.2.1"}, {})
        if sync_type == "pulsestreamer"
        else synchronisers.MockGenerator({}, {})
    )
    with pytest.raises(yaml.constructor.ConstructorError):
        sync.load_sequence(sequence_file)


def test_complex_sequence_default_blocks_are_independent() -> None:
    first = ComplexSequence(YamlSequence(10), "mw", 1, 0.1, 0.2)
    second = ComplexSequence(YamlSequence(10), "mw", 1, 0.1, 0.2)
    first.blocks.append("extra")
    assert second.blocks == ["block_0"]


def test_arbitrary_sequence_default_blocks_are_independent() -> None:
    first = ArbitrarySequenceWriter("mw", 1, 0.2, 0.1, 1, 1e6)
    second = ArbitrarySequenceWriter("mw", 1, 0.2, 0.1, 1, 1e6)
    first.blocks.append("extra")
    assert second.blocks == ["block_0"]


def test_explicit_empty_sequence_blocks_do_not_add_a_default_block() -> None:
    sequence = YamlSequence(10)
    sequence.add_pulse("mw", 0, 1, sequence_blocks=[])
    assert sequence.pulse_sequence == {"total_duration": 10}


def test_unchanged_sequence_reuses_archive_without_writing(
    sequence_files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler = designer.PulseSequenceYaml(
        {"analog": 0, "marker": 1, "flag": "A1"}, [1, 2], 1e6, Path("sequence_0.yaml")
    )
    compiler.translate_yaml_to_numeric_instructions()
    path = get_seq_dir() / "sequence.npz"
    original = path.read_bytes()
    timestamp = path.stat().st_mtime_ns
    write = Mock(side_effect=AssertionError("cache hit must not rewrite archive"))
    monkeypatch.setattr(np, "savez", write)
    compiler.translate_yaml_to_numeric_instructions()
    write.assert_not_called()
    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == timestamp


def test_cache_from_previous_compiler_is_regenerated(
    sequence_files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler = designer.PulseSequenceYaml(
        {"analog": 0, "marker": 1, "flag": "A1"}, [1, 2], 1e6, sequence_files
    )
    compiler.translate_yaml_to_numeric_instructions()
    cache_path = get_seq_dir() / "sequence.npz.aux"
    previous = yaml.safe_load(cache_path.read_text(encoding="utf-8"))
    previous["instructions"].pop("compiler_version")
    cache_path.write_text(yaml.safe_dump(previous), encoding="utf-8")
    save = Mock(wraps=np.savez)
    monkeypatch.setattr(np, "savez", save)
    compiler.translate_yaml_to_numeric_instructions()
    save.assert_called_once()
    assert compiler._sequence_didnt_change()


@pytest.mark.parametrize("cache_content", ["[unclosed", "null", "42", "{}"])
def test_corrupt_cache_regenerates_valid_archive(
    sequence_files: Path, cache_content: str
) -> None:
    compiler = designer.PulseSequenceYaml(
        {"analog": 0, "marker": 1, "flag": "A1"}, [1, 2], 1e6, sequence_files
    )
    compiler.translate_yaml_to_numeric_instructions()
    (get_seq_dir() / "sequence.npz.aux").write_text(cache_content, encoding="utf-8")
    compiler.translate_yaml_to_numeric_instructions()
    assert compiler._sequence_didnt_change()
    with np.load(get_seq_dir() / "sequence.npz") as archive:
        np.testing.assert_array_equal(archive["arr_0"][0, 0], [0, 0.25, 0.25, 0])


def test_changed_flag_mapping_invalidates_cache(
    sequence_files: Path, awg_loader: AWGenerator
) -> None:
    for selector in ("A1", "A2", "A1"):
        compiler = designer.PulseSequenceYaml(
            {"analog": 0, "marker": 1, "flag": selector}, [1, 2], 1e6, sequence_files
        )
        compiler.translate_yaml_to_numeric_instructions()
        loader = awg_loader
        loader._load_sequence_block(get_seq_dir() / "sequence.npz")
        assert loader.flag_values == {"block_0": [selector]}


def test_changed_sampling_rate_regenerates_waveform(sequence_files: Path) -> None:
    mapping = {"analog": 0, "marker": 1, "flag": "A1"}
    for rate, expected in (
        (1e6, [0, 0.25, 0.25, 0]),
        (2e6, [0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0]),
    ):
        compiler = designer.PulseSequenceYaml(mapping, [1, 2], rate, sequence_files)
        compiler.translate_yaml_to_numeric_instructions()
        with np.load(get_seq_dir() / "sequence.npz") as archive:
            np.testing.assert_array_equal(archive["arr_0"][0, 0], expected)


def test_corrupt_numeric_artifact_is_recompiled(sequence_files: Path) -> None:
    compiler = designer.PulseSequenceYaml(
        {"analog": 0, "marker": 1, "flag": "A1"}, [1, 2], 1e6, sequence_files
    )
    compiler.translate_yaml_to_numeric_instructions()
    (get_seq_dir() / "sequence.npz").write_bytes(b"broken archive")
    compiler.translate_yaml_to_numeric_instructions()
    with np.load(get_seq_dir() / "sequence.npz") as archive:
        assert archive["arr_0"].shape == (1, 4, 4)


def test_archive_read_error_preserved_when_archive_close_also_fails(
    awg_loader: AWGenerator,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = OSError("invalid archive")
    archive = MagicMock()
    archive.__getitem__.side_effect = error
    archive.close.side_effect = OSError("archive close failed")
    monkeypatch.setattr(np, "load", Mock(return_value=archive))
    loader = awg_loader
    with pytest.raises(OSError) as caught:
        loader._load_sequence_block(Path("invalid.npz"))
    assert caught.value is error
    archive.close.assert_called_once_with()
    assert "Additional error during numeric sequence file close" in caplog.text


@pytest.mark.parametrize(
    "selector,channels,expected",
    [("A", [1], ("A", None)), ("D2", [1, 2], ("D", 2)), ("B1", [1], ("B", 1))],
)
def test_awg_flag_selector_uses_physical_channel(
    selector: str, channels: list[int], expected: tuple[str, int | None]
) -> None:
    assert designer.parse_awg_flag(selector, channels) == expected


@pytest.mark.parametrize(
    "selector,channels",
    [("A2", [1]), ("A3", [1, 2]), ("A12", [1, 2]), ("a1", [1]), (0, [1])],
)
def test_invalid_awg_flag_selector_rejected(selector: str, channels: list[int]) -> None:
    with pytest.raises(ValueError, match="AWG flag selector"):
        designer.parse_awg_flag(selector, channels)


@pytest.mark.parametrize(
    "instructions",
    [
        None,
        [],
        {"sequencing_order": "block_0", "sequencing_repeats": [1]},
        {"sequencing_order": ["block_0"], "sequencing_repeats": [1.5], "block_0": {}},
        {"sequencing_order": ["block_0"], "sequencing_repeats": [-1], "block_0": {}},
    ],
)
def test_invalid_sequence_metadata_is_rejected(instructions: object) -> None:
    with pytest.raises(ValueError):
        designer.validate_sequence_instructions(instructions)


@pytest.mark.parametrize("start,duration", [(-1, 1), (0, -1), (3, 2)])
def test_invalid_numeric_pulse_interval_does_not_mutate_waveform(
    start: float, duration: float
) -> None:
    sequence = designer.PulseSequence(1, 4, [1], samprate=1e6)
    with pytest.raises(ValueError, match="exceeds sequence bounds"):
        sequence.add_pulse(0, start, duration)
    assert not np.any(sequence.pulses)


def test_numeric_pulse_range_writes_selected_blocks_only() -> None:
    sequence = designer.PulseSequence(3, 4, [1], samprate=1e6)
    sequence.add_pulse((1, 3), 1, 2, amplitude=0.5, inputtype="points")
    np.testing.assert_array_equal(
        sequence.pulses[:, 0], [[0, 0, 0, 0], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0]]
    )
    with pytest.raises(ValueError, match="start and stop index"):
        sequence.add_pulse([0, 1, 2], 0, 1)


def test_waveform_frequency_uses_the_configured_sampling_interval() -> None:
    sequence = designer.PulseSequence(1, 4, [1], samprate=1e6)
    sequence.add_pulse(0, 0, 4, amplitude=0.5, freq=(250_000, 0))
    np.testing.assert_allclose(sequence.time, [0, 1, 2, 3])
    np.testing.assert_allclose(sequence.pulses[0, 0], [0.5, 0, -0.5, 0], atol=1e-12)


def test_unrepresentable_short_sequence_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one sampling point"):
        designer.PulseSequence(1, 0.1, [1], samprate=1e6)


def test_noninteger_sequence_duration_keeps_exact_sampling_interval(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sequence = designer.PulseSequence(1, 4.4, [1], samprate=1e6)
    np.testing.assert_allclose(sequence.time, [0, 1, 2, 3])
    assert "sequence duration is not an integer multiple of samples" in caplog.text


def test_noninteger_pulse_timing_is_rounded_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sequence = designer.PulseSequence(1, 4, [1], samprate=1e6)
    sequence.add_pulse(0, 0.6, 1.4)
    np.testing.assert_array_equal(sequence.pulses[0, 0], [0, 1, 0, 0])
    assert sequence.warning_counter == 2
    assert "rounding to 1" in caplog.text


def test_pulseblaster_compiler_preserves_initial_delay_and_repetitions(
    sequence_files: Path,
) -> None:
    sequence = YamlSequence(4)
    sequence.add_pulse("laser", 1, 2)
    sequence.sequencing_order = ["block_0"]
    sequence.sequencing_repeats = [2]
    sequence.write()
    compiler = designer.PulseBlasterSequence({"laser": 1}, sequence_files)
    compiler.parse_pulse_sequence_file()
    bits, durations = compiler.compile()
    assert bits == [0, 2, 0, 0, 2, 0]
    assert durations == [1, 2, 1, 1, 2, 1]


def test_pulsestreamer_compiles_real_sequence_and_runs_with_driver(
    sequence_files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qupyt.hardware import synchronisers

    sequence = YamlSequence(4)
    sequence.add_pulse("laser", 1, 2)
    sequence.sequencing_order = ["block_0"]
    sequence.sequencing_repeats = [2]
    sequence.write()
    driver = Mock()
    driver.isStreaming.side_effect = [False, True, False]
    monkeypatch.setattr(synchronisers, "PulseStreamer", Mock(return_value=driver))
    monkeypatch.setattr(synchronisers, "sleep", lambda _: None)
    sync = PStreamer({"address": "192.0.2.1"}, {"laser": 0})
    sync.open()
    sync.stop()
    sync.load_sequence(sequence_files)
    assert sync.sequence.getDuration() == 8000
    assert sync.sequence.getDigital()[0][0] == [
        (1000, 0),
        (2000, 1),
        (1000, 0),
        (1000, 0),
        (2000, 1),
        (1000, 0),
    ]
    sync.run()
    driver.stream.assert_called_once()
    assert driver.stream.call_args.args[0] is sync.sequence
    sync.trigger()
    driver.rearm.assert_called_once_with()
    driver.startNow.assert_called_once_with()


@pytest.mark.parametrize("return_value", ["[]", "1", "'invalid'"])
def test_user_generator_rejects_invalid_parameter_return(
    sequence_files: Path, tmp_path: Path, return_value: str
) -> None:
    module = tmp_path / "invalid_return.py"
    module.write_text(
        f"def generate_sequence(params):\n    return {return_value}\n", encoding="utf-8"
    )
    with pytest.raises(TypeError, match="parameter dictionary or None"):
        write_user_ps(module, {})


@pytest.mark.parametrize(
    "source,error",
    [
        ("raise ImportError('module dependency unavailable')", ImportError),
        (
            "def generate_sequence(params):\n    raise ValueError('invalid user pulse')",
            ValueError,
        ),
        ("pass", AttributeError),
    ],
)
def test_user_module_error_is_logged_and_propagated(
    sequence_files: Path,
    tmp_path: Path,
    source: str,
    error: type[Exception],
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = tmp_path / "failed_generator.py"
    module.write_text(source, encoding="utf-8")
    with pytest.raises(error):
        write_user_ps(module, {})
    assert str(module) in caplog.text
    assert "Pulse sequence generation failed" in caplog.text


def test_parameter_updates_merge_nested_values_and_allow_none() -> None:
    params = {
        "sensor": {
            "type": "MockCam",
            "config": {"number_measurements": 2, "image_roi": [1]},
        },
        "averages": 1,
    }
    assert update_params_dict(params, None) is params
    assert (
        update_params_dict(
            params, {"sensor": {"config": {"number_measurements": 4}}, "averages": 2}
        )
        is params
    )
    assert params == {
        "sensor": {
            "type": "MockCam",
            "config": {"number_measurements": 4, "image_roi": [1]},
        },
        "averages": 2,
    }
    with pytest.raises(TypeError, match="must be dictionaries"):
        update_params_dict(params, cast(Any, []))


def test_xy8_sequence_uses_expected_phases_and_pulse_positions() -> None:
    sequence = YamlSequence(30)
    writer = ComplexSequence(
        sequence, "mw", tau=1, pi_half_pulse_dur=0.1, pi_pulse_dur=0.2
    )
    writer.gen_phases("XY8", n=1, readout_phase=0)
    np.testing.assert_allclose(
        writer.phases, [0, 0, np.pi / 2, 0, np.pi / 2, np.pi / 2, 0, np.pi / 2, 0, 0]
    )
    writer.write_sequence(start=2)
    pulses = list(sequence.pulse_sequence["block_0"]["mw"].values())
    np.testing.assert_allclose(
        [pulse["start"] for pulse in pulses], [2, 3, 5, 7, 9, 11, 13, 15, 17, 18.1]
    )
    np.testing.assert_allclose(
        [pulse["duration"] for pulse in pulses],
        [0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1],
    )


@pytest.mark.parametrize("family", ["XY6", "UD2", "UD5", "INVALID8", "XY"])
def test_invalid_decoupling_sequence_is_rejected(family: str) -> None:
    writer = ComplexSequence(
        YamlSequence(30), "mw", tau=1, pi_half_pulse_dur=0.1, pi_pulse_dur=0.2
    )
    with pytest.raises(ValueError):
        writer.gen_phases(family)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="PulseBlaster fix deferred to a separate branch; see PULSEBLASTER_SIMULTANEOUS_EDGES_HANDOFF.md",
)
def test_pulseblaster_simultaneous_edges_keep_correct_channel_bits(
    sequence_files: Path,
) -> None:
    sequence = YamlSequence(4)
    sequence.add_pulse("laser", 0, 2)
    sequence.add_pulse("mw", 0, 2)
    sequence.sequencing_order = ["block_0"]
    sequence.sequencing_repeats = [1]
    sequence.write()
    compiler = designer.PulseBlasterSequence({"laser": 0, "mw": 1}, sequence_files)
    compiler.parse_pulse_sequence_file()
    bits, durations = compiler.compile()
    assert bits == [3, 0]
    assert durations == [2, 2]


@pytest.mark.parametrize(
    "start,duration", [(-1, 1), (1, -1), (float("nan"), 1), (3, 2)]
)
def test_pulseblaster_rejects_invalid_timing(
    sequence_files: Path, start: float, duration: float
) -> None:
    sequence = YamlSequence(4)
    sequence.add_pulse("laser", start, duration)
    sequence.sequencing_order = ["block_0"]
    sequence.sequencing_repeats = [1]
    sequence.write()
    compiler = designer.PulseBlasterSequence({"laser": 0}, sequence_files)
    with pytest.raises(ValueError):
        compiler.parse_pulse_sequence_file()


@pytest.mark.parametrize(
    "sampling_rate,duration,blocks",
    [
        (0, 4, 1),
        (-1, 4, 1),
        (float("nan"), 4, 1),
        (1e6, 0, 1),
        (1e6, -1, 1),
        (1e6, 4, 0),
    ],
)
def test_numeric_sequence_rejects_invalid_dimensions(
    sampling_rate: float, duration: float, blocks: int
) -> None:
    with pytest.raises(ValueError, match="positive sampling rate"):
        designer.PulseSequence(blocks, duration, [1], samprate=sampling_rate)


@pytest.mark.parametrize("device", ["real", "mock"])
@pytest.mark.parametrize("explicit_repetitions", [True, False])
def test_real_and_mock_preserve_the_same_repeated_child_plan(
    device: str, explicit_repetitions: bool, awg_loader: AWGenerator
) -> None:
    sync = awg_loader if device == "real" else synchronisers.MockGenerator({}, {})
    config = {"prefix_steps": 1, "child_steps": 2}
    if explicit_repetitions:
        config["repetitions"] = 2
    sync.sequence_options = {"awg_repeated_child": config}
    steps = [
        AWGenerator.AWGSequenceStep(name, 1, ("A1",) if name == "x" else ())
        for name in ["gate", "x", "y", "x", "y", "tail"]
    ]
    prefix, child, repetitions, suffix = sync._validate_repeated_child_region(steps)
    assert prefix == steps[:1]
    assert child == steps[1:3]
    assert repetitions == 2
    assert suffix == steps[5:]


@pytest.mark.parametrize("device", ["real", "mock"])
@pytest.mark.parametrize(
    "config,max_steps,max_repeats,error",
    [
        ("invalid", 16, 8, "must be a mapping"),
        ({}, 16, 8, "requires child_steps"),
        ({"prefix_steps": -1, "child_steps": 2}, 16, 8, "prefix_steps must be"),
        ({"child_steps": 0}, 16, 8, "child_steps must be"),
        ({"child_steps": 2}, 1, 8, "Repeated child sequence"),
        ({"prefix_steps": 6, "child_steps": 1}, 16, 8, "logical sequence length"),
        (
            {"prefix_steps": 1, "child_steps": 2, "repetitions": 0},
            16,
            8,
            "repetitions must be",
        ),
        (
            {"prefix_steps": 1, "child_steps": 2, "repetitions": 3},
            16,
            8,
            "generated sequence has only",
        ),
        (
            {"prefix_steps": 1, "child_steps": 2, "repetitions": 2},
            16,
            1,
            "repeat count 2 exceeds",
        ),
        (
            {"prefix_steps": 1, "child_steps": 1, "repetitions": 2},
            16,
            8,
            "block mismatch",
        ),
        ({"prefix_steps": 1, "child_steps": 1}, 16, 8, "at least two"),
        ({"prefix_steps": 1, "child_steps": 2, "repetitions": 1}, 3, 8, "Compact AWG"),
    ],
)
def test_real_and_mock_reject_the_same_invalid_repeated_child_plans(
    device: str,
    config: object,
    max_steps: int,
    max_repeats: int,
    error: str,
    awg_loader: AWGenerator,
) -> None:
    sync = awg_loader if device == "real" else synchronisers.MockGenerator({}, {})
    sync._update_from_configuration(
        {"awg_max_sequence_steps": max_steps, "awg_max_step_repeat": max_repeats}
    )
    sync.sequence_options = {"awg_repeated_child": config}
    steps = [
        AWGenerator.AWGSequenceStep(name, 1, ())
        for name in ["gate", "x", "y", "x", "y", "tail"]
    ]
    with pytest.raises(ValueError, match=error):
        sync._validate_repeated_child_region(steps)


@pytest.mark.parametrize("device", ["real", "mock"])
@pytest.mark.parametrize("mismatch", ["repeat", "flags"])
def test_repeated_child_validation_compares_repeat_counts_and_flags(
    device: str, mismatch: str, awg_loader: AWGenerator
) -> None:
    sync = awg_loader if device == "real" else synchronisers.MockGenerator({}, {})
    sync.sequence_options = {"awg_repeated_child": {"child_steps": 1, "repetitions": 2}}
    steps = [
        AWGenerator.AWGSequenceStep("x", 1, ("A1",)),
        AWGenerator.AWGSequenceStep(
            "x",
            2 if mismatch == "repeat" else 1,
            ("B1",) if mismatch == "flags" else ("A1",),
        ),
    ]
    with pytest.raises(ValueError, match="block mismatch"):
        sync._validate_repeated_child_region(steps)


@pytest.mark.parametrize("device", ["real", "mock"])
def test_repetition_inference_rejects_an_empty_child(
    device: str, awg_loader: AWGenerator
) -> None:
    sync = awg_loader if device == "real" else synchronisers.MockGenerator({}, {})
    with pytest.raises(ValueError, match="empty child sequence"):
        sync._infer_child_repetitions([], 0, [])


def test_mock_repeated_child_loading_restores_options_after_validation_failure(
    sequence_files: Path,
) -> None:
    instructions = yaml.safe_load(sequence_files.read_text(encoding="utf-8"))
    instructions["awg_sequence_mode"] = "repeated_child"
    instructions["awg_repeated_child"] = {"child_steps": 1, "repetitions": 2}
    sequence_files.write_text(yaml.safe_dump(instructions), encoding="utf-8")
    sync = synchronisers.MockGenerator({}, {})
    previous_options = {"awg_sequence_mode": "flat"}
    sync.sequence_options = previous_options
    with pytest.raises(ValueError, match="generated sequence has only"):
        sync.load_sequence(sequence_files)
    assert sync.sequence_options is previous_options
