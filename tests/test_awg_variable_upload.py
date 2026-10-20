"""Exercise production AWG compilation, archive loading, and VISA transfers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
import pyvisa
import yaml
from pyvisa.resources import MessageBasedResource

from qupyt.hardware.awg_sequence import (
    get_awg_sequence_options,
    validate_awg_sequence_plan,
)
from qupyt.hardware.synchronisers import AWGenerator, MockGenerator
from qupyt.pulse_sequences import SequenceDesigner as designer
from qupyt.pulse_sequences.SequenceDesigner import PulseSequenceYaml
from qupyt.set_up import get_seq_dir, make_userdirs

LEGACY_TRANSMISSIONS = json.loads(
    (Path(__file__).parent / "fixtures" / "awg_legacy_transmissions.json").read_text(
        encoding="utf-8"
    )
)


class BinaryTransport:
    """Record bytes produced by PyVISA's actual binary serializer."""

    _write_termination = "\n"
    _encoding = "ascii"

    def __init__(self) -> None:
        self.messages: list[bytes] = []

    def write_raw(self, message: bytes) -> int:
        self.messages.append(message)
        return len(message)


@pytest.fixture
def awg_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[AWGenerator, Mock]:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    make_userdirs()
    resource = Mock()
    query_responses = {
        "*IDN?": "TEKTRONIX,AWG70002B,test,1",
        "WLIST:WAVEFORM:LMINIMUM?": "1",
        "WLIST:WAVEFORM:GRANULARITY?": "1",
        "WLIST:WAVEFORM:LMAXIMUM?": "1000000",
    }
    resource.query.side_effect = lambda command: query_responses.get(
        command, '0,"No error"'
    )
    manager = Mock()
    manager.open_resource.return_value = resource
    monkeypatch.setattr(pyvisa, "ResourceManager", Mock(return_value=manager))
    awg = AWGenerator({"address": "test", "sampling_rate": 1e6}, {})
    monkeypatch.setattr(awg, "opc_wait", lambda: None)
    resource.reset_mock()
    return awg, resource


def write_variable_sequence(path: Path, mode: str = "flat") -> dict[str, Any]:
    """Write unequal blocks with a repeated child and a prefix and suffix."""
    pulse = {"start": 0, "duration": 1, "amplitude": 0.25, "frequency": 0, "phase": 0}
    instructions: dict[str, Any] = {
        "total_duration": 4,
        "block_durations": {"x": 6, "y": 3, "tail": 2},
        "sequencing_order": ["gate", "x", "y", "x", "y", "tail"],
        "sequencing_repeats": [1, 2, 1, 2, 1, 1],
        "awg_sequence_mode": mode,
        "awg_repeated_child": {"prefix_steps": 1, "child_steps": 2, "repetitions": 2},
        "gate": {},
        "x": {"analog": {"pulse1": pulse}, "marker": {"pulse1": pulse}},
        "y": {"flag": {"pulse1": pulse}},
        "tail": {},
    }
    path.write_text(yaml.safe_dump(instructions), encoding="utf-8")
    return instructions


@pytest.mark.parametrize("sources", [[1], [1, 2], [2, 1]])
@pytest.mark.parametrize("mode", ["flat", "repeated_child"])
def test_variable_blocks_upload_with_physical_source_order_and_flags(
    awg_environment: tuple[AWGenerator, Mock], sources: list[int], mode: str
) -> None:
    awg, resource = awg_environment
    awg.channels = sources
    flag_selector = f"A{sources[0]}"
    awg.channel_mapping = {"analog": 0, "marker": 1, "flag": flag_selector}
    awg.flag_channels = [flag_selector]
    path = get_seq_dir() / "variable.yaml"
    write_variable_sequence(path, mode)
    awg.load_sequence(path)

    assert awg.compiled_sequence.body_sample_count == 36
    assert awg.compiled_sequence.body_duration_s == pytest.approx(36e-6)
    assert {
        name: block.samples.shape[1]
        for name, block in awg.compiled_sequence.blocks.items()
    } == {"gate": 4, "tail": 2, "x": 6, "y": 3}
    commands = [call.args[0] for call in resource.write.call_args_list]
    binary_calls = resource.write_binary_values.call_args_list
    for source_index, source in enumerate(sources):
        for name, count in [("gate", 4), ("tail", 2), ("x", 6), ("y", 3)]:
            assert f'wlist:waveform:new "{name}_{source}",{count},real' in commands
        analog_call = next(
            call
            for call in binary_calls
            if call.args[0] == f'wlist:waveform:data "x_{source}",'
        )
        expected = np.zeros(6)
        if source_index == 0:
            expected[0] = 0.25
        np.testing.assert_array_equal(analog_call.args[1], expected)
        marker_call = next(
            call
            for call in binary_calls
            if call.args[0] == f'wlist:waveform:marker:data "x_{source}",'
        )
        marker_expected = np.zeros(6, dtype=np.uint8)
        if source_index == 0:
            marker_expected[0] = 32
        np.testing.assert_array_equal(marker_call.args[1], marker_expected)
        assert marker_call.args[1].dtype == np.uint8
        assert analog_call.kwargs == {
            "datatype": "f",
            "is_big_endian": False,
            "header_fmt": "ieee",
        }
        assert marker_call.kwargs == {
            "datatype": "B",
            "is_big_endian": False,
            "header_fmt": "ieee",
        }
        assert f'SOURCE{source}:CASS:SEQUENCE "autoseq",{source_index + 1}' in commands

    flag_commands = [command for command in commands if ":Aflag " in command]
    assert any("HIGH" in command for command in flag_commands)
    assert any("LOW" in command for command in flag_commands)
    if mode == "repeated_child":
        assert 'slist:sequence:new "child",2,1' in commands or (
            'slist:sequence:new "child",2,2' in commands
        )
        assert 'slist:sequence:step3:rcount "autoseq",2' in commands
    else:
        assert f'slist:sequence:new "sub",6,{len(sources)}' in commands
    with pytest.raises(ValueError, match="different sample counts"):
        _ = awg.waveform_block


@pytest.mark.parametrize("analog", [[0.25, -0.5, 1.0], [1.3, -1.3, -0.0]])
def test_waveform_uploader_uses_real_pyvisa_serialization(
    awg_environment: tuple[AWGenerator, Mock], analog: list[float]
) -> None:
    awg, resource = awg_environment
    transport = BinaryTransport()
    resource.write_binary_values.side_effect = lambda message, values, **kwargs: (
        MessageBasedResource.write_binary_values(
            cast(Any, transport), message, values, **kwargs
        )
    )
    waveform = np.asarray([analog, [128, 64, 240]], dtype=np.float64)
    awg._upload_waveform("wire", waveform)
    assert transport.messages == [
        b'wlist:waveform:data "wire",#212'
        + np.asarray(waveform[0], dtype="<f4").tobytes()
        + b"\n",
        b'wlist:waveform:marker:data "wire",#13\x80\x40\xf0\n',
    ]


@pytest.mark.parametrize("sources", [[1], [1, 2], [2, 1]])
@pytest.mark.parametrize("mode", ["flat", "repeated_child"])
@pytest.mark.parametrize("variable", [False, True])
def test_legacy_amplitudes_survive_compilation_loading_and_binary_upload(
    awg_environment: tuple[AWGenerator, Mock],
    sources: list[int],
    mode: str,
    variable: bool,
) -> None:
    awg, resource = awg_environment
    awg.channels = sources
    awg.channel_mapping = {"analog": 0, "marker": 1, "flag": f"A{sources[0]}"}
    awg.flag_channels = [f"A{sources[0]}"]
    path = get_seq_dir() / "legacy_amplitude.yaml"
    instructions = write_variable_sequence(path, mode)
    if not variable:
        instructions.pop("block_durations")
    instructions["x"]["analog"] = {
        "pulse1": {
            "start": 0,
            "duration": 1,
            "amplitude": 1.3,
            "frequency": 0,
            "phase": 0,
        },
        "pulse2": {
            "start": 1,
            "duration": 1,
            "amplitude": -1.3,
            "frequency": 0,
            "phase": 0,
        },
    }
    path.write_text(yaml.safe_dump(instructions), encoding="utf-8")
    transport = BinaryTransport()
    resource.write_binary_values.side_effect = lambda message, values, **kwargs: (
        MessageBasedResource.write_binary_values(
            cast(Any, transport), message, values, **kwargs
        )
    )
    awg.load_sequence(path)
    sample_count = 6 if variable else 4
    expected = np.zeros(sample_count, dtype=np.float64)
    expected[:2] = [1.3, -1.3]
    np.testing.assert_array_equal(
        awg.compiled_sequence.blocks["x"].samples[0], expected
    )
    header = b"#224" if variable else b"#216"
    assert (
        f'wlist:waveform:data "x_{sources[0]}",'.encode("ascii")
        + header
        + expected.astype("<f4").tobytes()
        + b"\n"
    ) in transport.messages


@pytest.mark.parametrize("case", LEGACY_TRANSMISSIONS)
def test_uniform_sequences_preserve_recorded_legacy_commands_and_wire_bytes(
    awg_environment: tuple[AWGenerator, Mock], case: dict[str, Any]
) -> None:
    awg, resource = awg_environment
    awg.channels = case["channels"]
    awg.channel_mapping = case["mapping"]
    awg.flag_channels = awg._extract_flags(awg.channel_mapping)
    path = get_seq_dir() / "legacy_reference.yaml"
    path.write_text(yaml.safe_dump(case["instructions"]), encoding="utf-8")
    PulseSequenceYaml(
        awg.channel_mapping, awg.channels, samprate=1e6, yaml_file=path
    ).translate_yaml_to_numeric_instructions()
    awg._load_sequence_block(get_seq_dir() / "sequence.npz")
    awg.sequence_options = get_awg_sequence_options(case["instructions"])
    calls: list[dict[str, Any]] = []
    resource.write.side_effect = lambda message: calls.append(
        {"kind": "text", "message": message}
    )

    def record_binary(
        message: str, values: Any, datatype: str = "f", **kwargs: Any
    ) -> None:
        payload = pyvisa.util.to_ieee_block(
            values, cast(Any, datatype), kwargs.get("is_big_endian", False)
        )
        calls.append(
            {
                "kind": "binary",
                "message": message,
                "datatype": datatype,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    resource.write_binary_values.side_effect = record_binary
    awg._upload_waveforms()
    awg._sequence("autoseq")
    assert calls == case["calls"]


@pytest.mark.parametrize(
    "waveform",
    [
        np.zeros((3, 2)),
        np.zeros((2, 0)),
        np.asarray([[float("nan")], [0]]),
        np.asarray([[0], [256]]),
        np.asarray([[0], [-1]]),
        np.asarray([[0], [1.5]]),
    ],
)
def test_invalid_waveform_never_reaches_visa(
    awg_environment: tuple[AWGenerator, Mock], waveform: np.ndarray[Any, Any]
) -> None:
    awg, resource = awg_environment
    with pytest.raises(ValueError):
        awg._upload_waveform("invalid", waveform)
    resource.write.assert_not_called()
    resource.write_binary_values.assert_not_called()


@pytest.mark.parametrize("mode", ["flat", "repeated_child"])
def test_aggregate_sequence_capacity_counts_gate_and_stored_children(mode: str) -> None:
    steps = [AWGenerator.AWGSequenceStep(name, 1, ()) for name in ["x", "y"] * 3]
    config = {"child_steps": 2, "repetitions": 3}
    required = 8 if mode == "flat" else 4
    validate_awg_sequence_plan(
        steps,
        mode,
        config,
        max_sequence_steps=required,
        max_step_repeat=8,
        device_type="Test AWG",
    )
    with pytest.raises(ValueError, match="aggregate stored sequence steps"):
        validate_awg_sequence_plan(
            steps,
            mode,
            config,
            max_sequence_steps=required - 1,
            max_step_repeat=8,
            device_type="Test AWG",
        )


@pytest.mark.parametrize("mode", ["flat", "repeated_child"])
def test_mock_variable_sequence_uses_production_archive(
    awg_environment: tuple[AWGenerator, Mock], mode: str
) -> None:
    path = get_seq_dir() / "mock_variable.yaml"
    write_variable_sequence(path, mode)
    mock = MockGenerator(
        {"sampling_rate": 1e6, "channels": [1]},
        {"analog": 0, "marker": 1, "flag": "A1"},
    )
    mock.load_sequence(path)
    assert mock.total_duration == 4000
    assert mock.block_durations == {"gate": 4, "tail": 2, "x": 6, "y": 3}
    assert mock.body_duration_s == pytest.approx(36e-6)
    assert mock.compiled_sequence is not None
    assert mock.compiled_sequence.body_sample_count == 36
    assert (get_seq_dir() / "sequence.npz").is_file()


@pytest.mark.parametrize("failure", ["mode", "child", "capacity", "flags"])
def test_invalid_sequence_is_rejected_before_hardware_is_cleared(
    awg_environment: tuple[AWGenerator, Mock], failure: str
) -> None:
    awg, resource = awg_environment
    awg.channels = [1]
    awg.channel_mapping = {"analog": 0, "marker": 1, "flag": "A1"}
    awg.flag_channels = ["A1"]
    path = get_seq_dir() / "invalid.yaml"
    instructions = write_variable_sequence(path, "repeated_child")
    if failure == "mode":
        instructions["awg_sequence_mode"] = "unknown"
    elif failure == "child":
        instructions["awg_repeated_child"]["child_steps"] = 1
    elif failure == "capacity":
        awg.awg_max_sequence_steps = 5
    else:
        awg.channel_mapping["flag"] = "A2"
    path.write_text(yaml.safe_dump(instructions), encoding="utf-8")
    with pytest.raises(ValueError):
        awg.load_sequence(path)
    resource.write.assert_not_called()
    resource.write_binary_values.assert_not_called()


@pytest.mark.parametrize("invalid_field", ["rate", "sources", "flags", "repeats"])
def test_failed_archive_load_preserves_previously_loaded_state(
    awg_environment: tuple[AWGenerator, Mock], invalid_field: str
) -> None:
    awg, _ = awg_environment
    awg.channels = [1]
    awg.channel_mapping = {"analog": 0, "marker": 1, "flag": "A1"}
    awg.flag_channels = ["A1"]
    path = get_seq_dir() / "valid.yaml"
    write_variable_sequence(path)
    PulseSequenceYaml(
        awg.channel_mapping, awg.channels, samprate=1e6, yaml_file=path
    ).translate_yaml_to_numeric_instructions()
    archive_path = get_seq_dir() / "sequence.npz"
    awg._load_sequence_block(archive_path)
    previous = (awg.compiled_sequence, awg.wavenames, awg.seqrepeats, awg.flag_values)
    if invalid_field == "rate":
        awg.samprate = 2e6
    elif invalid_field == "sources":
        awg.channels = [2]
    elif invalid_field == "repeats":
        awg.awg_max_step_repeat = 1
    else:
        with np.load(archive_path, allow_pickle=False) as archive:
            fields = {name: archive[name] for name in archive.files}
        flags = json.loads(str(fields["flags_json"].item()))
        flags["gate"] = ["A3"]
        fields["flags_json"] = np.asarray(json.dumps(flags))
        np.savez(archive_path, **fields)
    with pytest.raises(ValueError):
        awg._load_sequence_block(archive_path)
    assert awg.compiled_sequence is previous[0]
    assert awg.wavenames is previous[1]
    assert awg.seqrepeats is previous[2]
    assert awg.flag_values is previous[3]


def test_repeated_child_rejects_unsupported_outer_repetition_before_programming(
    awg_environment: tuple[AWGenerator, Mock]
) -> None:
    awg, resource = awg_environment
    awg.wavenames = ["x", "x"]
    awg.seqrepeats = [1, 1]
    awg.flag_values = {"x": []}
    awg.sequence_options = {
        "awg_sequence_mode": "repeated_child",
        "awg_repeated_child": {"child_steps": 1, "repetitions": 2},
    }
    with pytest.raises(ValueError, match="nongatereps=1 only"):
        awg._sequence("autoseq", nongatereps=2)
    resource.write.assert_not_called()


def test_failed_sequence_preflight_restores_running_program_host_state(
    awg_environment: tuple[AWGenerator, Mock]
) -> None:
    awg, resource = awg_environment
    awg.channels = [1]
    awg.channel_mapping = {"analog": 0, "marker": 1, "flag": "A1"}
    awg.flag_channels = ["A1"]
    path = get_seq_dir() / "running.yaml"
    write_variable_sequence(path, "flat")
    awg.load_sequence(path)
    previous = {
        name: getattr(awg, name)
        for name in (
            "compiled_sequence",
            "sequence_options",
            "wavenames",
            "seqrepeats",
            "flag_values",
        )
    }
    resource.reset_mock()
    invalid_path = get_seq_dir() / "invalid_child.yaml"
    instructions = write_variable_sequence(invalid_path, "repeated_child")
    instructions["awg_repeated_child"]["child_steps"] = 1
    invalid_path.write_text(yaml.safe_dump(instructions), encoding="utf-8")
    with pytest.raises(ValueError, match="block mismatch"):
        awg.load_sequence(invalid_path)
    for name, value in previous.items():
        assert getattr(awg, name) is value
    resource.write.assert_not_called()
    resource.write_binary_values.assert_not_called()


def set_waveform_limits(
    resource: Mock, minimum: str, granularity: str, maximum: str
) -> None:
    """Override capability responses while preserving ID and error queries."""
    previous_query = resource.query.side_effect
    limits = {
        "WLIST:WAVEFORM:LMINIMUM?": minimum,
        "WLIST:WAVEFORM:GRANULARITY?": granularity,
        "WLIST:WAVEFORM:LMAXIMUM?": maximum,
    }
    resource.query.side_effect = lambda command: (
        limits[command] if command in limits else previous_query(command)
    )


@pytest.mark.parametrize(
    "minimum,granularity,maximum,block,count",
    [("5", "1", "100", "gate", 4), ("1", "1", "5", "x", 6), ("1", "2", "100", "y", 3)],
)
def test_connected_length_limits_reject_all_blocks_before_numeric_allocation(
    awg_environment: tuple[AWGenerator, Mock],
    monkeypatch: pytest.MonkeyPatch,
    minimum: str,
    granularity: str,
    maximum: str,
    block: str,
    count: int,
) -> None:
    awg, resource = awg_environment
    awg.channels = [1]
    awg.channel_mapping = {"analog": 0, "marker": 1, "flag": "A1"}
    awg.flag_channels = ["A1"]
    path = get_seq_dir() / "hardware_invalid.yaml"
    write_variable_sequence(path)
    set_waveform_limits(resource, minimum, granularity, maximum)
    builder = Mock(side_effect=AssertionError("No numeric allocation before preflight"))
    monkeypatch.setattr(designer, "PulseSequence", builder)
    with pytest.raises(
        ValueError, match=f"block '{block}'.*has {count} samples"
    ) as caught:
        awg.load_sequence(path)
    for limit in (
        f"minimum={minimum}",
        f"maximum={maximum}",
        f"granularity={granularity}",
    ):
        assert limit in str(caught.value)
    builder.assert_not_called()
    resource.write.assert_not_called()
    resource.write_binary_values.assert_not_called()


def test_connected_limits_are_rechecked_when_compiled_archive_is_cached(
    awg_environment: tuple[AWGenerator, Mock]
) -> None:
    awg, resource = awg_environment
    awg.channels = [1]
    awg.channel_mapping = {"analog": 0, "marker": 1, "flag": "A1"}
    awg.flag_channels = ["A1"]
    path = get_seq_dir() / "cached.yaml"
    write_variable_sequence(path)
    awg.load_sequence(path)
    archive_path = get_seq_dir() / "sequence.npz"
    timestamp = archive_path.stat().st_mtime_ns
    previous_sequence = awg.compiled_sequence
    previous_options = awg.sequence_options
    resource.reset_mock()
    set_waveform_limits(resource, "1", "1", "5")
    with pytest.raises(ValueError, match="block 'x'.*has 6 samples"):
        awg.load_sequence(path)
    assert archive_path.stat().st_mtime_ns == timestamp
    assert awg.compiled_sequence is previous_sequence
    assert awg.sequence_options is previous_options
    assert [
        call.args[0]
        for call in resource.query.call_args_list
        if call.args[0].startswith("WLIST:WAVEFORM:")
    ] == [
        "WLIST:WAVEFORM:LMINIMUM?",
        "WLIST:WAVEFORM:GRANULARITY?",
        "WLIST:WAVEFORM:LMAXIMUM?",
    ]
    resource.write.assert_not_called()
    resource.write_binary_values.assert_not_called()


def test_connected_length_bounds_accept_exact_limits_without_padding(
    awg_environment: tuple[AWGenerator, Mock]
) -> None:
    awg, resource = awg_environment
    awg.channels = [1]
    awg.channel_mapping = {"analog": 0, "marker": 1, "flag": "A1"}
    awg.flag_channels = ["A1"]
    path = get_seq_dir() / "exact_limits.yaml"
    instructions = write_variable_sequence(path)
    instructions["block_durations"]["y"] = 8
    path.write_text(yaml.safe_dump(instructions), encoding="utf-8")
    set_waveform_limits(resource, " +2\n", "2", "8")
    awg.load_sequence(path)
    assert {
        name: block.samples.shape[1]
        for name, block in awg.compiled_sequence.blocks.items()
    } == {"gate": 4, "tail": 2, "x": 6, "y": 8}
    assert awg.compiled_sequence.body_sample_count == 46


@pytest.mark.parametrize("response", ["0", "-1", "1.0", "1e3", "", "nan", "1,2"])
def test_invalid_connected_capability_response_prevents_stop_and_upload(
    awg_environment: tuple[AWGenerator, Mock], response: str
) -> None:
    awg, resource = awg_environment
    path = get_seq_dir() / "capability_invalid.yaml"
    write_variable_sequence(path)
    set_waveform_limits(resource, "1", response, "100")
    with pytest.raises(ValueError, match=r"GRANULARITY\?.*expected a positive integer"):
        awg.load_sequence(path)
    resource.write.assert_not_called()
    resource.write_binary_values.assert_not_called()


def test_connected_capability_query_failure_preserves_cause_and_running_state(
    awg_environment: tuple[AWGenerator, Mock]
) -> None:
    awg, resource = awg_environment
    path = get_seq_dir() / "capability_failure.yaml"
    write_variable_sequence(path)
    previous_options = awg.sequence_options
    failure = OSError("Instrument connection lost")
    original_query = resource.query.side_effect

    def query(command: str) -> Any:
        if command == "WLIST:WAVEFORM:LMINIMUM?":
            raise failure
        return original_query(command)

    resource.query.side_effect = query
    with pytest.raises(ValueError, match=r"LMINIMUM\? failed") as caught:
        awg.load_sequence(path)
    assert caught.value.__cause__ is failure
    assert awg.sequence_options is previous_options
    assert not hasattr(awg, "compiled_sequence")
    resource.write.assert_not_called()
    resource.write_binary_values.assert_not_called()


def test_inconsistent_connected_length_bounds_prevent_upload(
    awg_environment: tuple[AWGenerator, Mock]
) -> None:
    awg, resource = awg_environment
    path = get_seq_dir() / "capability_inconsistent.yaml"
    write_variable_sequence(path)
    set_waveform_limits(resource, "10", "1", "9")
    with pytest.raises(ValueError, match="minimum 10 exceeds maximum 9"):
        awg.load_sequence(path)
    resource.write.assert_not_called()
    resource.write_binary_values.assert_not_called()


@pytest.mark.parametrize("empty_overrides", [False, True])
def test_legacy_scalar_sequences_do_not_query_connected_waveform_limits(
    awg_environment: tuple[AWGenerator, Mock], empty_overrides: bool
) -> None:
    awg, resource = awg_environment
    case = LEGACY_TRANSMISSIONS[0]
    awg.channels = case["channels"]
    awg.channel_mapping = case["mapping"]
    awg.flag_channels = awg._extract_flags(awg.channel_mapping)
    instructions = dict(case["instructions"])
    if empty_overrides:
        instructions["block_durations"] = {}
    path = get_seq_dir() / "scalar.yaml"
    path.write_text(yaml.safe_dump(instructions), encoding="utf-8")
    awg.load_sequence(path)
    assert not any(
        call.args[0].startswith("WLIST:WAVEFORM:")
        for call in resource.query.call_args_list
    )


@pytest.mark.parametrize("device", ["real", "mock"])
@pytest.mark.parametrize("overrides", [None, [], "", 0, False])
def test_real_and_mock_reject_falsey_nonmapping_block_durations(
    awg_environment: tuple[AWGenerator, Mock], device: str, overrides: object
) -> None:
    awg, resource = awg_environment
    path = get_seq_dir() / "invalid_duration_metadata.yaml"
    instructions = write_variable_sequence(path)
    instructions["block_durations"] = overrides
    path.write_text(yaml.safe_dump(instructions), encoding="utf-8")
    awg.channel_mapping = {"analog": 0, "marker": 1, "flag": "A1"}
    awg.flag_channels = ["A1"]
    synchroniser = awg if device == "real" else MockGenerator({}, awg.channel_mapping)
    with pytest.raises(ValueError, match="block_durations must be a mapping"):
        synchroniser.load_sequence(path)
    assert not (get_seq_dir() / "sequence.npz").exists()
    resource.write.assert_not_called()
    resource.write_binary_values.assert_not_called()


def test_connected_length_validator_rejects_oversized_single_transfer_without_allocation(
    awg_environment: tuple[AWGenerator, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    awg, resource = awg_environment
    set_waveform_limits(resource, "1", "1", "8000000000")
    allocate = Mock(side_effect=AssertionError("Length preflight must not allocate"))
    monkeypatch.setattr(np, "zeros", allocate)
    validate_sample_count = awg._get_waveform_sample_count_validator("long_block.yaml")
    # Check the exact supported boundary without constructing a waveform.
    validate_sample_count("long", 249_999_999)
    with pytest.raises(
        ValueError, match="single-transfer limit of 249999999"
    ) as caught:
        validate_sample_count("long", 250_000_000)
    assert "maximum=8000000000" in str(caught.value)
    assert "Chunked waveform uploads are not implemented" in str(caught.value)
    allocate.assert_not_called()
    resource.write.assert_not_called()
    resource.write_binary_values.assert_not_called()
