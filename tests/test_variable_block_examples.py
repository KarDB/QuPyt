"""Run the unequal-block examples through measurement and numeric compilation."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import yaml

from qupyt.hardware.device_handler import DeviceHandler, DynamicDeviceHandler
from qupyt.pulse_sequences.awg_archive import CompiledSequence, read_awg_archive
from qupyt.pulse_sequences.pulse_sequence_handler import write_user_ps
from qupyt.pulse_sequences.SequenceDesigner import PulseSequenceYaml
from qupyt.set_up import get_seq_dir, make_userdirs

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_NAMES = ("Variable_blocks_flat", "Variable_blocks_child")
measurement_module = importlib.import_module("qupyt.measurement_logic.run_measurement")


def _load_example(name: str) -> dict[str, Any]:
    """Resolve an example's generator independently of the test launch directory."""
    params: dict[str, Any] = yaml.safe_load(
        (ROOT / "source/example_configs" / f"{name}.yaml").read_text(encoding="utf-8")
    )
    params["ps_path"] = str(ROOT / "source/example_pulse_sequences/variable_blocks.py")
    return params


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_variable_block_mock_measurement_saves_data_and_compiled_waveforms(
    name: str, service: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = _load_example(name)
    monkeypatch.setattr(measurement_module, "sleep", lambda _: None)
    monkeypatch.setattr(measurement_module, "tqdm", lambda iterable, **_: iterable)
    instruction = tmp_path / "instruction.yaml"
    instruction.write_text(yaml.safe_dump(params), encoding="utf-8")
    static = DeviceHandler({})
    dynamic = DynamicDeviceHandler({}, number_dynamic_steps=1)
    try:
        service._process_instruction(instruction, static, dynamic)
    finally:
        static.update_devices({})
        dynamic.update_devices({})

    assert not instruction.exists()
    assert not list(tmp_path.glob("instruction.yaml_failed_*"))
    (metadata_file,) = tmp_path.glob(f"{name}_*.yaml")
    metadata = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    assert metadata["measurement_status"] == "success"
    assert metadata["body_duration_us"] == pytest.approx(13.92)
    assert metadata["sensor"]["config"]["number_measurements"] == 2
    saved_data = np.load(metadata_file.with_suffix(".npy"), allow_pickle=False)
    assert saved_data.shape == (1, 1, 1, 2, 2, 2)
    assert np.isfinite(saved_data).all()

    archive_path = get_seq_dir() / "sequence.npz"
    compiled = read_awg_archive(
        archive_path, expected_source_channels=[1, 2], expected_sample_rate_hz=5e9
    )
    assert compiled.step_block_names == (
        "prep",
        "drive",
        "readout",
        "drive",
        "readout",
        "done",
    )
    assert compiled.repeat_counts == (1, 2, 1, 2, 1, 1)
    assert {name: block.sample_count for name, block in compiled.blocks.items()} == {
        "prep": 6000,
        "drive": 12000,
        "readout": 4800,
        "done": 6000,
    }
    assert compiled.body_sample_count == 69600
    assert compiled.body_duration_s == pytest.approx(13.92e-6)
    assert compiled.flag_values == {
        "prep": [],
        "drive": ["A1"],
        "readout": ["B2"],
        "done": [],
    }
    assert all(block.samples.shape[0] == 4 for block in compiled.blocks.values())
    assert not np.any(compiled.blocks["done"].samples)
    # Waveform idle tails remain present independently of sequencer flag state.
    assert not np.any(compiled.blocks["drive"].samples[:, 5000:])
    assert not np.any(compiled.blocks["readout"].samples[:, 2500:])
    np.testing.assert_array_equal(
        compiled.blocks["readout"].samples[1, 500:1000], np.full(500, 128.0)
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        assert "arr_0" not in archive.files
        assert all(not archive[field].dtype.hasobject for field in archive.files)


def test_flat_and_child_examples_compile_identical_logical_waveforms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    make_userdirs()
    compiled_modes: list[CompiledSequence] = []
    for name in EXAMPLE_NAMES:
        params = _load_example(name)
        updates = write_user_ps(Path(params["ps_path"]), params["pulse_sequence"])
        assert updates is not None and updates["body_duration_us"] == pytest.approx(
            13.92
        )
        PulseSequenceYaml(
            params["synchroniser"]["channel_mapping"],
            [1, 2],
            5e9,
            get_seq_dir() / "sequence_0.yaml",
        ).translate_yaml_to_numeric_instructions()
        compiled_modes.append(read_awg_archive(get_seq_dir() / "sequence.npz"))

    flat, child = compiled_modes
    assert flat.step_block_names == child.step_block_names
    assert flat.repeat_counts == child.repeat_counts
    assert flat.flag_values == child.flag_values
    assert flat.body_sample_count == child.body_sample_count
    for name in flat.blocks:
        np.testing.assert_array_equal(
            flat.blocks[name].samples, child.blocks[name].samples
        )


def test_example_supports_single_output_without_flags_and_numbered_sequences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    make_userdirs()
    params = _load_example("Variable_blocks_child")
    pulse_params = {
        **params["pulse_sequence"],
        "pulse_sequence_steps": 3,
        "second_source": False,
        "include_flags": False,
    }
    updates = write_user_ps(Path(params["ps_path"]), pulse_params)
    assert updates is not None
    assert updates["data"]["ps_steps"] == updates["pulse_sequence_steps"] == 3
    assert sorted(path.name for path in get_seq_dir().glob("sequence_*.yaml")) == [
        "sequence_0.yaml",
        "sequence_1.yaml",
        "sequence_2.yaml",
    ]
    PulseSequenceYaml(
        {"MW": 0, "READ": 1}, [1], 5e9, get_seq_dir() / "sequence_2.yaml"
    ).translate_yaml_to_numeric_instructions()
    compiled = read_awg_archive(get_seq_dir() / "sequence.npz")
    assert compiled.source_channels == (1,)
    assert all(block.samples.shape[0] == 2 for block in compiled.blocks.values())
    assert not any(compiled.flag_values.values())
    assert not np.any(compiled.blocks["prep"].samples)


def test_example_reports_duration_after_independent_sample_rounding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    make_userdirs()
    params = _load_example("Variable_blocks_flat")
    pulse_params = {**params["pulse_sequence"], "block_durations": {"prep": 1.20011}}
    updates = write_user_ps(Path(params["ps_path"]), pulse_params)
    assert updates is not None
    PulseSequenceYaml(
        params["synchroniser"]["channel_mapping"],
        [1, 2],
        5e9,
        get_seq_dir() / "sequence_0.yaml",
    ).translate_yaml_to_numeric_instructions()
    compiled = read_awg_archive(get_seq_dir() / "sequence.npz")
    assert compiled.blocks["prep"].sample_count == 6001
    assert updates["body_duration_us"] == pytest.approx(compiled.body_duration_s * 1e6)
