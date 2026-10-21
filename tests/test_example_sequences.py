"""Check the public example entry points and their original numeric helpers."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from qupyt.set_up import get_seq_dir, make_userdirs
from source.example_pulse_sequences import (
    casr,
    casr_calibration,
    daq_sweep,
    odmr_cam,
    odmr_pd_daq,
    rabi_pd_daq,
)


@pytest.mark.parametrize(
    "example,key",
    [
        (casr, "AWG_frequency"),
        (casr_calibration, "AWG_frequency"),
        (daq_sweep, "number_measurements"),
        (odmr_cam, "mw_duration"),
        (odmr_pd_daq, "mw_duration"),
        (rabi_pd_daq, "max_mw_duration"),
    ],
)
def test_missing_required_example_parameter_reports_its_key(
    example: ModuleType, key: str
) -> None:
    with pytest.raises(KeyError) as caught:
        example.generate_sequence({})
    assert caught.value.args == (key,)


@pytest.mark.parametrize("example", [casr, casr_calibration])
@pytest.mark.parametrize(
    "values,steps,step,expected",
    [
        (None, 3, 0, None),
        ("5.5", 1, 0, 5.5),
        (["1.5", "2.5"], 1, 0, 2.5),
        (["1", "5"], 3, 1, 3.0),
        (["1", "3.5", "5"], 3, 1, 3.5),
        ([1, 2, 3, 4], 3, 1, None),
    ],
)
def test_sweep_helper_returns_optional_float_consistently(
    example: ModuleType,
    values: str | list[str] | list[int] | None,
    steps: int,
    step: int,
    expected: float | None,
) -> None:
    value = example._value_for_step(
        step, sweep_values=values, pulse_sequence_steps=steps
    )
    assert value == expected
    assert value is None or isinstance(value, float)


@pytest.mark.parametrize("example", [casr, casr_calibration])
@pytest.mark.parametrize(
    "values,steps,step,expected",
    [
        (None, 3, 0, None),
        ("XY4", 1, 0, "XY4"),
        (["XY8", "XY4"], 1, 0, "XY4"),
        (["XY4", "XY8"], 2, 0, "XY4"),
        (("XY4", "XY8"), 2, 1, "XY8"),
        (["XY4", "XY8", "XY4"], 3, 1, "XY8"),
    ],
)
def test_seq_type_sweep_helper_preserves_sequence_names(
    example: ModuleType,
    values: str | list[str] | tuple[str, str] | None,
    steps: int,
    step: int,
    expected: str | None,
) -> None:
    assert (
        example._value_for_step(
            step, sweep_values=values, pulse_sequence_steps=steps, categorical=True
        )
        == expected
    )


@pytest.mark.parametrize("example", [casr, casr_calibration])
def test_seq_type_sweep_requires_a_sequence_name_for_each_step(
    example: ModuleType,
) -> None:
    with pytest.raises(ValueError, match="one sequence name per pulse_sequence_steps"):
        example._value_for_step(
            0, sweep_values=["XY4", "XY8"], pulse_sequence_steps=3, categorical=True
        )


@pytest.mark.parametrize("example", [casr, casr_calibration])
@pytest.mark.parametrize("families", [["XY4"], ["XY4", "XY8"], ["XY4", "XY8", "XY4"]])
def test_seq_type_sweep_generates_the_same_pulses_as_individual_sequences(
    example: ModuleType,
    families: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    make_userdirs()
    config_path = (
        Path(__file__).resolve().parents[1]
        / "source/example_configs/CASR_calibration_1.yaml"
    )
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pulse_params: dict[str, Any] = config["pulse_sequence"]
    sweep_params = {
        **pulse_params,
        "sweep_param": "seq_type",
        "sweep_values": families,
        "pulse_sequence_steps": len(families),
    }
    sweep_result = example.generate_sequence(sweep_params)
    sequence_dir = get_seq_dir()
    assert sorted(path.name for path in sequence_dir.glob("sequence_*.yaml")) == [
        f"sequence_{step}.yaml" for step in range(len(families))
    ]
    generated_sequences = [
        yaml.safe_load(
            (sequence_dir / f"sequence_{step}.yaml").read_text(encoding="utf-8")
        )
        for step in range(len(families))
    ]
    for family, generated_sequence in zip(families, generated_sequences, strict=True):
        individual_result = example.generate_sequence(
            {**pulse_params, "seq_type": family}
        )
        individual_sequence = yaml.safe_load(
            (sequence_dir / "sequence_0.yaml").read_text(encoding="utf-8")
        )
        assert generated_sequence == individual_sequence
    assert sweep_result == individual_result


def test_rabi_discrete_grid_returns_float_values() -> None:
    values = rabi_pd_daq.linspace_discrete_by_intervals(4, 4, 1e6)
    assert values == [1.0, 2.0, 3.0, 4.0]
    assert all(isinstance(value, float) for value in values)


@pytest.mark.parametrize("example", [casr, casr_calibration, odmr_cam])
@pytest.mark.parametrize("steps", [0, -1])
def test_examples_reject_nonpositive_sequence_step_counts(
    example: ModuleType, steps: int
) -> None:
    with pytest.raises(ValueError, match="pulse_sequence_steps must be positive"):
        example.generate_sequence({"pulse_sequence_steps": steps})


@pytest.mark.parametrize("example", [casr, casr_calibration])
def test_sweep_helpers_reject_empty_values(example: ModuleType) -> None:
    with pytest.raises(ValueError, match="sweep_values must not be empty"):
        example._value_for_step(0, sweep_values=[], pulse_sequence_steps=1)


def test_camera_example_rejects_empty_sweep_values() -> None:
    with pytest.raises(ValueError, match="sweep_values must not be empty"):
        odmr_cam.generate_sequence(
            {"sweep_param": "mw_duration", "sweep_values": [], "mw_duration": 1}
        )


def test_camera_example_writes_each_requested_step_without_a_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    make_userdirs()
    result = odmr_cam.generate_sequence(
        {
            "pulse_sequence_steps": 3,
            "mw_duration": 1,
            "laser_duration": 4,
            "number_measurements": 2,
            "readout_time": 1,
            "max_framerate": 1000,
        }
    )
    assert result == {}
    assert sorted(path.name for path in get_seq_dir().glob("sequence_*.yaml")) == [
        "sequence_0.yaml",
        "sequence_1.yaml",
        "sequence_2.yaml",
    ]


@pytest.mark.parametrize("example", [casr, casr_calibration])
def test_casr_timing_errors_report_the_requested_values_in_microseconds(
    example: ModuleType,
) -> None:
    with pytest.raises(ValueError, match="Requested spacing") as caught:
        example.generate_sequence(
            {
                "AWG_frequency": 1e6,
                "seq_type": "XY8",
                "N": 1,
                "tau": 1,
                "pi_half": 0.15,
                "pi": 0.25,
                "laser_duration": 4,
                "number_measurements": 4,
                "sampling_rate": 10e6,
            }
        )
    assert "pi_half = 0.150 µs" in str(caught.value)
    assert "pi = 0.250 µs" in str(caught.value)
