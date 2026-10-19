"""Exercise upload decisions and recovery through the production measurement loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock, call

import numpy as np
import pytest
import yaml
from numpy.typing import NDArray

from qupyt.measurement_logic.run_measurement import run_measurement
from qupyt.pulse_sequences.yaml_sequence import YamlSequence
from qupyt.set_up import get_seq_dir, make_userdirs
from tests.support import MeasurementEnvironment


@pytest.fixture
def upload_environment(
    measurement_environment: MeasurementEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> MeasurementEnvironment:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    make_userdirs()
    for index in range(2):
        sequence = YamlSequence(10)
        sequence.add_pulse("laser", 1, 2)
        sequence.sequencing_order = ["block_0"]
        sequence.sequencing_repeats = [1]
        sequence.write(index)
    synchroniser = measurement_environment.synchroniser
    upload = Mock(wraps=synchroniser.load_sequence)
    monkeypatch.setattr(synchroniser, "load_sequence", upload)
    measurement_environment.sequence_upload = upload
    return measurement_environment


@pytest.mark.parametrize(
    "upload_value,expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("false", False),
        ("yes", True),
        ("no", False),
        (1, True),
        (0, False),
        (" TRUE ", True),
        (" FaLsE ", False),
        (" ON ", True),
        ("off", False),
        ("1", True),
        ("0", False),
        (None, False),
    ],
)
def test_upload_from_yaml_value(
    upload_environment: MeasurementEnvironment, upload_value: Any, expected: bool
) -> None:
    environment = upload_environment
    environment.params["synchroniser"]["upload"] = upload_value
    status = run_measurement(
        environment.static,
        environment.dynamic,
        environment.sensor,
        environment.synchroniser,
        environment.params,
    )
    assert status == "success"
    expected_calls = [
        call(get_seq_dir() / f"sequence_{index}.yaml") for index in range(2)
    ]
    assert environment.sequence_upload is not None
    assert environment.sequence_upload.call_args_list == (
        expected_calls if expected else []
    )
    assert environment.acquisition.call_count == 8


@pytest.mark.parametrize("configuration", [None, {}])
def test_upload_defaults_to_true(
    upload_environment: MeasurementEnvironment, configuration: dict[str, Any] | None
) -> None:
    environment = upload_environment
    if configuration is None:
        del environment.params["synchroniser"]
    else:
        environment.params["synchroniser"] = configuration
    assert (
        run_measurement(
            environment.static,
            environment.dynamic,
            environment.sensor,
            environment.synchroniser,
            environment.params,
        )
        == "success"
    )
    assert environment.sequence_upload is not None
    assert environment.sequence_upload.call_count == 2


def test_later_failure_preserves_completed_steps_without_averaging(
    measurement_environment: MeasurementEnvironment,
) -> None:
    environment = measurement_environment
    error = ValueError("frame seven failed")
    environment.acquisition.side_effect = [
        *(np.full((2, 1), value, dtype=float) for value in range(1, 7)),
        error,
    ]
    assert (
        run_measurement(
            environment.static,
            environment.dynamic,
            environment.sensor,
            environment.synchroniser,
            environment.params,
        )
        == "failed"
    )
    saved = np.load(environment.params["filename"] + ".npy")
    np.testing.assert_array_equal(
        saved[:, :, :, 0, 0], [[[3, 7], [11, 0]], [[3, 7], [11, 0]]]
    )
    metadata = yaml.safe_load(
        Path(environment.params["filename"] + ".yaml").read_text()
    )
    assert metadata["measurement_error"] == {
        "type": "ValueError",
        "message": "frame seven failed",
        "stage": "sensor acquisition",
        "pulse_sequence": 1,
        "dynamic_step": 1,
        "average": 0,
    }
    environment.sensor_close.assert_called_once_with()
    environment.synchroniser_close.assert_called_once_with()


@pytest.mark.parametrize("error", [KeyboardInterrupt("stop"), SystemExit(2)])
def test_interrupt_preserves_exception_and_saves_partial_result(
    measurement_environment: MeasurementEnvironment, error: BaseException
) -> None:
    environment = measurement_environment
    environment.acquisition.side_effect = [np.full((2, 1), 3.0), error]
    with pytest.raises(type(error)) as caught:
        run_measurement(
            environment.static,
            environment.dynamic,
            environment.sensor,
            environment.synchroniser,
            environment.params,
        )
    assert caught.value is error
    saved = np.load(environment.params["filename"] + ".npy")
    np.testing.assert_array_equal(saved[:, 0, 0, 0, 0], [3, 3])
    metadata = yaml.safe_load(
        Path(environment.params["filename"] + ".yaml").read_text()
    )
    assert metadata["measurement_status"] == "failed"
    assert metadata["measurement_error"]["type"] == type(error).__name__
    environment.sensor_close.assert_called_once_with()
    environment.synchroniser_close.assert_called_once_with()


def test_success_clears_stale_failure_metadata(
    measurement_environment: MeasurementEnvironment,
) -> None:
    environment = measurement_environment
    environment.params["measurement_error"] = {"message": "old failure"}
    environment.params["cleanup_errors"] = [{"message": "old cleanup failure"}]
    assert (
        run_measurement(
            environment.static,
            environment.dynamic,
            environment.sensor,
            environment.synchroniser,
            environment.params,
        )
        == "success"
    )
    metadata = yaml.safe_load(
        Path(environment.params["filename"] + ".yaml").read_text()
    )
    assert "measurement_error" not in metadata
    assert "cleanup_errors" not in metadata
    np.testing.assert_array_equal(np.load(environment.params["filename"] + ".npy"), 1)


def test_failed_dynamic_configuration_does_not_advance_step(
    measurement_environment: MeasurementEnvironment,
) -> None:
    environment = measurement_environment
    environment.dynamic.update_devices(
        {
            "source": {
                "device_type": "Mock",
                "address": "test",
                "config": {"unsupported": [1, 3]},
            }
        }
    )
    assert (
        run_measurement(
            environment.static,
            environment.dynamic,
            environment.sensor,
            environment.synchroniser,
            environment.params,
        )
        == "failed"
    )
    assert environment.dynamic.current_dynamic_step == 0
    environment.acquisition.assert_not_called()
    assert (
        environment.params["measurement_error"]["stage"]
        == "dynamic device configuration"
    )
    assert environment.params["measurement_error"]["type"] == "KeyError"


@pytest.mark.parametrize(
    "sensor_error",
    [OSError("sensor close refused"), SystemExit("sensor driver aborted")],
)
def test_multiple_cleanup_failures_are_recorded_and_data_is_averaged(
    measurement_environment: MeasurementEnvironment, sensor_error: BaseException
) -> None:
    environment = measurement_environment
    environment.sensor_close.side_effect = sensor_error
    environment.synchroniser_close.side_effect = RuntimeError(
        "synchroniser close refused"
    )
    assert (
        run_measurement(
            environment.static,
            environment.dynamic,
            environment.sensor,
            environment.synchroniser,
            environment.params,
        )
        == "failed"
    )
    assert environment.acquisition.call_count == 8
    environment.sensor_close.assert_called_once_with()
    environment.synchroniser_close.assert_called_once_with()
    np.testing.assert_array_equal(np.load(environment.params["filename"] + ".npy"), 1)
    metadata = yaml.safe_load(
        Path(environment.params["filename"] + ".yaml").read_text()
    )
    assert metadata["measurement_error"] == {
        "type": type(sensor_error).__name__,
        "message": str(sensor_error),
        "stage": "sensor close",
    }
    assert metadata["cleanup_errors"] == [
        metadata["measurement_error"],
        {
            "type": "RuntimeError",
            "message": "synchroniser close refused",
            "stage": "synchroniser close",
        },
    ]


def test_chunk_write_failure_is_recorded_at_data_accumulation(
    measurement_environment: MeasurementEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = measurement_environment
    environment.params["data"]["save_in_chunks"] = 2
    original_save = np.save
    failure = OSError("chunk disk full")
    # Fail the chunk write, then use the actual NumPy writer for partial data.
    with monkeypatch.context() as context:

        def save_with_chunk_failure(filename: str | Path, array: NDArray[Any]) -> None:
            if "_ch-" in str(filename):
                raise failure
            return original_save(filename, array)

        context.setattr(np, "save", save_with_chunk_failure)
        assert (
            run_measurement(
                environment.static,
                environment.dynamic,
                environment.sensor,
                environment.synchroniser,
                environment.params,
            )
            == "failed"
        )
    assert environment.params["measurement_error"]["message"] == "chunk disk full"
    assert environment.params["measurement_error"]["stage"] == "data accumulation"
    assert not list(environment.directory.glob("*_ch-*.npy"))
    partial = np.load(environment.params["filename"] + ".npy")
    np.testing.assert_array_equal(partial[:, 0, 0, 0, 0], [2, 2])
    assert not np.any(partial[:, 1])


@pytest.mark.parametrize("error", [KeyboardInterrupt("stop"), SystemExit(2)])
def test_interrupt_keeps_primary_even_when_cleanup_fails(
    measurement_environment: MeasurementEnvironment, error: BaseException
) -> None:
    environment = measurement_environment
    environment.acquisition.side_effect = error
    environment.sensor_close.side_effect = OSError("secondary close failure")
    with pytest.raises(type(error)) as caught:
        run_measurement(
            environment.static,
            environment.dynamic,
            environment.sensor,
            environment.synchroniser,
            environment.params,
        )
    assert caught.value is error
    environment.synchroniser_close.assert_called_once_with()
    metadata = yaml.safe_load(
        Path(environment.params["filename"] + ".yaml").read_text()
    )
    assert metadata["measurement_error"]["type"] == type(error).__name__
    assert metadata["cleanup_errors"][0]["message"] == "secondary close failure"
