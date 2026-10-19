"""Regression coverage for measurement diagnostics and failure recovery."""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from qupyt.hardware.sensors import DAQ, MockCam
from qupyt.hardware.synchronisers import PulseBlaster, SynchroniserFactory
from qupyt.measurement_logic.data_handling import Data
from qupyt.measurement_logic.run_measurement import (
    run_measurement as execute_measurement,
)
from qupyt.mixins import ConfigurationMixin
from qupyt.pulse_sequences.yaml_sequence import YamlSequence
from tests.support import MeasurementEnvironment

measurement_module = importlib.import_module("qupyt.measurement_logic.run_measurement")
data_module = importlib.import_module("qupyt.measurement_logic.data_handling")


def run_measurement(environment: MeasurementEnvironment) -> str:
    return execute_measurement(
        environment.static,
        environment.dynamic,
        environment.sensor,
        environment.synchroniser,
        environment.params,
    )


def test_measurement_logs_progress_and_saved_outputs(
    measurement_environment: MeasurementEnvironment, caplog: pytest.LogCaptureFixture
) -> None:
    environment = measurement_environment
    assert run_measurement(environment) == "success"
    saved = np.load(environment.params["filename"] + ".npy")
    np.testing.assert_array_equal(saved, np.ones((2, 2, 2, 1, 1)))
    metadata = yaml.safe_load(
        Path(environment.params["filename"] + ".yaml").read_text()
    )
    assert metadata["measurement_status"] == "success"
    assert "pulse sequence 2/2 completed" in caplog.text
    assert "dynamic step 2/2 completed (2 averages)" in caplog.text
    assert "Saved data to" in caplog.text
    assert "Saved measurement metadata to" in caplog.text
    assert "status=success" in caplog.text
    assert "elapsed=" in caplog.text
    assert "Acquired data:" not in caplog.text


def test_debug_logs_acquisition_details(
    measurement_environment: MeasurementEnvironment, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    assert run_measurement(measurement_environment) == "success"
    assert "pulse_sequence=2, dynamic_step=2, average=2" in caplog.text
    assert "shape=(2, 1), dtype=float64" in caplog.text


def test_direct_sensor_acquisition_logs_details(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    sensor = MockCam({"number_measurements": 2, "image_roi": [1]})
    monkeypatch.setattr(np.random, "poisson", Mock(return_value=np.ones((2, 1))))
    data = sensor.acquire_data()
    np.testing.assert_array_equal(data, np.ones((2, 1)))
    assert "Starting MockCam acquisition: measurements=2" in caplog.text
    assert "sensor=MockCam, shape=(2, 1), dtype=float64" in caplog.text
    assert "elapsed=" in caplog.text


def test_sequence_upload_reports_input_path(
    measurement_environment: MeasurementEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    environment = measurement_environment
    environment.params["synchroniser"]["upload"] = True
    sequence_directory = environment.directory / "sequences"
    sequence_directory.mkdir()
    sequence = {
        "total_duration": 10,
        "sequencing_order": ["block_0"],
        "sequencing_repeats": [1],
        "block_0": {},
    }
    for step in range(environment.params["pulse_sequence_steps"]):
        (sequence_directory / f"sequence_{step}.yaml").write_text(
            yaml.safe_dump(sequence), encoding="utf-8"
        )
    monkeypatch.setattr(measurement_module, "get_seq_dir", lambda: sequence_directory)
    upload = Mock(wraps=environment.synchroniser.load_sequence)
    monkeypatch.setattr(environment.synchroniser, "load_sequence", upload)
    assert run_measurement(environment) == "success"
    assert upload.call_count == 2
    assert upload.call_args.args == (sequence_directory / "sequence_1.yaml",)
    assert "Loaded sequence for MockSynchroniser from" in caplog.text
    assert str(sequence_directory / "sequence_1.yaml") in caplog.text


def test_acquisition_failure_reports_position_and_saves_partial_data(
    measurement_environment: MeasurementEnvironment, caplog: pytest.LogCaptureFixture
) -> None:
    environment = measurement_environment
    environment.acquisition.side_effect = [
        np.ones((2, 1)),
        RuntimeError("read timed out"),
    ]
    assert run_measurement(environment) == "failed"
    failure = next(
        record
        for record in caplog.records
        if "failed during sensor acquisition" in record.getMessage()
    )
    assert "pulse_sequence=0, dynamic_step=0, average=1" in failure.getMessage()
    assert failure.exc_info is not None
    assert failure.exc_info[0] is RuntimeError
    assert "Saving partial, unaveraged data" in caplog.text
    assert "status=success" not in caplog.text
    assert np.load(environment.params["filename"] + ".npy")[0, 0, 0, 0, 0] == 1
    environment.sensor_close.assert_called_once()
    environment.synchroniser_close.assert_called_once()


@pytest.mark.parametrize(
    "close_error", [RuntimeError("close failed"), SystemExit("driver exited")]
)
def test_cleanup_failure_does_not_skip_other_cleanup_or_saving(
    measurement_environment: MeasurementEnvironment,
    caplog: pytest.LogCaptureFixture,
    close_error: BaseException,
) -> None:
    environment = measurement_environment
    environment.sensor_close.side_effect = close_error
    environment.synchroniser_close.side_effect = RuntimeError("second close failed")
    assert run_measurement(environment) == "failed"
    environment.synchroniser_close.assert_called_once()
    assert "failed to close sensor" in caplog.text
    assert "failed to close synchroniser" in caplog.text
    assert Path(environment.params["filename"] + ".npy").exists()
    metadata = yaml.safe_load(
        Path(environment.params["filename"] + ".yaml").read_text()
    )
    assert metadata["measurement_status"] == "failed"
    # Completed acquisition is still averaged even if cleanup failed.
    assert metadata["measurement_error"]["message"] == str(close_error)
    assert metadata["measurement_error"]["stage"] == "sensor close"
    assert len(metadata["cleanup_errors"]) == 2
    np.testing.assert_array_equal(np.load(environment.params["filename"] + ".npy"), 1)


def test_allocation_failure_is_logged_without_secondary_error(
    measurement_environment: MeasurementEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    environment = measurement_environment
    monkeypatch.setattr(
        Data, "create_array", Mock(side_effect=MemoryError("allocation failed"))
    )
    assert run_measurement(environment) == "failed"
    assert "failed during data allocation" in caplog.text
    assert "no allocated data to save" in caplog.text
    assert "Saved measurement metadata" in caplog.text
    assert not Path(environment.params["filename"] + ".npy").exists()


def test_static_configuration_failure_is_logged_and_cleans_up(
    measurement_environment: MeasurementEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    environment = measurement_environment
    monkeypatch.setattr(
        environment.static,
        "set_all_params",
        Mock(side_effect=ValueError("bad parameter")),
    )
    assert run_measurement(environment) == "failed"
    assert "failed during static device configuration" in caplog.text
    environment.sensor_close.assert_called_once()
    environment.synchroniser_close.assert_called_once()


def test_sequence_compilation_failure_does_not_report_unopened_daq_close_error(
    measurement_environment: MeasurementEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    environment = measurement_environment
    environment.sensor = DAQ({"number_measurements": 2})
    environment.params["synchroniser"]["upload"] = True
    monkeypatch.setattr(
        environment.synchroniser,
        "load_sequence",
        Mock(side_effect=ValueError("pulse compilation failed")),
    )
    assert run_measurement(environment) == "failed"
    assert "failed during sequence upload" in caplog.text
    assert "failed to close sensor" not in caplog.text
    metadata = yaml.safe_load(
        Path(environment.params["filename"] + ".yaml").read_text()
    )
    assert metadata["measurement_error"]["message"] == "pulse compilation failed"
    assert metadata["measurement_error"]["stage"] == "sequence upload"


def test_acquisition_failure_remains_in_metadata_when_cleanup_and_saving_fail(
    measurement_environment: MeasurementEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    environment = measurement_environment
    environment.acquisition.side_effect = ValueError("primary read error")
    environment.sensor_close.side_effect = OSError("secondary close error")
    monkeypatch.setattr(np, "save", Mock(side_effect=OSError("secondary save error")))
    assert run_measurement(environment) == "failed"
    metadata = yaml.safe_load(
        Path(environment.params["filename"] + ".yaml").read_text()
    )
    assert metadata["measurement_error"]["type"] == "ValueError"
    assert metadata["measurement_error"]["message"] == "primary read error"
    assert "Additional cleanup error after measurement failure" in caplog.text
    environment.synchroniser_close.assert_called_once()


@pytest.mark.parametrize(
    "parameter", ["averages", "dynamic_steps", "pulse_sequence_steps"]
)
def test_zero_measurement_counts_cannot_be_reported_as_success(
    measurement_environment: MeasurementEnvironment, parameter: str
) -> None:
    environment = measurement_environment
    environment.params[parameter] = 0
    assert run_measurement(environment) == "failed"
    environment.acquisition.assert_not_called()
    assert "must be positive" in environment.params["measurement_error"]["message"]


def test_save_failure_marks_metadata_failed(
    measurement_environment: MeasurementEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    environment = measurement_environment
    monkeypatch.setattr(np, "save", Mock(side_effect=OSError("disk full")))
    assert run_measurement(environment) == "failed"
    assert "Failed to save data to" in caplog.text
    assert "failed to save data" in caplog.text
    assert "Saved data to" not in caplog.text
    metadata = yaml.safe_load(
        Path(environment.params["filename"] + ".yaml").read_text()
    )
    assert metadata["measurement_status"] == "failed"


def test_metadata_failure_does_not_report_success(
    measurement_environment: MeasurementEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    environment = measurement_environment
    monkeypatch.setattr(
        data_module.yaml, "dump", Mock(side_effect=OSError("metadata write failed"))
    )
    assert run_measurement(environment) == "failed"
    assert "failed to save metadata" in caplog.text
    assert "status=success" not in caplog.text
    assert "Saved measurement metadata" not in caplog.text
    assert Path(environment.params["filename"] + ".npy").exists()


def test_chunk_save_reports_filename_and_chunk(
    measurement_environment: MeasurementEnvironment, caplog: pytest.LogCaptureFixture
) -> None:
    environment = measurement_environment
    environment.params["data"]["save_in_chunks"] = 2
    assert run_measurement(environment) == "success"
    assert len(list(environment.directory.glob("logging_test_ch-*.npy"))) == 4
    assert "chunk=True" in caplog.text


def test_configuration_setter_keyerror_keeps_original_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = ConfigurationMixin()
    config.attribute_map = {
        "frequency": Mock(side_effect=KeyError("driver key missing"))
    }
    with pytest.raises(KeyError, match="driver key missing"):
        config._update_from_configuration({"frequency": 10})
    assert "Failed to configure ConfigurationMixin.frequency=10" in caplog.text
    assert "Unknown configuration attribute" not in caplog.text


def test_unknown_synchroniser_logs_type_and_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with pytest.raises(ValueError, match="Unknown synchroniser type"):
        SynchroniserFactory.create_synchroniser("UnknownHardware", {}, {})
    record = next(
        record
        for record in caplog.records
        if "Could not create synchroniser" in record.getMessage()
    )
    assert "UnknownHardware" in record.getMessage()
    assert record.exc_info is not None
    assert record.exc_info[0] is ValueError


def test_daq_read_error_reaches_log_and_propagates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensor = DAQ({})
    sensor.NsampsPerDAQread = 2
    sensor.daq_task = Mock()
    sensor.daq_task.read.side_effect = RuntimeError("DAQ disconnected")
    with pytest.raises(RuntimeError, match="DAQ disconnected"):
        sensor.acquire_data()
    record = next(
        record
        for record in caplog.records
        if "DAQ acquisition failed during read" in record.getMessage()
    )
    assert "input=Dev3/ai0" in record.getMessage()
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError


def test_pulseblaster_error_reaches_log_and_propagates(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    module = importlib.import_module("qupyt.hardware.synchronisers")
    monkeypatch.setattr(
        module, "spapi", Mock(pb_get_error=Mock(return_value="board unavailable"))
    )
    synchroniser = PulseBlaster({}, {})
    with pytest.raises(RuntimeError, match="board unavailable"):
        synchroniser.error_catcher(-1)
    assert "PulseBlaster returned status=-1: board unavailable" in caplog.text


def test_yaml_write_failure_logs_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    module = importlib.import_module("qupyt.pulse_sequences.yaml_sequence")
    monkeypatch.setattr(module.set_up, "get_seq_dir", lambda: tmp_path / "missing")
    sequence = YamlSequence(10)
    sequence.sequencing_order = ["block_0"]
    sequence.sequencing_repeats = [1]
    with pytest.raises(FileNotFoundError):
        sequence.write()
    assert "Failed to write pulse sequence to" in caplog.text
    assert "sequence_0.yaml" in caplog.text


def test_cli_captures_import_warnings_and_instruction_failure(tmp_path: Path) -> None:
    # Import main in a separate process because it configures the root logger.
    script = r"""
import builtins
import faulthandler
import logging
from pathlib import Path
from queue import Queue
from types import ModuleType
import sys
from typing import Any
from unittest.mock import Mock, patch

home = Path(sys.argv[1])
sys.argv = ["qupyt", "--debug"]
original_import = builtins.__import__
faulthandler.enable()
faulthandler.dump_traceback_later(60)
def without_egrabber(name: str, *args: Any, **kwargs: Any) -> ModuleType:
    if name == "egrabber":
        raise ImportError("optional driver intentionally unavailable")
    return original_import(name, *args, **kwargs)
with patch.object(Path, "home", return_value=home), patch.object(builtins, "__import__", new=without_egrabber):
    import qupyt.main as main
    instruction = main.get_waiting_room() / "broken.yaml"
    instruction.write_text("ps_path: missing.py\npulse_sequence: {}\n", encoding="utf-8")
    main.queue = Queue()
    main.queue.put(str(instruction))
    main.event_thread = Mock()
    main.event_thread.wait.side_effect = KeyboardInterrupt
    try:
        main.parse_input()
    except KeyboardInterrupt:
        pass
logging.shutdown()
faulthandler.cancel_dump_traceback_later()
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        pytest.fail(
            f"CLI subprocess exceeded {error.timeout} seconds.\n"
            f"Captured stdout:\n{stdout}\nCaptured stderr:\n{stderr}",
            pytrace=False,
        )
    assert result.returncode == 0, result.stderr
    logs = list((tmp_path / ".qupyt" / "log").glob("*.log"))
    assert len(logs) == 1
    content = logs[0].read_text(encoding="utf-8")
    assert "level=DEBUG" in content
    assert "Could not load egrabber library" in content
    assert "[MainThread]" in content
    assert "Claimed instruction file" in content
    assert "Pulse sequence generation failed for missing.py" in content
    assert "Exception while processing measurement instructions" in content
    assert "broken.yaml" in content
    assert "Traceback" in content
