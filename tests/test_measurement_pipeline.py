"""Integration tests using the real worker, factories, generation, and acquisition.

Only timing and random input are substituted; QuPyt's mock hardware runs its
original methods. The tiny user modules are test inputs to the actual loader.
"""

from __future__ import annotations

import importlib
import logging
import threading
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from qupyt.hardware import signal_sources
from qupyt.set_up import get_seq_dir, get_waiting_room
from tests.support import PipelineFixture

measurement_module = importlib.import_module("qupyt.measurement_logic.run_measurement")


@pytest.fixture
def pipeline(
    service: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> PipelineFixture:
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(measurement_module, "sleep", lambda _: None)
    monkeypatch.setattr(measurement_module, "tqdm", lambda iterable, **_: iterable)
    monkeypatch.setattr(signal_sources, "sleep", lambda _: None)
    acquisition = Mock(return_value=np.ones((2, 1)))
    monkeypatch.setattr(np.random, "poisson", acquisition)
    sensor_factory = Mock(wraps=service.SensorFactory.create_sensor)
    synchroniser_factory = Mock(wraps=service.SynchroniserFactory.create_synchroniser)
    monkeypatch.setattr(service.SensorFactory, "create_sensor", sensor_factory)
    monkeypatch.setattr(
        service.SynchroniserFactory, "create_synchroniser", synchroniser_factory
    )
    generator = tmp_path / "user_sequence.py"
    generator.write_text(
        "from qupyt.pulse_sequences.yaml_sequence import YamlSequence\n"
        "def generate_sequence(params):\n"
        "    for index in range(2):\n"
        "        sequence = YamlSequence(10)\n"
        "        sequence.add_pulse('laser', 1, 2)\n"
        "        sequence.sequencing_order = ['block_0']\n"
        "        sequence.sequencing_repeats = [1]\n"
        "        sequence.write(index)\n"
        "    return {'averages': 2, 'pulse_sequence_steps': 2, 'data': {'ps_steps': 2}}\n",
        encoding="utf-8",
    )
    params = {
        "experiment_type": "pipeline",
        "ps_path": str(generator),
        "pulse_sequence": {},
        "averages": 1,
        "dynamic_steps": 2,
        "pulse_sequence_steps": 1,
        "synchroniser": {
            "type": "MockSynchroniser",
            "config": {},
            "channel_mapping": {},
        },
        "sensor": {
            "type": "MockCam",
            "config": {"number_measurements": 2, "image_roi": [1]},
        },
        "static_devices": {
            "static": {
                "device_type": "Mock",
                "address": "mock-static",
                "config": {"amplitude": -5},
            }
        },
        "dynamic_devices": {
            "dynamic": {
                "device_type": "Mock",
                "address": "mock-dynamic",
                "config": {"frequency": [100, 200]},
            }
        },
        "data": {"averaging_mode": "spread", "dynamic_steps": 2, "ps_steps": 1},
    }
    return service, params, acquisition, sensor_factory, synchroniser_factory


def test_generation_error_retains_instruction_and_next_measurement_succeeds(
    pipeline: PipelineFixture, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    service, params, acquisition, sensor_factory, synchroniser_factory = pipeline
    invalid_generator = tmp_path / "invalid_sequence.py"
    invalid_generator.write_text(
        "def generate_sequence(params):\n    raise ValueError('invalid pulse duration')\n",
        encoding="utf-8",
    )
    failed = get_waiting_room() / "bad.yaml"
    failed.write_text(
        yaml.safe_dump({**params, "ps_path": str(invalid_generator)}), encoding="utf-8"
    )
    valid = get_waiting_room() / "good.yaml"
    valid.write_text(yaml.safe_dump(params), encoding="utf-8")
    service.queue.put(str(failed))
    service.queue.put(str(valid))
    stop_event = threading.Event()
    stop_event.set()

    service.parse_input(stop_event)

    assert service.queue.unfinished_tasks == 0
    assert service.queue.empty()
    assert len(list(get_waiting_room().glob("bad.yaml_failed_*"))) == 1
    assert not valid.exists()
    assert not list(get_waiting_room().glob("*running*"))
    sensor_factory.assert_called_once_with("MockCam", params["sensor"]["config"])
    synchroniser_factory.assert_called_once_with("MockSynchroniser", {}, {})
    assert acquisition.call_count == 8
    metadata_files = list(tmp_path.glob("pipeline_*.yaml"))
    assert len(metadata_files) == 1
    metadata = yaml.safe_load(metadata_files[0].read_text(encoding="utf-8"))
    assert metadata["measurement_status"] == "success"
    assert metadata["averages"] == 2
    assert metadata["data"] == {
        "averaging_mode": "spread",
        "dynamic_steps": 2,
        "ps_steps": 2,
    }
    assert "measurement_error" not in metadata
    saved = np.load(tmp_path / (metadata["filename"] + ".npy"))
    assert saved.shape == (2, 2, 2, 1, 1)
    np.testing.assert_array_equal(saved, 1)
    assert (get_seq_dir() / "sequence_1.yaml").exists()
    assert "ValueError: invalid pulse duration" in caplog.text
    assert "continuing with the queue" in caplog.text
    assert "status=success" in caplog.text
    assert "DAQ" not in caplog.text


def test_acquisition_error_saves_partial_data_and_next_instruction_recovers(
    pipeline: PipelineFixture, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    service, params, acquisition, sensor_factory, _ = pipeline
    acquisition.side_effect = [
        np.full((2, 1), 3.0),
        OSError("camera connection lost"),
        *(np.ones((2, 1)) for _ in range(8)),
    ]
    for experiment in ("failed_acquisition", "recovered"):
        path = get_waiting_room() / f"{experiment}.yaml"
        path.write_text(
            yaml.safe_dump({**params, "experiment_type": experiment}), encoding="utf-8"
        )
        service.queue.put(str(path))
    stop_event = threading.Event()
    stop_event.set()

    service.parse_input(stop_event)

    assert service.queue.unfinished_tasks == 0
    assert acquisition.call_count == 10
    assert sensor_factory.call_count == 2
    assert len(list(get_waiting_room().glob("failed_acquisition.yaml_failed_*"))) == 1
    assert not list(get_waiting_room().glob("recovered*"))
    failed_metadata = yaml.safe_load(
        next(tmp_path.glob("failed_acquisition_*.yaml")).read_text()
    )
    assert failed_metadata["measurement_error"] == {
        "type": "OSError",
        "message": "camera connection lost",
        "stage": "sensor acquisition",
        "pulse_sequence": 0,
        "dynamic_step": 0,
        "average": 1,
    }
    partial = np.load(tmp_path / (failed_metadata["filename"] + ".npy"))
    np.testing.assert_array_equal(partial[:, 0, 0, 0, 0], [3, 3])
    assert not np.any(partial[:, 1])
    assert not np.any(partial[:, 0, 1])
    success = yaml.safe_load(next(tmp_path.glob("recovered_*.yaml")).read_text())
    assert success["measurement_status"] == "success"
    assert "measurement_error" not in success
    np.testing.assert_array_equal(np.load(tmp_path / (success["filename"] + ".npy")), 1)
    assert "Updated MockSignalSource(address: mock-dynamic)" in caplog.text
    assert "OSError: camera connection lost" in caplog.text


def test_idle_worker_wakes_and_stops_cleanly(
    service: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = threading.Event()
    original_ready = service._set_ready

    def notify_ready() -> None:
        original_ready()
        ready.set()

    monkeypatch.setattr(service, "_set_ready", notify_ready)
    stop_event = threading.Event()
    worker = threading.Thread(
        target=service.parse_input, args=(stop_event,), daemon=True
    )
    worker.start()
    try:
        assert ready.wait(timeout=5)
        assert (service.get_home_dir() / "status.txt").read_text() == "ready"
    finally:
        stop_event.set()
        service.event_thread.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert service.queue.unfinished_tasks == 0


def test_duplicate_and_missing_instruction_notifications_do_not_rerun_measurement(
    pipeline: PipelineFixture, caplog: pytest.LogCaptureFixture
) -> None:
    service, params, acquisition, sensor_factory, _ = pipeline
    instruction = get_waiting_room() / "single.yaml"
    instruction.write_text(yaml.safe_dump(params), encoding="utf-8")
    service.queue.put(str(get_waiting_room() / "missing.yaml"))
    service.queue.put(str(instruction))
    service.queue.put(str(instruction))
    stop_event = threading.Event()
    stop_event.set()
    service.parse_input(stop_event)
    assert service.queue.unfinished_tasks == 0
    assert acquisition.call_count == 8
    sensor_factory.assert_called_once()
    assert caplog.text.count("Skipping instruction file that no longer exists") == 2


@pytest.mark.parametrize("content", ["", "[not a mapping]", "sensor: [unterminated"])
def test_invalid_instruction_is_retained_without_hardware_creation(
    pipeline: PipelineFixture, content: str
) -> None:
    service, _, acquisition, sensor_factory, synchroniser_factory = pipeline
    instruction = get_waiting_room() / "invalid.yaml"
    instruction.write_text(content, encoding="utf-8")
    service.queue.put(str(instruction))
    stop_event = threading.Event()
    stop_event.set()
    service.parse_input(stop_event)
    assert service.queue.unfinished_tasks == 0
    assert len(list(get_waiting_room().glob("invalid.yaml_failed_*"))) == 1
    acquisition.assert_not_called()
    sensor_factory.assert_not_called()
    synchroniser_factory.assert_not_called()


@pytest.mark.parametrize(
    "error", ["KeyboardInterrupt('user interruption')", "SystemExit(2)"]
)
def test_generator_interrupt_propagates_and_queue_is_accounted_for(
    pipeline: PipelineFixture, tmp_path: Path, error: str
) -> None:
    service, params, acquisition, sensor_factory, synchroniser_factory = pipeline
    generator = tmp_path / "interrupted_sequence.py"
    generator.write_text(
        f"def generate_sequence(params):\n    raise {error}\n", encoding="utf-8"
    )
    instruction = get_waiting_room() / "interrupted.yaml"
    instruction.write_text(
        yaml.safe_dump({**params, "ps_path": str(generator)}), encoding="utf-8"
    )
    service.queue.put(str(instruction))
    stop_event = threading.Event()
    stop_event.set()
    with pytest.raises(
        KeyboardInterrupt if error.startswith("KeyboardInterrupt") else SystemExit
    ):
        service.parse_input(stop_event)
    assert service.queue.unfinished_tasks == 0
    assert len(list(get_waiting_room().glob("interrupted.yaml_failed_*"))) == 1
    acquisition.assert_not_called()
    sensor_factory.assert_not_called()
    synchroniser_factory.assert_not_called()


def test_service_shutdown_signals_stop_and_attempts_all_joins(
    service: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    observer, worker = Mock(), Mock()
    monkeypatch.setattr(service, "_get_observer", Mock(return_value=observer))
    thread_factory = Mock(return_value=worker)
    monkeypatch.setattr(service.threading, "Thread", thread_factory)
    monkeypatch.setattr(service, "sleep", Mock(side_effect=KeyboardInterrupt()))
    service.main()
    stop_event = thread_factory.call_args.kwargs["args"][0]
    assert stop_event.is_set()
    assert service.event_thread.is_set()
    observer.stop.assert_called_once_with()
    observer.join.assert_called_once_with()
    worker.join.assert_called_once_with()
    assert "QuPyt measurement service stopped" in caplog.text


def test_worker_death_remains_primary_if_service_shutdown_fails(
    service: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    observer, worker = Mock(), Mock()
    worker.is_alive.return_value = False
    observer.stop.side_effect = OSError("observer stop refused")
    monkeypatch.setattr(service, "_get_observer", Mock(return_value=observer))
    monkeypatch.setattr(service.threading, "Thread", Mock(return_value=worker))
    monkeypatch.setattr(service, "sleep", lambda _: None)
    with pytest.raises(
        RuntimeError, match="Measurement worker terminated unexpectedly"
    ):
        service.main()
    observer.join.assert_called_once_with()
    worker.join.assert_called_once_with()
    assert "Additional error during measurement service shutdown" in caplog.text
