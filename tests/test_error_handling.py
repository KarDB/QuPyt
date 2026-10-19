"""Failures in recovery must not replace the original measurement error."""

from __future__ import annotations

import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock

import nidaqmx
import numpy as np
import pytest
import pyvisa
import serial
import yaml
from pypylon import pylon

from qupyt.hardware import sensors, signal_sources, synchronisers, visa_handler
from qupyt.hardware.device_handler import DeviceHandler
from qupyt.hardware.power_supplies import PowerSupply
from qupyt.measurement_logic.data_handling import Data
from qupyt.pulse_sequences import SequenceDesigner as designer
from qupyt.utils.decorators import loop_inputs
from qupyt.utils.errors import cleanup_all, cleanup_on_exit
from tests.support import SequenceDirectory


@pytest.mark.parametrize(
    "error", [ValueError("primary"), KeyboardInterrupt(), SystemExit(1)]
)
def test_cleanup_preserves_active_exception(
    error: BaseException, caplog: pytest.LogCaptureFixture
) -> None:
    close = Mock(side_effect=RuntimeError("secondary"))
    with pytest.raises(type(error)) as caught:
        with cleanup_on_exit(close, "test close"):
            raise error
    assert caught.value is error
    assert (
        "Additional error during test close; original failure retained" in caplog.text
    )


def test_cleanup_failure_after_success_is_not_suppressed() -> None:
    error = OSError("close failed")
    with pytest.raises(OSError) as caught:
        with cleanup_on_exit(Mock(side_effect=error), "test close"):
            pass
    assert caught.value is error


def test_cleanup_all_attempts_remaining_operations() -> None:
    error = RuntimeError("first close failed")
    later = Mock()
    with pytest.raises(RuntimeError) as caught:
        cleanup_all([("first", Mock(side_effect=error)), ("later", later)])
    assert caught.value is error
    later.assert_called_once()


def test_daq_task_creation_failure_has_no_uninitialized_cleanup_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    error = RuntimeError("no DAQ connected")
    monkeypatch.setattr(nidaqmx, "Task", Mock(side_effect=error))
    sensor = sensors.DAQ({})
    with pytest.raises(RuntimeError) as caught:
        sensor.open()
    assert caught.value is error
    assert getattr(sensor, "daq_task") is None
    assert "DAQ setup cleanup" not in caplog.text


def test_daq_configuration_failure_survives_task_close_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    error = ValueError("invalid DAQ channel")
    task = Mock()
    task.ai_channels.add_ai_voltage_chan.side_effect = error
    task.close.side_effect = RuntimeError("close failed")
    monkeypatch.setattr(nidaqmx, "Task", Mock(return_value=task))
    with pytest.raises(ValueError) as caught:
        sensors.DAQ({}).open()
    assert caught.value is error
    assert "Additional error during DAQ setup cleanup" in caplog.text


def test_daq_close_is_safe_before_open_and_after_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensor = sensors.DAQ({})
    sensor.close()
    task = Mock()
    monkeypatch.setattr(nidaqmx, "Task", Mock(return_value=task))
    sensor.open()
    sensor.close()
    sensor.close()
    task.close.assert_called_once()


def test_daq_reopen_releases_previous_task(monkeypatch: pytest.MonkeyPatch) -> None:
    first, second = Mock(), Mock()
    monkeypatch.setattr(nidaqmx, "Task", Mock(side_effect=[first, second]))
    sensor = sensors.DAQ({})
    sensor.open()
    sensor.open()
    first.close.assert_called_once()
    assert sensor.daq_task is second


@pytest.mark.parametrize(
    "sensor_type,dependency",
    [("PhantomS710", "EGenTL"), ("HeliC3", "heli"), ("DAQ", "nidaqmx")],
)
def test_unavailable_optional_driver_is_reported_by_name(
    monkeypatch: pytest.MonkeyPatch, sensor_type: str, dependency: str
) -> None:
    monkeypatch.setattr(sensors, dependency, None, raising=False)
    with pytest.raises(ImportError, match=dependency):
        sensors.SensorFactory.create_sensor(sensor_type, {})


@pytest.mark.parametrize("failure_at", ["start", "trigger", "read"])
def test_daq_acquisition_preserves_primary_when_stop_fails(
    failure_at: str, caplog: pytest.LogCaptureFixture
) -> None:
    error = RuntimeError(f"{failure_at} failed")
    sensor = sensors.DAQ({})
    sensor.NsampsPerDAQread = 2
    sensor.daq_task = Mock()
    sync = Mock()
    target = sync if failure_at == "trigger" else sensor.daq_task
    getattr(target, failure_at).side_effect = error
    sensor.daq_task.stop.side_effect = OSError("cannot stop")
    with pytest.raises(RuntimeError) as caught:
        sensor.acquire_data(sync)
    assert caught.value is error
    sensor.daq_task.stop.assert_called_once()
    assert "Additional error during DAQ acquisition stop" in caplog.text


def test_pulseblaster_programming_failure_exits_programming_mode(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    error = ValueError("bad pulse instruction")
    api = Mock()
    api.us = 1000
    api.pb_start_programming.return_value = 0
    api.pb_inst_pbonly.side_effect = error
    api.pb_stop_programming.return_value = -1
    api.pb_get_error.return_value = "cannot stop programming"
    monkeypatch.setattr(synchronisers, "spapi", api)
    sync = synchronisers.PulseBlaster({}, {})
    with pytest.raises(ValueError) as caught:
        sync.program_pb([1, 0], [1.0, 1.0])
    assert caught.value is error
    api.pb_stop_programming.assert_called_once()
    assert "Additional error during PulseBlaster programming stop" in caplog.text


@pytest.mark.parametrize(
    "bits,durations",
    [([], []), ([1], []), ([1], [0]), ([1], [-1]), ([1], [float("nan")])],
)
def test_pulseblaster_rejects_invalid_program_before_entering_hardware(
    monkeypatch: pytest.MonkeyPatch, bits: list[int], durations: list[float]
) -> None:
    api = Mock()
    monkeypatch.setattr(synchronisers, "spapi", api)
    sync = synchronisers.PulseBlaster({}, {})
    with pytest.raises(ValueError):
        sync.program_pb(bits, durations)
    api.pb_start_programming.assert_not_called()
    sync.close()
    api.pb_close.assert_not_called()


def test_pulsestreamer_connection_error_does_not_continue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = AssertionError("bad IP address")
    monkeypatch.setattr(synchronisers, "PulseStreamer", Mock(side_effect=error))
    sync = synchronisers.PStreamer({"address": "192.0.2.1"}, {})
    with pytest.raises(AssertionError) as caught:
        sync.open()
    assert caught.value is error


def test_pulsestreamer_missing_device_is_reported_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(synchronisers, "findPulseStreamers", Mock(return_value=[]))
    with pytest.raises(ConnectionError, match="No PulseStreamer"):
        synchronisers.PStreamer({"address": "None"}, {})


def test_pulsestreamer_stop_must_actually_stop() -> None:
    sync = synchronisers.PStreamer({"address": "192.0.2.1"}, {})
    sync.pulser = Mock()
    sync.pulser.isStreaming.return_value = True
    with pytest.raises(RuntimeError, match="still streaming"):
        sync.stop()


def test_pulsestreamer_stream_timeout_survives_stop_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sync = synchronisers.PStreamer({"address": "192.0.2.1"}, {})
    sync.pulser = Mock()
    sync.pulser.isStreaming.return_value = True
    sync.sequence = Mock(getDuration=Mock(return_value=1_000_000_000))
    monkeypatch.setattr(synchronisers, "monotonic", Mock(side_effect=[0, 12]))
    with pytest.raises(TimeoutError, match="did not finish"):
        sync.run()
    sync.pulser.forceFinal.assert_called_once()
    assert "Additional error during PulseStreamer run cleanup" in caplog.text


def test_awg_continuous_error_queue_is_bounded() -> None:
    sync = synchronisers.AWGenerator.__new__(synchronisers.AWGenerator)
    sync.instance = Mock(query=Mock(return_value='-1,"persistent error"'))
    with pytest.raises(RuntimeError, match="did not drain"):
        sync._raise_on_awg_errors("test upload")
    assert sync.instance.query.call_count == 100


@pytest.mark.parametrize(
    "operation",
    [
        "_limit_channels_to_available_outputs",
        "_raise_if_repeated_child_is_unverified_model",
    ],
)
def test_awg_identity_communication_failure_is_not_skipped(operation: str) -> None:
    sync = synchronisers.AWGenerator.__new__(synchronisers.AWGenerator)
    error = TimeoutError("identity query timed out")
    sync.instance = Mock(query=Mock(side_effect=error))
    with pytest.raises(TimeoutError) as caught:
        getattr(sync, operation)()
    assert caught.value is error


def test_visa_resource_manager_failure_preserves_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError("VISA backend missing")
    monkeypatch.setattr(pyvisa, "ResourceManager", Mock(side_effect=error))
    with pytest.raises(RuntimeError) as caught:
        visa_handler.VisaObject("resource", "SRS")
    assert caught.value is error


def test_visa_open_failure_never_closes_shared_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = Mock()
    error = OSError("device unavailable")
    manager.open_resource.side_effect = error
    monkeypatch.setattr(pyvisa, "ResourceManager", Mock(return_value=manager))
    with pytest.raises(OSError) as caught:
        visa_handler.VisaObject("resource", "SRS")
    assert caught.value is error
    manager.close.assert_not_called()
    manager.visalib._registry.clear.assert_not_called()


def test_visa_close_failure_propagates_and_can_be_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = Mock()
    resource = manager.open_resource.return_value
    monkeypatch.setattr(pyvisa, "ResourceManager", Mock(return_value=manager))
    device = visa_handler.VisaObject("resource", "SRS")
    error = OSError("connection close failed")
    resource.close.side_effect = error
    with pytest.raises(OSError) as caught:
        device.close()
    assert caught.value is error
    resource.close.side_effect = None
    device.close()
    device.close()
    assert resource.close.call_count == 2
    resource.visalib._registry.clear.assert_not_called()


def test_visa_timeout_configuration_failure_survives_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = ValueError("invalid resource timeout")

    class Resource:
        @property
        def timeout(self) -> int:
            return 0

        @timeout.setter
        def timeout(self, value: Any) -> None:
            raise error

        close = Mock(side_effect=OSError("resource close failed"))

    resource = Resource()
    manager = Mock(open_resource=Mock(return_value=resource))
    monkeypatch.setattr(pyvisa, "ResourceManager", Mock(return_value=manager))
    with pytest.raises(ValueError) as caught:
        visa_handler.VisaObject("resource", "SRS")
    assert caught.value is error
    resource.close.assert_called_once()
    manager.close.assert_not_called()


def test_basler_constructor_preserves_configuration_error_if_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cam = Mock()
    error = ValueError("invalid camera configuration")
    cam.BinningHorizontal.SetValue.side_effect = error
    cam.IsGrabbing.return_value = False
    cam.Close.side_effect = OSError("camera close failed")
    monkeypatch.setattr(pylon, "InstantCamera", Mock(return_value=cam))
    monkeypatch.setattr(pylon, "TlFactory", Mock())
    with pytest.raises(ValueError) as caught:
        sensors.BaslerCam({})
    assert caught.value is error
    cam.Close.assert_called_once()


def test_basler_failed_frame_conversion_releases_frame_and_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensor = sensors.BaslerCam.__new__(sensors.BaslerCam)
    sensor.number_measurements = 1
    sensor.roi_shape = [2, 2]
    sensor.cam = Mock()
    frame = Mock()
    frame.GrabSucceeded.return_value = True
    frame.Array = np.ones((3, 3))
    frame.Release.side_effect = OSError("release failed")
    sensor.cam.RetrieveResult.return_value = frame
    sensor.cam.StopGrabbing.side_effect = OSError("stop failed")
    with pytest.raises(ValueError, match="broadcast"):
        sensor.acquire_data()
    frame.Release.assert_called_once()
    sensor.cam.StopGrabbing.assert_called_once()


def test_harvester_close_attempts_reset_after_destroy_failure() -> None:
    sensor = sensors.GenICamHarvester.__new__(sensors.GenICamHarvester)
    error = RuntimeError("destroy failed")
    sensor.cam = Mock(destroy=Mock(side_effect=error))
    sensor.harvester = Mock(reset=Mock(side_effect=OSError("reset failed")))
    with pytest.raises(RuntimeError) as caught:
        sensor.close()
    assert caught.value is error
    sensor.harvester.reset.assert_called_once()


def test_harvester_fetch_failure_stops_acquisition() -> None:
    sensor = sensors.GenICamHarvester.__new__(sensors.GenICamHarvester)
    sensor.number_measurements = 1
    sensor.cam = Mock()
    sensor.cam.remote_device.node_map.Height.value = 1
    sensor.cam.remote_device.node_map.Width.value = 1
    error = TimeoutError("frame fetch timed out")
    sensor.cam.fetch.side_effect = error
    sensor.cam.stop.side_effect = OSError("camera stop failed")
    with pytest.raises(TimeoutError) as caught:
        sensor.acquire_data()
    assert caught.value is error
    sensor.cam.stop.assert_called_once()


def test_phantom_single_frame_does_not_fail_while_logging_framerate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensor = sensors.GenICamPhantom.__new__(sensors.GenICamPhantom)
    sensor.number_measurements = 1
    sensor.cam = Mock()
    sensor.cam.remote.get.return_value = 1
    monkeypatch.setattr(sensor, "_grab_frame_info", Mock(return_value=(0, 0, 0, 0)))
    monkeypatch.setattr(
        sensor,
        "_move_frame_from_pool",
        Mock(return_value=np.ones((4, 1), dtype=np.uint8)),
    )
    monkeypatch.setattr(
        sensors, "Buffer", Mock(return_value=MagicMock()), raising=False
    )
    result = sensor.acquire_data()
    np.testing.assert_array_equal(result, np.ones((1, 4, 1)))
    sensor.cam.stop.assert_called_once()


def test_helicam_timeout_does_not_process_invalid_camera_data() -> None:
    sensor = sensors.HeliCam.__new__(sensors.HeliCam)
    sensor.number_measurements = 2
    sensor.he_sys = Mock()
    sensor.he_sys.Acquire.return_value = -1
    with pytest.raises(TimeoutError, match="status=-1"):
        sensor.acquire_data()
    sensor.he_sys.ProcessCamData.assert_not_called()
    sensor.close()
    sensor.he_sys.Close.assert_not_called()


def test_helicam_buffer_flush_is_bounded_and_preserves_timeout_when_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensor = sensors.HeliCam.__new__(sensors.HeliCam)
    sensor._is_open = False
    sensor.settings = {}
    sensor.he_sys = Mock()
    sensor.he_sys.Acquire.return_value = 1
    sensor.he_sys.Close.side_effect = OSError("camera close failed")
    monkeypatch.setattr(sensors, "monotonic", Mock(side_effect=[0, 11]))
    with pytest.raises(TimeoutError, match="buffer flush"):
        sensor.open()
    sensor.he_sys.Close.assert_called_once()


def test_windfreak_serial_connection_failure_keeps_original_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = OSError("serial port unavailable")
    monkeypatch.setattr(serial, "Serial", Mock(side_effect=error))
    with pytest.raises(OSError) as caught:
        signal_sources.WindFreakSNV("COM1", {})
    assert caught.value is error


def test_windfreak_partial_initialization_closes_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = OSError("serial write failed")
    connection = Mock(
        write=Mock(side_effect=error),
        close=Mock(side_effect=RuntimeError("close failed")),
    )
    monkeypatch.setattr(serial, "Serial", Mock(return_value=connection))
    with pytest.raises(OSError) as caught:
        signal_sources.WindFreakSNV("COM1", {})
    assert caught.value is error
    connection.close.assert_called_once()


def test_power_supply_connection_is_closed_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = Mock()
    monkeypatch.setattr(serial, "Serial", Mock(return_value=connection))
    device = PowerSupply("COM1")
    device.close()
    device.close()
    connection.close.assert_called_once()


def test_device_handler_closes_remaining_devices_after_a_failure() -> None:
    handler = DeviceHandler({})
    error = OSError("first close failed")
    first, second = Mock(close=Mock(side_effect=error)), Mock()
    handler.devices = {
        "first": {"address": "one", "device": first},
        "second": {"address": "two", "device": second},
    }
    with pytest.raises(OSError) as caught:
        handler.update_devices({})
    assert caught.value is error
    second.close.assert_called_once()
    assert list(handler.devices) == ["first"]


def test_numeric_string_channel_value_does_not_mutate_tuple_or_hide_typeerror() -> None:
    received = []

    @loop_inputs
    def setter(self: object, value: Any) -> None:
        received.append(value)

    setter(None, [("channel_1", "3.5")])
    assert received == [("1", 3.5)]
    with pytest.raises(ValueError, match="pair"):
        setter(None, ["bad input"])


@pytest.fixture
def sequence_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SequenceDirectory:
    monkeypatch.setattr(designer, "get_seq_dir", lambda: tmp_path)
    sequence = {
        "total_duration": 1,
        "sequencing_order": ["block_0"],
        "sequencing_repeats": [1],
        "block_0": {},
    }
    first = tmp_path / "sequence_0.yaml"
    first.write_text(yaml.safe_dump(sequence), encoding="utf-8")
    return tmp_path, sequence, first


def test_failed_compilation_is_retried_instead_of_becoming_a_cache_hit(
    sequence_directory: SequenceDirectory, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, _, first = sequence_directory
    compiler = designer.PulseSequenceYaml({}, [1], 1e6, first)
    original = np.savez
    error = OSError("numeric output write failed")
    save = Mock(side_effect=error)
    monkeypatch.setattr(np, "savez", save)
    with pytest.raises(OSError) as caught:
        compiler.translate_yaml_to_numeric_instructions()
    assert caught.value is error
    assert not (directory / "sequence.npz.aux").exists()
    monkeypatch.setattr(np, "savez", original)
    compiler.translate_yaml_to_numeric_instructions()
    assert (directory / "sequence.npz").exists()
    assert compiler._sequence_didnt_change()


def test_alternating_yaml_inputs_cannot_reuse_another_numeric_sequence(
    sequence_directory: SequenceDirectory,
) -> None:
    directory, sequence, first = sequence_directory
    sequence["total_duration"] = 2
    second = directory / "sequence_1.yaml"
    second.write_text(yaml.safe_dump(sequence), encoding="utf-8")
    compiler_a = designer.PulseSequenceYaml({}, [1], 1e6, first)
    compiler_b = designer.PulseSequenceYaml({}, [1], 1e6, second)
    for compiler, length in ((compiler_a, 1), (compiler_b, 2), (compiler_a, 1)):
        compiler.translate_yaml_to_numeric_instructions()
        with np.load(directory / "sequence.npz", allow_pickle=True) as compiled:
            assert compiled["arr_0"].shape[-1] == length
    (directory / "sequence.npz").unlink()
    assert not compiler_a._sequence_didnt_change()


@pytest.mark.xfail(
    strict=True,
    raises=IndexError,
    reason="PulseBlaster fix deferred to a separate branch; see PULSEBLASTER_SIMULTANEOUS_EDGES_HANDOFF.md",
)
def test_pulseblaster_removes_multiple_zero_duration_segments() -> None:
    sequence = designer.PulseBlasterSequence.__new__(designer.PulseBlasterSequence)
    sequence.event_durations = [0, 0, 1, 0, 2]
    sequence.channel_bits = [1, 2, 3, 4, 5]
    sequence._pop_uncessary_entries()
    assert sequence.event_durations == [1, 2]
    assert sequence.channel_bits == [3, 5]


@pytest.mark.parametrize(
    "order,repeats",
    [([], []), (["block_0"], []), (["block_0"], [0]), (["missing"], [1])],
)
def test_invalid_sequence_metadata_fails_before_compilation(
    order: list[str], repeats: list[int]
) -> None:
    with pytest.raises(ValueError):
        designer.validate_sequence_instructions(
            {"sequencing_order": order, "sequencing_repeats": repeats, "block_0": {}}
        )


def test_pulse_outside_sequence_bounds_is_not_silently_truncated() -> None:
    sequence = designer.PulseSequence(1, 10, [1], samprate=1e6)
    with pytest.raises(ValueError, match="exceeds sequence bounds"):
        sequence.add_pulse(0, 9, 2)


def test_pulse_write_error_is_not_retried_as_a_different_index() -> None:
    sequence = designer.PulseSequence(2, 10, [1], samprate=1e6)
    error = RuntimeError("waveform write failed")

    class FailedArray:
        def __setitem__(self, key: object, value: Any) -> None:
            raise error

    setattr(sequence, "pulses", FailedArray())
    with pytest.raises(RuntimeError) as caught:
        sequence.add_pulse((0, 2), 0, 1)
    assert caught.value is error


def test_data_rejects_missing_frames_instead_of_broadcasting_them() -> None:
    data = Data({"averaging_mode": "spread", "dynamic_steps": 1})
    data.set_dims_from_sensor(
        sensors.MockCam({"number_measurements": 4, "image_roi": [2]})
    )
    data.create_array()
    with pytest.raises(ValueError, match="Sensor returned data shape"):
        data.update_data(np.ones((2, 2)), 0, 0, 0)
    assert not np.any(data.data)


def preparation_params() -> dict[str, Any]:
    return {
        "ps_path": "user_sequence.py",
        "pulse_sequence": {},
        "synchroniser": {
            "type": "MockSynchroniser",
            "config": {},
            "channel_mapping": {},
        },
        "sensor": {"type": "DAQ", "config": {}},
        "static_devices": {},
        "dynamic_devices": {},
        "dynamic_steps": 1,
    }


def test_pulse_generation_failure_never_constructs_or_closes_daq(
    service: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = ValueError("pulse generation failed")
    monkeypatch.setattr(service, "write_user_ps", Mock(side_effect=error))
    sensor_factory = Mock()
    sync_factory = Mock()
    monkeypatch.setattr(service.SensorFactory, "create_sensor", sensor_factory)
    monkeypatch.setattr(
        service.SynchroniserFactory, "create_synchroniser", sync_factory
    )
    instruction = tmp_path / "measurement.yaml"
    instruction.write_text(yaml.safe_dump(preparation_params()))
    with pytest.raises(ValueError) as caught:
        service._process_instruction(str(instruction), Mock(), Mock())
    assert caught.value is error
    sensor_factory.assert_not_called()
    sync_factory.assert_not_called()
    assert len(list(tmp_path.glob("measurement.yaml_failed_*"))) == 1
    assert not list(tmp_path.glob("measurement.yaml_running_*"))
    assert "DAQ setup cleanup" not in caplog.text


def test_sensor_creation_error_survives_synchroniser_cleanup_failure(
    service: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = ValueError("bad sensor configuration")
    sync = Mock(close=Mock(side_effect=OSError("sync close failed")))
    monkeypatch.setattr(service, "write_user_ps", Mock(return_value=None))
    monkeypatch.setattr(
        service.SynchroniserFactory, "create_synchroniser", Mock(return_value=sync)
    )
    monkeypatch.setattr(service.SensorFactory, "create_sensor", Mock(side_effect=error))
    with pytest.raises(ValueError) as caught:
        service._prepare_measurement(preparation_params(), Mock(), Mock())
    assert caught.value is error
    sync.close.assert_called_once()
    assert "Additional error during synchroniser preparation cleanup" in caplog.text


def test_failed_instruction_retention_cannot_mask_generation_error(
    service: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = RuntimeError("pulse generation failed")
    monkeypatch.setattr(service, "write_user_ps", Mock(side_effect=error))
    replace = service.os.replace
    calls = 0

    def failing_retention(source: str | Path, destination: str | Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("rename denied")
        replace(source, destination)

    monkeypatch.setattr(service.os, "replace", failing_retention)
    instruction = tmp_path / "measurement.yaml"
    instruction.write_text(yaml.safe_dump(preparation_params()))
    with pytest.raises(RuntimeError) as caught:
        service._process_instruction(str(instruction), Mock(), Mock())
    assert caught.value is error
    assert "Additional error during failed instruction retention" in caplog.text


def test_worker_continues_after_failed_instruction_and_accounts_for_queue(
    service: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    first.write_text(yaml.safe_dump(preparation_params()))
    second.write_text(yaml.safe_dump(preparation_params()))
    monkeypatch.setattr(
        service, "write_user_ps", Mock(side_effect=[ValueError("invalid pulse"), None])
    )
    monkeypatch.setattr(
        service.SynchroniserFactory, "create_synchroniser", Mock(return_value=Mock())
    )
    monkeypatch.setattr(
        service.SensorFactory, "create_sensor", Mock(return_value=Mock())
    )
    measurement = Mock(return_value="success")
    monkeypatch.setattr(service, "run_measurement", measurement)
    service.queue.put(str(first))
    service.queue.put(str(second))
    service.queue.put(str(second))  # duplicate event, now pointing to a removed file
    stopped = threading.Event()
    stopped.set()
    service.parse_input(stopped)
    measurement.assert_called_once()
    assert service.queue.unfinished_tasks == 0
    assert len(list(tmp_path.glob("first.yaml_failed_*"))) == 1
    assert not second.exists()


def test_invalid_instruction_mapping_is_retained(
    service: ModuleType, tmp_path: Path
) -> None:
    instruction = tmp_path / "invalid.yaml"
    instruction.write_text("- this is a list\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        service._process_instruction(str(instruction), Mock(), Mock())
    assert len(list(tmp_path.glob("invalid.yaml_failed_*"))) == 1


def test_idle_cleanup_failure_does_not_skip_other_handler_or_shutdown(
    service: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    static = Mock(update_devices=Mock(side_effect=OSError("close failed")))
    dynamic = Mock()
    monkeypatch.setattr(service, "DeviceHandler", Mock(return_value=static))
    monkeypatch.setattr(service, "DynamicDeviceHandler", Mock(return_value=dynamic))
    stopped = threading.Event()
    monkeypatch.setattr(
        service, "event_thread", Mock(wait=Mock(side_effect=lambda **_: stopped.set()))
    )
    service.parse_input(stopped)
    assert dynamic.update_devices.call_count == 2


def test_service_start_failure_survives_shutdown_failure(
    service: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = RuntimeError("observer could not start")
    observer = Mock(
        start=Mock(side_effect=error),
        stop=Mock(side_effect=OSError("observer stop failed")),
        ident=None,
    )
    worker = Mock(ident=None)
    monkeypatch.setattr(service, "_get_observer", Mock(return_value=observer))
    monkeypatch.setattr(
        service,
        "threading",
        SimpleNamespace(Event=threading.Event, Thread=Mock(return_value=worker)),
    )
    with pytest.raises(RuntimeError) as caught:
        service.main()
    assert caught.value is error
    observer.stop.assert_called_once()


@pytest.mark.parametrize(
    "operating_system,event_name,should_queue",
    [
        ("Windows", "FileModifiedEvent", True),
        ("Windows", "FileClosedEvent", False),
        ("Windows", "FileCreatedEvent", False),
        ("Windows", "FileMovedEvent", False),
        ("Linux", "FileModifiedEvent", False),
        ("Linux", "FileClosedEvent", True),
        ("Linux", "FileCreatedEvent", False),
        ("Linux", "FileMovedEvent", False),
        ("Darwin", "FileModifiedEvent", True),
        ("Darwin", "FileClosedEvent", False),
        ("Darwin", "FileCreatedEvent", True),
        ("Darwin", "FileMovedEvent", True),
    ],
)
def test_instruction_event_dispatch_uses_only_the_platform_handlers(
    service: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    operating_system: str,
    event_name: str,
    should_queue: bool,
) -> None:
    monkeypatch.setattr(service.platform, "system", lambda: operating_system)
    monkeypatch.setattr(service, "sleep", lambda _: None)
    handler = service.WaitingRoomEventHandler()
    event_type = getattr(service, event_name)
    event = (
        event_type("temporary.txt", "measurement.yaml")
        if event_name == "FileMovedEvent"
        else event_type("measurement.yaml")
    )
    handler.dispatch(event)
    if should_queue:
        assert service.queue.get_nowait() == "measurement.yaml"
    assert service.queue.empty()


def test_moving_an_instruction_to_running_does_not_queue_the_recovery_file(
    service: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service, "sleep", lambda _: None)
    handler = service.WaitingRoomEventHandler()
    handler._on_moved(
        SimpleNamespace(
            src_path="measurement.yaml", dest_path="measurement.yaml_running_1"
        )
    )
    assert service.queue.empty()
    handler._on_moved(
        SimpleNamespace(src_path="temporary.txt", dest_path="measurement.yaml")
    )
    assert service.queue.get_nowait() == "measurement.yaml"
