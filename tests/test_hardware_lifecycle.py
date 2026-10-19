"""Call actual hardware adapters, substituting only their external drivers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call

import nidaqmx
import numpy as np
import pytest
import pyvisa
import serial

from qupyt.hardware import (
    power_supplies,
    sensors,
    signal_sources,
    synchronisers,
    visa_handler,
)
from qupyt.mixins import ConfigurationError
from tests.support import MeasurementEnvironment


@pytest.fixture
def visa_connection(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    resource = Mock()
    manager = Mock()
    manager.open_resource.return_value = resource
    monkeypatch.setattr(pyvisa, "ResourceManager", Mock(return_value=manager))
    monkeypatch.setattr(visa_handler, "sleep", lambda _: None)
    return resource, manager


def test_daq_setup_acquisition_and_cleanup_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = Mock()
    task.read.return_value = [2.5, 7.5]
    monkeypatch.setattr(nidaqmx, "Task", Mock(return_value=task))
    sensor = sensors.DAQ({"number_measurements": 2, "apd_input": "Dev1/ai1"})
    sensor.open()
    assert task.ai_channels.add_ai_voltage_chan.call_args.args[:2] == ("Dev1/ai1", "")
    assert task.timing.cfg_samp_clk_timing.call_args.args[-1] == 2
    assert (
        task.triggers.start_trigger.cfg_dig_edge_start_trig.call_args.args[0] == "PFI0"
    )
    sync = Mock()
    events = Mock()
    events.attach_mock(task.start, "start")
    events.attach_mock(sync.trigger, "trigger")
    events.attach_mock(task.read, "read")
    events.attach_mock(task.stop, "stop")
    events.attach_mock(task.close, "close")
    np.testing.assert_array_equal(sensor.acquire_data(sync), [[2.5], [7.5]])
    sensor.close()
    assert events.mock_calls == [
        call.start(),
        call.trigger(),
        call.read(2, 10),
        call.stop(),
        call.close(),
    ]
    assert getattr(sensor, "daq_task") is None


def test_daq_successful_read_does_not_hide_stop_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    task = Mock()
    task.read.return_value = [2.5, 7.5]
    error = OSError("stop refused")
    task.stop.side_effect = error
    monkeypatch.setattr(nidaqmx, "Task", Mock(return_value=task))
    sensor = sensors.DAQ({"number_measurements": 2})
    sensor.open()
    with pytest.raises(OSError) as caught:
        sensor.acquire_data()
    assert caught.value is error
    task.start.assert_not_called()
    assert "failed during task stop" in caplog.text


def test_daq_close_can_be_retried_after_driver_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = Mock()
    error = OSError("close refused")
    task.close.side_effect = [error, None]
    monkeypatch.setattr(nidaqmx, "Task", Mock(return_value=task))
    sensor = sensors.DAQ({})
    sensor.open()
    with pytest.raises(OSError) as caught:
        sensor.close()
    assert caught.value is error
    assert sensor.daq_task is task
    sensor.close()
    sensor.close()
    assert getattr(sensor, "daq_task") is None
    assert task.close.call_count == 2


def test_measurement_reports_daq_open_error_without_secondary_close_error(
    measurement_environment: MeasurementEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qupyt.measurement_logic.run_measurement import run_measurement

    environment = measurement_environment
    error = OSError("DAQ not connected")
    monkeypatch.setattr(nidaqmx, "Task", Mock(side_effect=error))
    sensor = sensors.DAQ({"number_measurements": 2})
    assert (
        run_measurement(
            environment.static,
            environment.dynamic,
            sensor,
            environment.synchroniser,
            environment.params,
        )
        == "failed"
    )
    assert environment.params["measurement_error"]["stage"] == "sensor open"
    assert environment.params["measurement_error"]["message"] == "DAQ not connected"
    assert "cleanup_errors" not in environment.params
    assert getattr(sensor, "daq_task") is None
    assert not np.any(np.load(environment.params["filename"] + ".npy"))


@pytest.mark.parametrize(
    "value", ["1000000", ("channel_1", "1000000"), [("channel_1", "1000000")]]
)
def test_signal_source_configuration_coercion_calls_original_setter(
    visa_connection: tuple[Mock, Mock], value: Any
) -> None:
    resource, _ = visa_connection
    resource.query.return_value = "1"
    source = signal_sources.VisaSignalSource("test", "SMA", {})
    source.set_frequency(value)
    resource.write.assert_called_once_with("FREQ 1000000.0")
    resource.query.assert_called_once_with("*OPC?")


def test_signal_source_stops_multi_channel_update_at_first_driver_error(
    visa_connection: tuple[Mock, Mock]
) -> None:
    resource, _ = visa_connection
    error = OSError("first channel write failed")
    resource.write.side_effect = error
    source = signal_sources.VisaSignalSource("test", "Rigol", {})
    with pytest.raises(OSError) as caught:
        source.set_frequency([("channel_1", 100), ("channel_2", 200)])
    assert caught.value is error
    resource.write.assert_called_once_with("FREQ 100")
    resource.query.assert_not_called()


def test_power_supply_sends_output_commands_and_closes_serial_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = Mock()
    monkeypatch.setattr(serial, "Serial", Mock(return_value=connection))
    source = power_supplies.PowerSupply("test")
    source.on()
    source.off()
    assert connection.write.call_args_list == [call(b"OUT1"), call(b"OUT0")]
    source.close()
    source.close()
    connection.close.assert_called_once_with()


@pytest.mark.parametrize("operation", ["on", "off"])
def test_power_supply_write_error_is_logged_and_propagated(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, operation: str
) -> None:
    error = OSError("serial connection lost")
    connection = Mock()
    connection.write.side_effect = error
    monkeypatch.setattr(serial, "Serial", Mock(return_value=connection))
    source = power_supplies.PowerSupply("test")
    with pytest.raises(OSError) as caught:
        getattr(source, operation)()
    assert caught.value is error
    assert "serial connection lost" in caplog.text
    assert f"Failed to turn {operation.upper()} power supply output" in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        ValueError("invalid serial configuration"),
        serial.SerialException("device missing"),
    ],
)
def test_power_supply_connection_error_keeps_original_exception(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    monkeypatch.setattr(serial, "Serial", Mock(side_effect=error))
    with pytest.raises(type(error)) as caught:
        power_supplies.PowerSupply("test")
    assert caught.value is error


@pytest.mark.parametrize("response", ["1\n", "50", " 1.0 "])
def test_visa_opc_accepts_numeric_ready_responses(
    visa_connection: tuple[Mock, Mock], response: str
) -> None:
    resource, manager = visa_connection
    resource.query.return_value = response
    source = visa_handler.VisaObject("test", "SRS")
    source.opc_wait()
    resource.query.assert_called_once_with("*OPC?")
    source.close()
    source.close()
    resource.close.assert_called_once_with()
    manager.close.assert_not_called()


def test_visa_opc_retries_invalid_response_and_then_waits_for_ready(
    visa_connection: tuple[Mock, Mock], caplog: pytest.LogCaptureFixture
) -> None:
    resource, _ = visa_connection
    resource.query.side_effect = ["busy", "0", "1"]
    visa_handler.VisaObject("test", "SRS").opc_wait()
    assert resource.query.call_args_list == [call("*OPC?")] * 3
    assert "invalid OPC response 'busy'" in caplog.text


@pytest.mark.parametrize("response", ["busy", "0", ""])
def test_visa_opc_timeout_is_bounded(
    visa_connection: tuple[Mock, Mock], response: str
) -> None:
    resource, _ = visa_connection
    resource.query.return_value = response
    with pytest.raises(TimeoutError, match="Last response"):
        visa_handler.VisaObject("test", "SRS").opc_wait()
    assert resource.query.call_count == 3


def test_visa_opc_communication_error_is_not_retried(
    visa_connection: tuple[Mock, Mock]
) -> None:
    resource, _ = visa_connection
    error = OSError("connection lost")
    resource.query.side_effect = error
    with pytest.raises(OSError) as caught:
        visa_handler.VisaObject("test", "SRS").opc_wait()
    assert caught.value is error
    resource.query.assert_called_once_with("*OPC?")


def test_nested_source_initialization_keeps_driver_error(
    visa_connection: tuple[Mock, Mock]
) -> None:
    resource, manager = visa_connection
    error = OSError("device missing")
    manager.open_resource.side_effect = error
    with pytest.raises(OSError) as caught:
        signal_sources.DeviceFactory.create_device(
            {"device_type": "Rigol", "address": "test", "config": {}}
        )
    assert caught.value is error
    resource.close.assert_not_called()
    manager.close.assert_not_called()


@pytest.mark.parametrize(
    "configuration,commands",
    [
        ("off", ["BURS:STAT OFF"]),
        ("gate", ["BURS:STAT ON", "BURS:MODE GAT"]),
        (("channel_2", "gate"), ["SOUR2:BURS:STAT ON", "SOUR2:BURS:MODE GAT"]),
        (
            [("channel_1", "OFF"), ("channel_2", "GATE")],
            ["BURS:STAT OFF", "SOUR2:BURS:STAT ON", "SOUR2:BURS:MODE GAT"],
        ),
    ],
)
def test_rigol_gate_configuration_accepts_all_supported_shapes(
    visa_connection: tuple[Mock, Mock],
    configuration: signal_sources.ParameterInput,
    commands: list[str],
) -> None:
    resource, _ = visa_connection
    source = signal_sources.RigolSignalSource("test", "Rigol", {})
    resource.write.reset_mock()
    source._set_gate_mode(configuration)
    assert resource.write.call_args_list == [call(command) for command in commands]


def test_rigol_invalid_gate_mode_does_not_write_to_device(
    visa_connection: tuple[Mock, Mock]
) -> None:
    resource, _ = visa_connection
    source = signal_sources.RigolSignalSource("test", "Rigol", {})
    resource.write.reset_mock()
    with pytest.raises(ConfigurationError, match="Burst mode"):
        source._set_gate_mode("invalid")
    resource.write.assert_not_called()


@pytest.fixture
def spin_api(monkeypatch: pytest.MonkeyPatch) -> Mock:
    api = Mock()
    api.us = 1000
    api.PULSE_PROGRAM = 0
    api.Inst = SimpleNamespace(CONTINUE=0, BRANCH=6, STOP=1)
    api.pb_init.return_value = 0
    api.pb_close.return_value = 0
    api.pb_start_programming.return_value = 0
    api.pb_stop_programming.return_value = 0
    api.pb_get_error.return_value = "driver fault"
    monkeypatch.setattr(synchronisers, "spapi", api)
    return api


def test_pulseblaster_program_uses_returned_branch_address_and_units(
    spin_api: Mock,
) -> None:
    spin_api.pb_inst_pbonly.side_effect = [42, 43, 44, 45]
    sync = synchronisers.PulseBlaster({}, {})
    sync.open()
    sync.open()
    sync.program_pb([1, 2, 0], [1.0, 2.0, 3.0])
    assert spin_api.pb_inst_pbonly.call_args_list == [
        call(1, 0, 0, 1000),
        call(2, 0, 0, 2000),
        call(0, 6, 42, 3000),
        call(0, 1, 0, 5000),
    ]
    spin_api.pb_start_programming.assert_called_once_with(0)
    spin_api.pb_stop_programming.assert_called_once_with()
    spin_api.pb_init.assert_called_once_with()
    sync.close()
    sync.close()
    spin_api.pb_close.assert_called_once_with()


def test_pulseblaster_negative_instruction_remains_primary_if_stop_fails(
    spin_api: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    spin_api.pb_inst_pbonly.return_value = -99
    spin_api.pb_stop_programming.return_value = -2
    sync = synchronisers.PulseBlaster({}, {})
    with pytest.raises(RuntimeError, match="status=-99"):
        sync.program_pb([1, 0], [1.0, 1.0])
    spin_api.pb_stop_programming.assert_called_once_with()
    assert "Additional error during PulseBlaster programming stop" in caplog.text


def test_pulseblaster_initialization_error_does_not_close_unopened_board(
    spin_api: Mock,
) -> None:
    spin_api.pb_init.return_value = -1
    sync = synchronisers.PulseBlaster({}, {})
    with pytest.raises(RuntimeError, match="status=-1"):
        sync.open()
    sync.close()
    spin_api.pb_close.assert_not_called()


def test_pulseblaster_error_detail_failure_keeps_original_status(
    spin_api: Mock,
) -> None:
    spin_api.pb_get_error.side_effect = OSError("details unavailable")
    sync = synchronisers.PulseBlaster({}, {})
    with pytest.raises(RuntimeError, match="status=-99: error details unavailable"):
        sync.error_catcher(-99)


@pytest.mark.parametrize(
    "failure_at,stage",
    [
        ("open", "synchroniser open"),
        ("stop", "synchroniser stop"),
        ("run", "synchroniser run"),
    ],
)
def test_measurement_uses_real_pulsestreamer_failures(
    measurement_environment: MeasurementEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    failure_at: str,
    stage: str,
) -> None:
    from qupyt.measurement_logic.run_measurement import run_measurement

    environment = measurement_environment
    error = OSError(f"{failure_at} driver failure")
    driver = Mock()
    driver.isStreaming.return_value = False
    driver.forceFinal.side_effect = error if failure_at == "stop" else None
    driver.reset.side_effect = error if failure_at == "run" else None
    factory = (
        Mock(side_effect=error) if failure_at == "open" else Mock(return_value=driver)
    )
    monkeypatch.setattr(synchronisers, "PulseStreamer", factory)
    sync = synchronisers.PStreamer({"address": "192.0.2.1"}, {})
    assert (
        run_measurement(
            environment.static,
            environment.dynamic,
            environment.sensor,
            sync,
            environment.params,
        )
        == "failed"
    )
    environment.acquisition.assert_not_called()
    assert environment.params["measurement_error"]["stage"] == stage
    assert environment.params["measurement_error"]["message"] == str(error)
    assert "cleanup_errors" not in environment.params
    environment.sensor_close.assert_called_once_with()
