"""Shared fixtures which construct real QuPyt objects and isolate external effects."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import threading
from pathlib import Path
from queue import Queue
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest

from qupyt.hardware.device_handler import DeviceHandler, DynamicDeviceHandler
from qupyt.hardware.sensors import MockCam
from qupyt.hardware.synchronisers import MockGenerator
from tests.support import MeasurementEnvironment

measurement_module = importlib.import_module("qupyt.measurement_logic.run_measurement")


@pytest.fixture
def measurement_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> MeasurementEnvironment:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(measurement_module, "sleep", lambda _: None)
    monkeypatch.setattr(measurement_module, "tqdm", lambda iterable, **_: iterable)
    caplog.set_level(logging.INFO)
    sensor = MockCam({"number_measurements": 2, "image_roi": [1]})
    acquisition = Mock(return_value=np.ones((2, 1)))
    monkeypatch.setattr(np.random, "poisson", acquisition)
    sensor_close = Mock(wraps=sensor.close)
    monkeypatch.setattr(sensor, "close", sensor_close)
    synchroniser = MockGenerator({}, {})
    synchroniser_close = Mock(wraps=synchroniser.close)
    monkeypatch.setattr(synchroniser, "close", synchroniser_close)
    params = {
        "experiment_type": "logging_test",
        "averages": 2,
        "dynamic_steps": 2,
        "pulse_sequence_steps": 2,
        "synchroniser": {"upload": False},
        "data": {"averaging_mode": "spread", "dynamic_steps": 2, "ps_steps": 2},
    }
    return MeasurementEnvironment(
        static=DeviceHandler({}),
        dynamic=DynamicDeviceHandler({}, number_dynamic_steps=2),
        sensor=sensor,
        acquisition=acquisition,
        synchroniser=synchroniser,
        params=params,
        directory=tmp_path,
        sensor_close=sensor_close,
        synchroniser_close=synchroniser_close,
    )


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # main configures logging and parses CLI arguments on import. Isolate those
    # effects while exercising its real worker and preparation functions.
    spec = importlib.util.spec_from_file_location(
        "qupyt_main_test", Path(__file__).resolve().parents[1] / "qupyt/main.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    with monkeypatch.context() as context:
        context.setattr(sys, "argv", ["qupyt"])
        context.setattr(logging, "basicConfig", lambda **_: None)
        spec.loader.exec_module(module)
    for handler in module.handlers:
        handler.close()
    setattr(module, "queue", Queue())
    setattr(module, "event_thread", threading.Event())
    return module
