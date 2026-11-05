"""Use real handlers and mock signal sources to check reconfiguration and reuse."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from qupyt.hardware import signal_sources
from qupyt.hardware.device_handler import DeviceHandler, DynamicDeviceHandler


@pytest.fixture
def requested_device(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(signal_sources, "sleep", lambda _: None)
    return {
        "source": {
            "device_type": "Mock",
            "address": "mock:1",
            "config": {"amplitude": -5},
        }
    }


def test_reusing_static_source_applies_updated_configuration(
    requested_device: dict[str, Any]
) -> None:
    handler = DeviceHandler({})
    handler.update_devices(requested_device)
    source = handler.devices["source"]["device"]
    handler.set_all_params()
    updated = deepcopy(requested_device)
    updated["source"]["config"] = {"amplitude": -10, "frequency": 200}
    handler.update_devices(updated)
    assert handler.devices["source"]["device"] is source
    handler.set_all_params()
    assert source.configuration == {"amplitude": -10, "frequency": 200}
    assert updated == {
        "source": {
            "device_type": "Mock",
            "address": "mock:1",
            "config": {"amplitude": -10, "frequency": 200},
        }
    }


def test_address_change_replaces_device_and_releases_original(
    requested_device: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    handler = DeviceHandler({})
    handler.update_devices(requested_device)
    original = handler.devices["source"]["device"]
    close = Mock(wraps=original.close)
    monkeypatch.setattr(original, "close", close)
    updated = deepcopy(requested_device)
    updated["source"]["address"] = "mock:2"
    handler.update_devices(updated)
    close.assert_called_once_with()
    assert handler.devices["source"]["device"] is not original
    assert handler.devices["source"]["address"] == "mock:2"
    handler.update_devices(None)
    assert handler.devices == {}


def test_failed_close_retains_only_failed_device_and_can_be_retried(
    requested_device: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    requested_device["other"] = {
        "device_type": "Mock",
        "address": "mock:2",
        "config": {},
    }
    handler = DeviceHandler({})
    handler.update_devices(requested_device)
    source = handler.devices["source"]["device"]
    error = OSError("source busy")
    close = Mock(wraps=source.close, side_effect=[error, None])
    monkeypatch.setattr(source, "close", close)
    with pytest.raises(OSError) as caught:
        handler.update_devices({})
    assert caught.value is error
    assert list(handler.devices) == ["source"]
    handler.update_devices({})
    assert handler.devices == {}
    assert close.call_count == 2


def test_dynamic_reuse_removes_old_sweeps_and_resets_counter(
    requested_device: dict[str, Any]
) -> None:
    requested_device["source"]["config"] = {
        "frequency": [100, 200],
        "amplitude": [-5, -10],
    }
    original = deepcopy(requested_device)
    handler = DynamicDeviceHandler({}, number_dynamic_steps=3)
    handler.update_devices(requested_device)
    source = handler.devices["source"]["device"]
    np.testing.assert_array_equal(
        handler.devices["source"]["sweep_lists"]["frequency"]["channel_1"],
        [100, 150, 200],
    )
    handler.next_dynamic_step()
    handler.next_dynamic_step()
    assert source.configuration == {
        "frequency": [("channel_1", 150.0)],
        "amplitude": [("channel_1", -7.5)],
    }
    updated = deepcopy(requested_device)
    updated["source"]["config"] = {"frequency": [300, 600, 900]}
    handler.update_devices(updated)
    assert handler.devices["source"]["device"] is source
    assert handler.current_dynamic_step == 0
    assert list(handler.devices["source"]["sweep_lists"]) == ["frequency"]
    handler.next_dynamic_step()
    assert source.configuration == {"frequency": [("channel_1", 300)]}
    assert requested_device == original


def test_manual_sweep_with_wrong_length_is_rejected(
    requested_device: dict[str, Any]
) -> None:
    requested_device["source"]["config"] = {"frequency": [100, 200, 300]}
    handler = DynamicDeviceHandler({}, number_dynamic_steps=4)
    with pytest.raises(ValueError, match="number of dynamic_steps"):
        handler.update_devices(requested_device)
    assert handler.current_dynamic_step == 0
