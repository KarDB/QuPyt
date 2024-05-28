"""
This file handles all aspect of the various signal sources in
active use. Newly requested devices are opened and added
to dict of used devices, while no longer needed ones are close
and deleted.
Edge cases like no requested devices as well as values to be set
or sweeped are handled here as well.
"""
import copy
import logging
from typing import Dict, Any, Tuple
import numpy as np
from qupyt.hardware.signal_sources import DeviceFactory


def close_superfluous_devices(devs: Dict[str, Any],
                              requested_devs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Close all devices not requested for the next measurement.
    Compare dict of requested and existing devices.
    Close and remove devices not requested.
    """
    requested_name_address_tuples = [
        (key, val["address"]) for key, val in requested_devs.items()
    ]
    for key, value in list(devs.items()):
        if (key, value["address"]) not in requested_name_address_tuples:
            devs[key]["device"].close()
            rem = devs.pop(key)
            logging.info(f"Removed {rem} from active devices dict"
                         .ljust(65, '.') + '[done]')
    return devs


def open_new_requested_devices(devs: Dict[str, Any],
                               requested_devs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Open all devices requested for the next measurement that are
    not in the current active dict.
    Compare dict of requested and existing devices.
    """
    current_name_address_tuples = [(key, val["address"])
                                   for key, val in devs.items()]
    for key, value in list(requested_devs.items()):
        if (key, value["address"]) not in current_name_address_tuples:
            device = DeviceFactory.create_device(value)
            devs[key] = value
            devs[key]["device"] = device
            logging.info(f"Added {repr(devs[key]['device'])} to active devices dict"
                         .ljust(65, '.') + '[done]')
        else:
            devs[key]['channels'] = value['channels']
            logging.info(f"Updated {repr(devs[key]['device'])} in active devices dict"
                         .ljust(65, '.') + '[done]')
    return devs


def set_all_static_params(devs: Dict[str, Any]) -> None:
    """Set all values requested for static devices"""
    for value in devs.values():
        value["device"].set_values()


def set_all_dynamic_params(dynamic_devices: Dict[str, Any],
                           index_value: Dict[str, Any]) -> None:
    """
    Set all values requested for dynamic devices.
    This is a function of the sweep value index.
    """
    for dynamic_device in dynamic_devices.values():
        # channel looks like channel_1
        # therefore channel[-1] would be 1
        for channel, channel_values in dynamic_device["channels"].items():
            dynamic_device["device"].set_frequency(
                float(channel_values["frequency_sweep_values"][index_value]),
                channel[-1],
            )
            dynamic_device["device"].set_amplitude(
                float(channel_values["amplitude_sweep_values"][index_value]),
                channel[-1],
            )


def make_sweep_lists(dynamic_devices: Dict[str, Any],
                     steps: int) -> Dict[str, Any]:
    """Contruct array / listof values to be sweeped"""
    for device_values in dynamic_devices.values():
        for channel_values in device_values["channels"].values():
            if channel_values["min_amplitude"] is not None:
                channel_values["amplitude_sweep_values"] = np.linspace(
                    float(channel_values["min_amplitude"]),
                    float(channel_values["max_amplitude"]),
                    steps,
                )
            else:
                x = np.linspace(0, 1, steps)
                channel_values["amplitude_sweep_values"] = eval(
                    channel_values["functional_amplitude"]
                )
            if channel_values["min_frequency"] is not None:
                channel_values["frequency_sweep_values"] = np.linspace(
                    float(channel_values["min_frequency"]),
                    float(channel_values["max_frequency"]),
                    steps,
                )
            else:
                x = np.linspace(0, 1, steps)
                channel_values["frequency_sweep_values"] = eval(
                    channel_values["functional_frequency"]
                )
    return dynamic_devices


def get_device_dicts(content: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract static and dynamic device request dicts from configuration
    YAML file.
    Creates a deep copy to not alter the original configuration.

    :param content: Full configuration dictionary as loaded from config YAML
    :type content: Dict[str, Any]
    """
    static_devices_requested = content.get("static_devices", {})
    dynamic_devices_requested = content.get("dynamic_devices", {})
    return copy.deepcopy(static_devices_requested), copy.deepcopy(dynamic_devices_requested)
