# pylint: disable=logging-format-interpolation
# pylint: disable=logging-not-lazy
"""Manage signal-source lifetimes, configuration, and measurement sweeps.

Open newly requested devices, close unused devices, and apply static or dynamic
output values. Empty device requests release the active devices.
"""

from __future__ import annotations

import copy
import logging
from functools import partial
from typing import Any, TypeAlias, cast

import numpy as np
from pydantic import validate_call

from qupyt.hardware.signal_sources import DeviceFactory
from qupyt.utils.errors import cleanup_all

DynamicParameterInput: TypeAlias = (
    list[float | int]
    | tuple[str, list[float | int]]
    | list[tuple[str, list[float | int]]]
)


class DeviceHandler:
    """A class to handle the management of devices for measurements.

    This class provides methods to update the list of requested devices,
    close superfluous devices, open new requested devices, and set parameters
    for all active devices.

    From the point of view of this class, all devices are static for the
    duration of one measurement.
    """

    def __init__(self, requested_devices: dict[str, Any]) -> None:
        """Initialize the handler with the requested device configurations.

        Parameters
        ----------
        requested_devices : dict[str, Any]
            Initial device configurations, keyed by the device's logical name.
        """
        self.requested_devices: dict[str, Any] = {}
        self.devices: dict[str, Any] = {}
        self.update_requested_device_dict(requested_devices)

    def update_devices(self, requested_devices: dict[str, Any] | None) -> None:
        """Close unused devices and open or reconfigure the requested devices.

        Parameters
        ----------
        requested_devices : dict[str, Any] or None
            Updated device configurations. None requests no active devices.
        """
        if requested_devices is None:
            self.requested_devices = {}
        else:
            self.requested_devices = copy.deepcopy(requested_devices)
        logging.debug(
            "Updating %s: requested_devices=%s",
            type(self).__name__,
            list(self.requested_devices),
        )
        self.close_superfluous_devices()
        self.open_new_requested_devices()

    def close_superfluous_devices(self) -> None:
        """Close all devices not requested for the next measurement.

        This method compares the dictionaries of requested and existing devices,
        and closes and removes devices that are not requested.
        """
        requested_name_address_tuples = [
            (key, val["address"]) for key, val in self.requested_devices.items()
        ]

        def close_device(key: str) -> None:
            value = self.devices[key]
            logging.info("Closing unused device %s at %s", key, value["address"])
            value["device"].close()
            self.devices.pop(key)
            logging.info("Removed device %s at %s", key, value["address"])

        cleanup_all(
            (f"device {key} close at {value['address']}", partial(close_device, key))
            for key, value in list(self.devices.items())
            if (key, value["address"]) not in requested_name_address_tuples
        )

    def open_new_requested_devices(self) -> None:
        """Open newly requested devices and reconfigure existing devices.

        Compare each logical name and address with the active device dictionary.
        """
        current_name_address_tuples = [
            (key, val["address"]) for key, val in self.devices.items()
        ]
        for key, value in list(self.requested_devices.items()):
            if (key, value["address"]) not in current_name_address_tuples:
                device = DeviceFactory.create_device(value)
                self.devices[key] = value
                self.devices[key]["device"] = device
                logging.info(
                    f"Added {repr(self.devices[key]['device'])} to active devices dict".ljust(
                        65, "."
                    )
                    + "[done]"
                )
            else:
                self.devices[key]["config"] = value["config"]
                self.devices[key]["device"].update_configuration(value["config"])
                logging.info(
                    f"Updated {repr(self.devices[key]['device'])} in active devices dict".ljust(
                        65, "."
                    )
                    + "[done]"
                )

    def set_all_params(self) -> None:
        """Set all values requested for static devices.

        This method sets the parameters for all active devices as per the
        requested configuration.
        """
        for name, value in self.devices.items():
            logging.debug(
                "Applying static parameters for device %s at %s", name, value["address"]
            )
            try:
                value["device"].set_values()
            except Exception:
                logging.exception(
                    "Failed to apply static parameters for device %s at %s",
                    name,
                    value["address"],
                )
                raise
        logging.info(
            "Static device configuration completed: devices=%s", list(self.devices)
        )

    def update_requested_device_dict(self, requested_devices: dict[str, Any]) -> None:
        """Copy the requested configurations without modifying the caller's dictionary.

        Parameters
        ----------
        requested_devices : dict[str, Any]
            Full device configuration dictionary loaded from the measurement YAML.
        """
        self.requested_devices = copy.deepcopy(requested_devices)


class DynamicDeviceHandler(DeviceHandler):
    """Manage device configurations that change between measurement steps.

    Extend DeviceHandler by applying a new set of otherwise static values at each
    dynamic step.
    """

    def __init__(
        self, requested_devices: dict[str, Any], number_dynamic_steps: int = 1
    ) -> None:
        """Initialize the requested devices and number of dynamic steps.

        Parameters
        ----------
        requested_devices : dict[str, Any]
            Initial configurations keyed by logical device name.
        number_dynamic_steps : int, optional
            Number of configuration steps, defaulting to 1.
        """
        self.number_dynamic_steps = number_dynamic_steps
        self.current_dynamic_step = 0
        super().__init__(requested_devices)

    def open_new_requested_devices(self) -> None:
        """Open or reconfigure devices and prepare their dynamic sweep values."""
        current_name_address_tuples = [
            (key, val["address"]) for key, val in self.devices.items()
        ]
        for key, value in list(self.requested_devices.items()):
            if (key, value["address"]) not in current_name_address_tuples:
                # Remove config dicts.
                # These will be updated for every new value.
                creation_dict = copy.deepcopy(value)
                creation_dict["config"] = {}
                device = DeviceFactory.create_device(creation_dict)
                self.devices[key] = value
                self.devices[key]["device"] = device
                self.devices[key]["sweep_config"] = copy.deepcopy(value["config"])
                logging.info(
                    f"Added {repr(self.devices[key]['device'])} to active devices dict".ljust(
                        65, "."
                    )
                    + "[done]"
                )
            else:
                self.devices[key]["sweep_config"] = copy.deepcopy(value["config"])
                logging.info(
                    f"Updated {repr(self.devices[key]['device'])} in active devices dict".ljust(
                        65, "."
                    )
                    + "[done]"
                )
        self._reset_step_counter()
        self._make_sweep_lists()

    def next_dynamic_step(self) -> None:
        """Apply the current sweep values and advance the dynamic step counter.

        Update each device's configuration to the current step and apply its values.
        """
        logging.info(
            "Applying dynamic step %s/%s: devices=%s",
            self.current_dynamic_step + 1,
            self.number_dynamic_steps,
            list(self.devices),
        )
        for name, device in self.devices.items():
            current_config: dict[str, list[tuple[str, float | int]]] = {}
            for parameter, channel_config in device["sweep_lists"].items():
                current_config[parameter] = []
                for channel, sweep_values in channel_config.items():
                    current_config[parameter].append(
                        (channel, sweep_values[self.current_dynamic_step])
                    )
            logging.debug(
                "Applying dynamic step %s/%s for device %s: %s",
                self.current_dynamic_step + 1,
                self.number_dynamic_steps,
                name,
                current_config,
            )
            try:
                device["device"].update_configuration(current_config)
                device["device"].set_values()
            except Exception:
                logging.exception(
                    "Failed to apply dynamic step %s/%s for device %s",
                    self.current_dynamic_step + 1,
                    self.number_dynamic_steps,
                    name,
                )
                raise
        self.current_dynamic_step += 1

    def _reset_step_counter(self) -> None:
        """Reset the dynamic step counter to zero."""
        self.current_dynamic_step = 0
        logging.debug("Reset dynamic device step counter")

    def _make_sweep_lists(self) -> None:
        """Construct arrays or lists of values to be swept for each dynamic device.

        This method prepares the sweep values for each device parameter based on
        the specified configuration and number of dynamic steps.
        """
        for name, device in self.devices.items():
            device["sweep_lists"] = {}
            for parameter, requested_values in device["sweep_config"].items():
                channel_values = self._coerce_input_shape_dynamic(requested_values)
                device.setdefault("sweep_lists", {}).setdefault(parameter, {})
                for channel, value_range in channel_values:
                    if len(value_range) == 2:
                        device["sweep_lists"][parameter][channel] = np.linspace(
                            value_range[0], value_range[1], self.number_dynamic_steps
                        )
                    else:
                        if len(value_range) != self.number_dynamic_steps:
                            logging.error(
                                "Invalid sweep for device %s, parameter=%s, channel=%s: values=%s, expected_steps=%s",
                                name,
                                parameter,
                                channel,
                                len(value_range),
                                self.number_dynamic_steps,
                            )
                            raise ValueError(
                                "Trying to set manual sweep value list. Please make sure the number of dynamic_steps matches the length of the provided list"
                            )
                        device["sweep_lists"][parameter][channel] = value_range
        if self.devices:
            logging.info(
                "Prepared dynamic sweeps: devices=%s, steps=%s",
                list(self.devices),
                self.number_dynamic_steps,
            )

    @validate_call
    def _coerce_input_shape_dynamic(
        self, arg: DynamicParameterInput
    ) -> list[tuple[str, list[float | int]]]:
        """Normalize dynamic parameter values to a list of channel/value pairs.

        Parameters
        ----------
        arg : DynamicParameterInput
            A value list, a channel/value-list tuple, or a list of such tuples.

        Returns
        -------
        list[tuple[str, list[float or int]]]
            Channel/value-list pairs. A bare value list uses channel_1.

        Raises
        ------
        ValueError
            If the input cannot be normalized.

        Notes
        -----
        Pydantic validates the input types before this method is called.
        """
        if isinstance(arg, list) and all(isinstance(item, tuple) for item in arg):
            return cast(list[tuple[str, list[float | int]]], arg)
        if isinstance(arg, tuple):
            return [arg]
        if isinstance(arg, list):
            return [("channel_1", cast(list[float | int], arg))]
        raise ValueError("Dynamic parameter coercion failed")
