# pylint: disable=logging-format-interpolation
# pylint: disable=logging-not-lazy
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
from typing import Dict, Any, List, Union, Tuple
import numpy as np
from qupyt.hardware.signal_sources import DeviceFactory


class DeviceHandler:
    def __init__(
        self,
        requested_devices: Dict[str, Any],
    ) -> None:
        self.requested_devices: Dict[str, Any] = {}
        self.devices: Dict[str, Any] = {}
        self.update_requested_device_dict(requested_devices)

    def update_devices(self, requested_devices: Dict[str, Any]) -> None:
        """
        Updates the device dicts by first closing all
        superfluous devices and subsequently opening
        newly requested devices.
        """
        self.requested_devices = requested_devices
        self.close_superfluous_devices()
        self.open_new_requested_devices()

    def close_superfluous_devices(self) -> None:
        """
        Close all devices not requested for the next measurement.
        Compare dict of requested and existing devices.
        Close and remove devices not requested.
        """
        requested_name_address_tuples = [
            (key, val["address"]) for key, val in self.requested_devices.items()
        ]
        for key, value in list(self.devices.items()):
            if (key, value["address"]) not in requested_name_address_tuples:
                self.devices[key]["device"].close()
                rem = self.devices.pop(key)
                logging.info(
                    f"Removed {rem} from active devices dict".ljust(
                        65, ".") + "[done]"
                )

    def open_new_requested_devices(self) -> None:
        """
        Open all devices requested for the next measurement that are
        not in the current active dict.
        Compare dict of requested and existing devices.
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
                logging.info(
                    f"Updated {repr(self.devices[key]['device'])} in active devices dict".ljust(
                        65, "."
                    )
                    + "[done]"
                )

    def set_all_params(self) -> None:
        """Set all values requested for static devices"""
        for value in self.devices.values():
            value["device"].set_values()

    def update_requested_device_dict(
        self,
        requested_devices: Dict[str, Any],
    ) -> None:
        """
        Creates a deep copy to not alter the original configuration.

        :param content: Full configuration dictionary as loaded from config YAML
        :type content: Dict[str, Any]
        """
        self.requested_devices = copy.deepcopy(requested_devices)


class DynamicDeviceHandler(DeviceHandler):
    def __init__(
        self, requested_devices: Dict[str, Any], number_dynamic_steps: int
    ) -> None:
        self.number_dynamic_steps = number_dynamic_steps
        self.current_dynamic_step = 0
        super().__init__(requested_devices)

    # TO OPEN DYNAMIC DEVICES, remove the config subdict and pass an emtpy dict.
    # create dynamic values and pass new config to device on every update.

    def open_new_requested_devices(self) -> None:
        """
        Open all devices requested for the next measurement that are
        not in the current active dict.
        Compare dict of requested and existing devices.
        """
        current_name_address_tuples = [
            (key, val["address"]) for key, val in self.devices.items()
        ]
        for key, value in list(self.requested_devices.items()):
            if (key, value["address"]) not in current_name_address_tuples:
                # Remove config dicts.
                # These will be updated for every new value.
                creation_dict = copy.deepcopy(value)
                creation_dict["sweep_config"] = creation_dict["config"]
                creation_dict["config"] = {}
                device = DeviceFactory.create_device(creation_dict)
                self.devices[key] = value
                self.devices[key]["device"] = device
                logging.info(
                    f"Added {repr(self.devices[key]['device'])} to active devices dict".ljust(
                        65, "."
                    )
                    + "[done]"
                )
            else:
                self.devices[key]["sweep_config"] = copy.deepcopy(
                    value["config"])
                logging.info(
                    f"Updated {repr(self.devices[key]['device'])} in active devices dict".ljust(
                        65, "."
                    )
                    + "[done]"
                )
        self._make_sweep_lists()

    def next_dynamic_step(self) -> None:
        """
        Set all values requested for dynamic devices.
        This is a function of the sweep value index.
        """
        for device in self.devices.values():
            current_config = {}
            for parameter, sweep_values in device["sweep_lists"]:
                current_config[parameter] = (
                    sweep_values['channel'],
                    sweep_values['sweep_values'][self.current_dynamic_step]
                )
            device['config'] = current_config
            device.set_values()
        self.current_dynamic_step += 1

    def _get_channel_and_sweeplist(self, value_list: Union[List[float], Tuple[str, List[float]]]) -> Tuple[str, List[float]]:
        if not isinstance(value_list[1], list):
            return 'channel_1', value_list
        return value_list[0], value_list[1]

    def _make_sweep_lists(self) -> None:
        """Contruct array / listof values to be sweeped"""
        for device in self.devices.values():
            for parameter, value_list in device["sweep_config"].values():
                channel, value_list = self._get_channel_and_sweeplist(
                    value_list)
                if all(isinstance(x, (int, float)) for x in value_list):
                    if len(value_list) == 2:
                        device["sweep_lists"][parameter]['sweep_values'] = np.linspace(
                            value_list[0], value_list[1], self.number_dynamic_steps
                        )
                        device["sweep_lists"][parameter]['channel'] = channel
                    else:
                        if len(value_list) != self.number_dynamic_steps:
                            raise ValueError(
                                "Trying to set manual sweep value list. Please make sure the number of dynamic_steps matches the length of the provided list"
                            )
                        device["sweep_lists"][parameter]['sweep_values'] = value_list
                        device["sweep_lists"][parameter]['channel'] = channel
                else:
                    raise ValueError(
                        "Currently only numeric values are allowed for dynamic devices"
                    )
