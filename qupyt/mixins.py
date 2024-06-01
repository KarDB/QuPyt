"""
A shared mixin class to unify
how parameters are updated or set from
a configuration dictionary.
"""

from typing import Protocol, Callable, Any, Dict
import logging


# pylint: disable=too-few-public-methods
class UpdateConfigurationProtocol(Protocol):
    """
    Protocoll defining structure of the attribute_map
    for the mixin class.
    """

    attribute_map: Dict[str, Callable[[Any], None]]


UpdateConfigurationType = Dict[str, Callable[[Any], None]]


# pylint: disable=too-few-public-methods
class ConfigurationMixin:
    """
    Mixin class to create a uniform update method
    for attributes an hardware values from a configuration
    dictionary.
    """

    def _update_from_configuration(
        self: UpdateConfigurationProtocol, configuration: Dict[str, Any]
    ) -> None:
        for attr, value in configuration.items():
            try:
                self.attribute_map[attr](value)
            except KeyError:
                logging.exception(f"Unknown attribute: {attr}")
                raise KeyError(f"Unknown attribute: {attr}")


class PulseSequenceError(Exception):
    "PulseStreamer sequence definintion contains non valid input (amplitude, frequency,...)"


class ConfigurationError(Exception):
    """Error type to be raised for hardware misconfigurations"""

    def __init__(self, config_param: Any, value: Any, valids: Any) -> None:
        self.config_param = config_param
        self.value = value
        self.valids = valids
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        return (
            f"Invalid configuration values!\n"
            f"It seems you tried to configure {self.config_param} as {self.value}\n"
            f"Try using one of the following configuration options instead:\n\n"
            " ===>   "
            f"{self.valids}\n\n"
        )
