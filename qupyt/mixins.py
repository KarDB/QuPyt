"""
A shared mixin class to unify
how parameters are updated or set from
a configuration dictionary.
"""
from typing import Protocol, Callable, Any, Dict


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

    def _update_from_configuration(self: UpdateConfigurationProtocol,
                                   configuration: Dict[str, Any]) -> None:
        for attr, value in configuration.items():
            if attr in self.attribute_map:
                self.attribute_map[attr](value)
            else:
                raise ValueError(f"Unknown attribute: {attr}")


class PulseSequenceError(Exception):
    "PulseStreamer sequence definintion contains non valid input (amplitude, frequency,...)"
    pass
