"""Update device attributes and hardware settings from configuration dictionaries."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol, TypeAlias


# pylint: disable=too-few-public-methods
class UpdateConfigurationProtocol(Protocol):
    """Protocol for objects exposing a configuration attribute map.

    Attributes
    ----------
    attribute_map : dict[str, Callable[[Any], None]]
        Mapping from configuration keys to their setter functions.
    """

    attribute_map: dict[str, Callable[[Any], None]]


UpdateConfigurationType: TypeAlias = dict[str, Callable[[Any], None]]


# pylint: disable=too-few-public-methods
class ConfigurationMixin:
    """Mixin for applying configuration values through an attribute map."""

    attribute_map: UpdateConfigurationType

    def _update_from_configuration(
        self: UpdateConfigurationProtocol, configuration: dict[str, Any]
    ) -> None:
        for attr, value in configuration.items():
            try:
                setter = self.attribute_map[attr]
            except KeyError:
                logging.exception(
                    "Unknown configuration attribute %s for %s",
                    attr,
                    type(self).__name__,
                )
                raise KeyError(f"Unknown attribute: {attr}") from None
            logging.debug("Configuring %s.%s=%r", type(self).__name__, attr, value)
            try:
                setter(value)
            except Exception:
                logging.exception(
                    "Failed to configure %s.%s=%r", type(self).__name__, attr, value
                )
                raise
            logging.debug("Configured %s.%s", type(self).__name__, attr)


class PulseSequenceError(Exception):
    """Error raised for invalid pulse-sequence values."""


class ConfigurationError(Exception):
    """Error raised for an unsupported hardware configuration value."""

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
