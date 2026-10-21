# pylint: disable=logging-format-interpolation
# pylint: disable=logging-not-lazy
"""Control Konrad KD300/6000 power supplies remotely.

Configuration through measurement YAML files is currently unsupported.
"""

from __future__ import annotations

import logging

import serial


class PowerSupply:
    """Remote control for Konrad KD300/6000 power supplies connected by USB.

    Attributes
    ----------
    address : str
        USB address of the power supply.
    instance : serial.Serial or None
        Active serial connection, or None after successful cleanup.
    """

    def __init__(self, address: str) -> None:
        """Initialize the power supply and establish its serial connection.

        Parameters
        ----------
        address : str
            USB address of the power supply.

        Raises
        ------
        ValueError
            If serial connection parameters are out of range.
        serial.SerialException
            If the device is missing or cannot be configured.
        """
        self.address = address
        try:
            self.instance = serial.Serial(self.address, timeout=1)
            logging.info(
                f"Connected to power supply {address}".ljust(65, ".") + "[done]"
            )
        except ValueError:
            logging.exception(
                f"Connection to power supply {address} failed".ljust(65, ".")
                + "[failed]"
            )
            raise
        except serial.SerialException:
            logging.exception(
                f"Connection to power supply {address} failed".ljust(65, ".")
                + "[failed]"
            )
            raise

    def __repr__(self) -> str:
        return f"PowerSupply(address: {self.address})"

    def __str__(self) -> str:
        return f"PowerSupply (address: {self.address})"

    def on(self) -> None:
        """Turn on the power supply output."""
        try:
            self.instance.write(b"OUT1")
        except Exception:
            logging.exception(
                "Failed to turn ON power supply output at %s", self.address
            )
            raise
        logging.info("Turned ON power supply output at %s", self.address)

    def off(self) -> None:
        """Turn off the power supply output."""
        try:
            self.instance.write(b"OUT0")
        except Exception:
            logging.exception(
                "Failed to turn OFF power supply output at %s", self.address
            )
            raise
        logging.info("Turned OFF power supply output at %s", self.address)

    def close(self) -> None:
        """Close the serial connection; repeated cleanup is harmless."""
        if getattr(self, "instance", None) is None:
            return
        self.instance.close()
        self.instance = None
        logging.info("Closed power supply connection at %s", self.address)
