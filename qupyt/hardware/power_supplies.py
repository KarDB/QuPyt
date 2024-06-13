# pylint: disable=logging-format-interpolation
# pylint: disable=logging-not-lazy
import traceback
import logging
import serial


class PowerSupply:
    """
    Remote control for Konrad KD300/6000 series power supplies.

    This class facilitates remote control of Konrad KD300/6000 series power supplies
    via USB connection. It provides easy access to the most relevant functions such
    as turning the power supply on and off.

    Attributes:
        address (str): The USB address of the power supply.
        instance (serial.Serial): The serial connection instance to the power supply.
    """

    def __init__(self, address: str) -> None:
        """
        Initializes the PowerSupply instance and establishes a serial connection.

        Args:
            address (str): The USB address of the power supply.

        Raises:
            ValueError: If connection parameters are out of range. This should
            not happen, as these values are set automatically.
            SerialException: If the device is not found or not configurable.
        """
        self.address = address
        try:
            self.instance = serial.Serial(self.address, timeout=1)
            logging.info(
                f"Connected to power supply {address}".ljust(65, ".") + "[done]"
            )
        except ValueError as exc:
            logging.exception(
                f"Connection to power supply {address} failed".ljust(65, ".")
                + "[failed]"
            )
            raise exc
        except serial.SerialException as exc:
            logging.exception(
                f"Connection to power supply {address} failed".ljust(65, ".")
                + "[failed]"
            )
            raise exc

    def __repr__(self) -> str:
        return f"PowerSupply(address: {self.address})"

    def __str__(self) -> str:
        return f"PowerSupply (address: {self.address})"

    def on(self) -> None:
        """
        Turns on the power supply output.

        This method sends a command to the power supply to turn on the output.
        """
        self.instance.write(b"OUT1")
        logging.info("Turned ON power supply output".ljust(65, ".") + "[done]")

    def off(self) -> None:
        """
        Turns off the power supply output.

        This method sends a command to the power supply to turn off the output.
        """
        self.instance.write(b"OUT0")
        logging.info("Turned OFF power supply output".ljust(65, ".") + "[done]")

    def close(self) -> None:
        """
        Closes the connection to the power supply.

        Note:
            This method is not yet implemented.
        """
        print("close not yet implemented")
