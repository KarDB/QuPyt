# pylint: disable=logging-format-interpolation
# pylint: disable=logging-not-lazy
import serial
import traceback
import logging


class power_supply:
    """remote controll for Konrad KD300/6000 series
    Connection Type USB. Provide easy access to most
    relevant functions"""

    def __init__(self, address: str) -> None:
        self.address = address
        try:
            self.instance = serial.Serial(self.address, timeout=1)
            logging.info(f'Connected to power supply {address}'
                         .ljust(65, '.')+'[done]')
        except Exception:
            logging.error(f'Connection to power supply {address} failed'
                          .ljust(65, '.')+'[failed]')
            traceback.print_exc()

    def on(self) -> None:
        self.instance.write(b"OUT1")
        logging.info('Turned ON power supply output'.ljust(65, '.')+'[done]')

    def off(self) -> None:
        self.instance.write(b"OUT0")
        logging.info('Turned OFF power supply output'.ljust(65, '.')+'[done]')

    def close(self) -> None:
        print("close not yet implemented")
