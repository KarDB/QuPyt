# pylint: disable=logging-format-interpolation
# pylint: disable=logging-not-lazy
"""
Create all controlls for microwave sources.
"""
import logging
import traceback
from abc import ABC, abstractmethod
from time import sleep
import serial
from qupyt.hardware import visa_handler


class MW_Sources(ABC):
    @abstractmethod
    def set_frequency(self, freq: float, channel: int) -> None:
        pass

    @abstractmethod
    def set_amplitude(self, ampl: float, channel: int) -> None:
        pass


class MockSignalSource(MW_Sources):
    def __init__(self, address: str) -> None:
        self.address = address

    def set_frequency(self, freq: float, channel: int) -> None:
        sleep(0.1)
        logging.info(
            "MOCKING! -> set frequency channel {} to"
            .ljust(65, ".")
            .format(channel) + f"{freq}"
        )

    def set_amplitude(self, ampl: float, channel: int) -> None:
        sleep(0.1)
        logging.info(
            f"MOCKING! -> set amplitued channel {channel} to"
            .ljust(65, ".") + f"{ampl}"
        )

    def close(self) -> None:
        pass


class SignalSource(visa_handler.VisaObject, MW_Sources):
    def get_amplitude(self, channel: int) -> float:
        ampl = self.instance.query(self.command[f"GetAmpl{channel}"])
        self.opc_wait()
        return ampl

    def set_amplitude(self, ampl: float, channel: int) -> None:
        self.instance.write(
            self.command[f"SetAmpl{channel}"] + str(ampl)
        )
        self.opc_wait()
        logging.info(
            f"{self.s_type} set amplitude channel {channel} to"
            .ljust(65, ".") + f"{ampl}"
        )

    def get_frequency(self, channel: int) -> float:
        freq = self.instance.query(self.command[f"GetFreq{channel}"])
        self.opc_wait()
        return freq

    def set_frequency(self, freq: float, channel: int) -> None:
        self.instance.write(
            self.command[f"SetFreq{channel}"] + str(freq)
        )
        self.opc_wait()
        logging.info(
            f"{self.s_type} set frequency channel {channel} to"
            .ljust(65, ".") + f"{freq}"
        )


class WindFreak(MW_Sources):
    def __init__(self, address: str) -> None:
        self.address = address
        try:
            self.instance = serial.Serial(self.address, timeout=1)
            logging.info(
                f'Connected to WindFreak on {address}'.ljust(65, '.')+'[done]')
        except Exception:
            logging.error(f'Connection to WindFreak on {address} failed'
                          .ljust(65, '.')+'[failed]')
            traceback.print_exc()
        self.set_power_level(1)

    def set_amplitude(self, ampl: float, channel: int) -> None:
        # channel is just here to conform with the API
        _ = channel
        self.instance.write(f"a{ampl}".encode())  # min 0 , max 63
        logging.info(
            "Windfreak set amplitude to"
            .ljust(65, ".") + f"{ampl}"
        )

    def set_frequency(self, freq: float, channel: int) -> None:
        # channel is just here to conform with the API
        _ = channel
        freq = freq / 1.0e6  # convert to MHz
        self.instance.write(f"f{round(freq, 1)}".encode())
        logging.info(
            "Windfreak set frequency to [MHz]"
            .ljust(65, ".") + f"{freq}"
        )

    def get_firmware_version(self) -> str:
        self.instance.write("v".encode())
        return (
            self.instance.readline().decode().replace("\n", "")
        )  # remove termination character \n from all the responses

    def get_model_type(self) -> str:
        self.instance.write("+".encode())
        return self.instance.readline().decode().replace("\n", "")

    def set_power_level(self, power_level: int) -> None:
        # High - 1, Low - 0
        self.instance.write(f"h{power_level}".encode())
        logging.info(
            "Windfreak power level set to"
            .ljust(65, ".") + f"{power_level}"
        )

    def on(self) -> None:
        self.instance.write("o1".encode())

    def off(self) -> None:
        self.instance.write("o0".encode())

    def close(self) -> None:
        self.instance.close()

    def query(self, query):
        self.instance.write(query.encode())
        return self.instance.readline().decode().replace("\n", "")


class WindFreakHDM(MW_Sources):
    def __init__(self, address: str) -> None:
        self.address = address
        self.instance = serial.Serial(self.address, timeout=1)
        self.set_power_level(1)

    def set_amplitude(self, ampl: float, channel: int) -> None:
        # channel is just here to conform with the API
        _ = channel
        self.instance.write(f"W{ampl}".encode())  # min 0 , max 63
        logging.info(
            "Windfreak set amplitude to"
            .ljust(65, ".") + f"{ampl}"
        )

    def set_frequency(self, freq: float, channel: int) -> None:
        # channel is just here to conform with the API
        _ = channel
        freq = freq / 1.0e6  # convert to MHz
        self.instance.write(f"f{round(freq, 8)}".encode())
        logging.info(
            "Windfreak set frequency to [MHz]"
            .ljust(65, ".") + f"{freq}"
        )

    def get_firmware_version(self) -> str:
        self.instance.write("v".encode())
        return (
            self.instance.readline().decode().replace("\n", "")
        )  # remove termination character \n from all the responses

    def get_model_type(self) -> str:
        self.instance.write("+".encode())
        return self.instance.readline().decode().replace("\n", "")

    def set_power_level(self, power_level: int) -> None:
        # High - 1, Low - 0
        self.instance.write(f"h{power_level}".encode())
        logging.info(
            "Windfreak power level set to"
            .ljust(65, ".") + f"{power_level}"
        )

    def on(self) -> None:
        self.instance.write("o1".encode())
        logging.info('WindFreak output set'.ljust(65, '.') + '[ON]')

    def off(self) -> None:
        self.instance.write("o0".encode())
        logging.info('WindFreak output set'.ljust(65, '.') + '[OFF]')

    def close(self) -> None:
        self.instance.close()
        logging.info('WindFreak instance closed'.ljust(65, '.') + '[done]')

    def query(self, query):
        self.instance.write(query.encode())
        return self.instance.readline().decode().replace("\n", "")
