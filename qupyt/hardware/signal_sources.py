# pylint: disable=logging-format-interpolation
# pylint: disable=logging-not-lazy
"""
Create all controlls for microwave sources.
"""
import logging
import traceback
from abc import ABC, abstractmethod
from time import sleep
from typing import Dict, Any, Union, Tuple, List
import serial
from windfreak import SynthHD
from pydantic import validate_call
from qupyt.hardware import visa_handler
from qupyt.mixins import UpdateConfigurationType, ConfigurationMixin, ConfigurationError
from qupyt.utils.decorators import coerce_device_config_shape, loop_inputs

ParameterInput = Union[
    Union[float, int, str],
    Tuple[str, Union[float, int, str]],
    List[Tuple[str, Union[float, int, str]]],
]

# pylint: disable=too-few-public-methods


class DeviceFactory:
    """
    Device Factory responsible for creating and returning an instance of the
    requested device.

    Any part of an experimental setup, that is not a sensor or synchronizer
    qualifies as a device. They could be signal sources, motors, power suplies
    or lasers and so fourth.
    Devices may in general be used in a static way, setting values at the
    begining of a measurement and not changing thereafter.
    Alternatively, devices my be dynamic and their values updated multiple
    times over the course of a meausrement.

    Because of the wide variety of devices, there is not standard interface.
    Instead, each device has an attribute map defining which parameters
    may be set. Furthermore, devices of a similar nature, such as
    signal sources, have a defined set of parameters the need to accept
    and therefore setter methods to apply them.
    In those cases, the adherence is enforced by inheriting from an
    appropriate abstract class.

    Methods:
        - :meth:`create_device`: This method creates the appropriate device
          instance and configures it. **Note for non programmers**: create_device
          is a static method. This means you don't have to create a class
          instance to call it.

    Example:
        >>> device = DeviceFactory.create_device('EoSense1.1CXP', {'number_measurements_referenced': 10})
    """

    @staticmethod
    def create_device(device_info: Dict[str, Any]):
        """
        :param device_info: Full device configuration dictionary.
        :type sensor_type: Dict[str, Any]
        :return: Instance of the requested device.
        :rtype:
        :raises ConfigurationError:
        """
        known_devices = [
            "WindFreak",
            "WindFreakHDM",
            "Mock",
            "SRS",
            "SMB",
            "Rigol",
            "TekAWG",
            "TekAFG",
            "WindFreakSNV",
            "WindFreakSHDMini"
        ]
        if device_info["device_type"] not in known_devices:
            raise ConfigurationError(
                "the device type", device_info["device_type"], known_devices
            )
        try:
            if device_info["device_type"] == "WindFreakSNV":
                return WindFreakSNV(device_info["address"], device_info["config"])
            if device_info["device_type"] == "WindFreakHDM":
                return WindFreakHDM(device_info["address"], device_info["config"])
            if device_info["device_type"] == "WindFreak":
                return WindFreakOfficial(device_info["address"], device_info["config"])
            if device_info["device_type"] == "WindFreakSHDMini":
                return WindFreakSHDMini(device_info["address"], device_info["config"])
            if device_info["device_type"] == "Mock":
                return MockSignalSource(device_info["address"], device_info["config"])
            if device_info["device_type"] == "SMB":
                return SMBVisaSignalSource(
                    device_info["address"],
                    device_info["device_type"],
                    device_info["config"],
                )
            if device_info["device_type"] == "Rigol":
                return RigolSignalSource(
                    device_info["address"],
                    device_info["device_type"],
                    device_info["config"],
                )
            return VisaSignalSource(
                device_info["address"],
                device_info["device_type"],
                device_info["config"],
            )

        except Exception as exc:
            logging.exception(
                "Could not open desired camera".ljust(65, ".") + "[failed]"
            )
            traceback.print_exc()
            raise exc


class SignalSource(ABC, ConfigurationMixin):
    attribute_map: UpdateConfigurationType

    def __init__(self, configuration: Dict[str, Any]) -> None:
        # pylint: disable=unused-argument
        # configuration is not used in the ABC, however
        # all child classes must take it as input.
        self.configuration = configuration
        self.attribute_map = {
            "frequency": self.set_frequency,
            "amplitude": self.set_amplitude,
        }

    @abstractmethod
    def set_frequency(self, freq: ParameterInput) -> None:
        """
        Force all Signal Sources to implement a way to
        set its output frequency.
        """

    @abstractmethod
    def set_amplitude(self, ampl: ParameterInput) -> None:
        """
        Force all Signal Sources to implement a way to
        set its output amplitude.
        Not all devices use the same units. However, wherever practically
        possible amplitudes will be interpreted in units of dBm.
        """

    def set_values(self) -> None:
        if self.configuration is not None:
            self._update_from_configuration(self.configuration)

    def update_configuration(self, config: Dict[str, Any]) -> None:
        setattr(self, "configuration", config)


class MockSignalSource(SignalSource):
    def __init__(self, address: str, configuration: Dict[str, Any]) -> None:
        self.address = address
        super().__init__(configuration)

    def __repr__(self) -> str:
        return f"MockSignalSource(address: {self.address})"

    def __str__(self) -> str:
        return f"Signal source of type MockSignalSource(address: {self.address})"

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_frequency(self, freq: ParameterInput) -> None:
        channel, freq = freq
        sleep(0.1)
        logging.info(
            f"MOCKING! -> set frequency channel {channel} to".ljust(65, ".") + f"{freq}"
        )

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_amplitude(self, ampl: ParameterInput) -> None:
        channel, ampl = ampl
        sleep(0.1)
        logging.info(
            f"MOCKING! -> set amplitued channel {channel} to".ljust(65, ".") + f"{ampl}"
        )

    def close(self) -> None:
        pass


class VisaSignalSource(visa_handler.VisaObject, SignalSource):
    """
    SignaSource implementation for devices that implement
    the VISA protocol
    """

    def __init__(
        self, address: str, device_type: str, configuration: Dict[str, Any]
    ) -> None:
        self.address = address
        visa_handler.VisaObject.__init__(self, address, device_type)
        SignalSource.__init__(self, configuration)

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_amplitude(self, ampl: ParameterInput) -> None:
        channel, ampl = ampl
        self.instance.write(self.command[f"SetAmpl{channel}"] + str(ampl))
        self.opc_wait()
        logging.info(
            f"{self.s_type} set amplitude channel {channel} to".ljust(65, ".")
            + f"{ampl}"
        )

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_frequency(self, freq: ParameterInput) -> None:
        channel, freq = freq
        self.instance.write(self.command[f"SetFreq{channel}"] + str(freq))
        self.opc_wait()
        logging.info(
            f"{self.s_type} set frequency channel {channel} to".ljust(65, ".")
            + f"{freq}"
        )


class RigolSignalSource(VisaSignalSource):
    """Special class for Rigol DG1022 to enable gating"""

    def __init__(
        self, address: str, device_type: str, configuration: Dict[str, Any]
    ) -> None:
        super().__init__(address, device_type, configuration)
        self.attribute_map["gating"] = self._set_gate_mode

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def _set_gate_mode(self, mode):
        valids = ["off", "gate"]
        channel, mode = mode
        if mode.lower() not in valids:
            raise ConfigurationError("Burst mode", mode, valids)
        if mode.lower() == "off":
            self.instance.write(self.command[f"SetBurstState{channel}"] + "OFF")
        if mode.lower() == "gate":
            self.instance.write(self.command[f"SetBurstState{channel}"] + "ON")
            self.instance.write(self.command[f"SetBurstMode{channel}"] + "GAT")


class SMBVisaSignalSource(visa_handler.VisaObject, SignalSource):
    """
    SignaSource implementation for devices that implement
    the VISA protocol and configure a switchable frequency list
    """

    def __init__(
        self, address: str, device_type: str, configuration: Dict[str, Any]
    ) -> None:
        self.address = address
        self.slist_frequencies: List[float] = []
        self.slist_amplitudes: List[float] = []
        visa_handler.VisaObject.__init__(self, address, device_type)
        SignalSource.__init__(self, configuration)
        self.attribute_map["slits_frequencies"] = self._set_slist_frequencies
        self.attribute_map["slits_amplitudes"] = self._set_slist_amplitudes

    def _set_slist_frequencies(self, slist_frequencies: List[float]) -> None:
        self.slist_frequencies = slist_frequencies
        if len(self.slist_frequencies) != 0 and (
            len(self.slist_amplitudes) == len(self.slist_amplitudes)
        ):
            self._configure_slist()

    def _set_slist_amplitudes(self, slist_amplitudes: List[float]) -> None:
        self.slist_amplitudes = slist_amplitudes
        if len(self.slist_amplitudes) != 0 and (
            len(self.slist_amplitudes) == len(self.slist_amplitudes)
        ):
            self._configure_slist()

    def _configure_slist(self) -> None:
        self.instance.write("*RST")
        self.opc_wait()
        self.instance.write("OUTP ON")
        self.opc_wait()
        self.instance.write("SOURce1:FREQ:MODE CW")
        self.opc_wait()
        # select/create list
        self.instance.write('SOURce1:LIST:SEL "SyncList"')
        self.opc_wait()

        # write frequency to list first row in arg first one being the NV
        # second one the overhauser-frequency
        set_slist_frequencies = f"SOURce1:LIST:FREQ {self.slist_frequencies[0]} Hz"
        for slist_freq in self.slist_frequencies:
            set_slist_frequencies += f", {slist_freq} Hz"
        self.instance.write(set_slist_frequencies)
        self.opc_wait()

        # write amp to list first row in arg first one being the NV
        # second one the overhauser-amp
        set_slist_amplitudes = f"SOURce1:LIST:POW {self.slist_amplitudes[0]} dBm"
        for slist_ampl in self.slist_amplitudes:
            set_slist_amplitudes += f", {slist_ampl} dBm"
        self.instance.write(set_slist_amplitudes)
        self.opc_wait()

        # set list mode to step not auto
        self.instance.write("SOURce1:LIST:MODE STEP")
        self.opc_wait()
        # set trigger type to external
        self.instance.write("SOURce1:LIST:TRIG:SOUR EXT")
        self.opc_wait()
        self.instance.write("SOURce1:FREQ:MODE LIST")
        self.opc_wait()
        logging.info("%s[done]", "SMB set slist values.".ljust(65, "."))

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_amplitude(self, ampl: ParameterInput) -> None:
        channel, ampl = ampl
        self.instance.write(self.command[f"SetAmpl{channel}"] + str(ampl))
        self.opc_wait()
        logging.info(
            f"{self.s_type} set amplitude channel {channel} to".ljust(65, ".")
            + f"{ampl}"
        )

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_frequency(self, freq: ParameterInput) -> None:
        channel, freq = freq
        self.instance.write(self.command[f"SetFreq{channel}"] + str(freq))
        self.opc_wait()
        logging.info(
            f"{self.s_type} set frequency channel {channel} to".ljust(65, ".")
            + f"{freq}"
        )


class WindFreakSNV(SignalSource):
    def __init__(self, address: str, configuration: Dict[str, Any]) -> None:
        self.address = address
        super().__init__(configuration)
        try:
            self.instance = serial.Serial(self.address, timeout=1)
            logging.info(
                f"Connected to WindFreak on {address}".ljust(65, ".") + "[done]"
            )
        except Exception:
            logging.error(
                f"Connection to WindFreak on {address} failed".ljust(65, ".")
                + "[failed]"
            )
            traceback.print_exc()
        self._set_power_level(1)
        self.attribute_map["power_level"] = self._set_power_level
        self.attribute_map["output_on_off"] = self._set_output_on_off

    def __repr__(self) -> str:
        return f"WindFreak(address: {self.address})"

    def __str__(self) -> str:
        return f"Signal source of type (synth-nv) WindFreak(address: {self.address})"

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_amplitude(self, ampl: ParameterInput) -> None:
        _channel, ampl = ampl
        self.instance.write(f"a{ampl}".encode())  # min 0 , max 63
        logging.info("Windfreak set amplitude to".ljust(65, ".") + f"{ampl}")

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_frequency(self, freq: ParameterInput) -> None:
        _channel, freq = freq
        freq = freq / 1.0e6  # convert to MHz
        self.instance.write(f"f{round(freq, 1)}".encode())
        logging.info("Windfreak set frequency to [MHz]".ljust(65, ".") + f"{freq}")

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def _set_power_level(self, power_level: ParameterInput) -> None:
        # High - 1, Low - 0
        _channel, power_level = power_level
        self.instance.write(f"h{power_level}".encode())
        logging.info("Windfreak power level set to".ljust(65, ".") + f"{power_level}")

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def _set_output_on_off(self, on_off: ParameterInput) -> None:
        _channel, on_off = on_off
        self.instance.write(f"o{on_off}".encode())
        logparam = "[ON]" if on_off == 1 else "[OFF]"
        logging.info("WindFreak output set".ljust(65, ".") + logparam)

    def close(self) -> None:
        self.instance.close()
        logging.info("WindFreak instance closed".ljust(65, ".") + "[done]")


class WindFreakHDM(SignalSource):
    def __init__(self, address: str, configuration: Dict[str, Any]) -> None:
        self.address = address
        super().__init__(configuration)
        self.instance = serial.Serial(self.address, timeout=1)
        self._set_power_level(1)
        self.attribute_map["power_level"] = self._set_power_level
        self.attribute_map["output_on_off"] = self._set_output_on_off

    def __repr__(self) -> str:
        return f"WindFreakHDM(address: {self.address})"

    def __str__(self) -> str:
        return f"Signal source of type (synth-hd) WindFreakHDM(address: {self.address})"

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_amplitude(self, ampl: ParameterInput) -> None:
        channel, ampl = ampl
        self.instance.write(f"C{channel}".encode())
        self.instance.write(f"W{ampl}".encode())  # min 0 , max 63
        logging.info("Windfreak set amplitude to".ljust(65, ".") + f"{ampl}")

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_frequency(self, freq: ParameterInput) -> None:
        channel, freq = freq
        self.instance.write(f"C{channel}".encode())
        freq = freq / 1.0e6  # convert to MHz
        self.instance.write(f"f{round(freq, 8)}".encode())
        logging.info("Windfreak set frequency to [MHz]".ljust(65, ".") + f"{freq}")

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def _set_power_level(self, power_level: ParameterInput) -> None:
        # High - 1, Low - 0
        _channel, power_level = power_level
        self.instance.write(f"h{power_level}".encode())
        logging.info("Windfreak power level set to".ljust(65, ".") + f"{power_level}")

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def _set_output_on_off(self, on_off: ParameterInput) -> None:
        _channel, on_off = on_off
        self.instance.write(f"o{on_off}".encode())
        logparam = "[ON]" if on_off == 1 else "[OFF]"
        logging.info("WindFreak output set".ljust(65, ".") + logparam)

    def close(self) -> None:
        self.instance.close()
        logging.info("WindFreak instance closed".ljust(65, ".") + "[done]")


class WindFreakOfficial(SignalSource):
    def __init__(self, address: str, configuration: Dict[str, Any]) -> None:
        self.address = address
        super().__init__(configuration)
        self.instance = SynthHD(self.address)
        self.instance.init()
        # self._set_power_level(1)
        # self.attribute_map["power_level"] = self._set_power_level
        self.attribute_map["output_on_off"] = self._set_output_on_off
        self._set_output_on_off(["channel_0", True])
        self._set_output_on_off(["channel_1", True])

    def __repr__(self) -> str:
        return f"WindFreakMini(address: {self.address})"

    def __str__(self) -> str:
        return (
            f"Signal source of type (synth-mini) WindFreakMini(address: {self.address})"
        )

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_amplitude(self, ampl: ParameterInput) -> None:
        channel, ampl = ampl
        channel = int(channel)
        ampl = float(ampl)
        self.instance[channel].power = ampl
        logging.info(
            f"Windfreak set amplitude channel{channel} to".ljust(65, ".") + f"{ampl}"
        )

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_frequency(self, freq: ParameterInput) -> None:
        channel, freq = freq
        channel = int(channel)
        freq = float(freq)
        # might need rouding
        self.instance[channel].frequency = freq
        logging.info(
            f"Windfreak set channel {channel} frequency to [Hz]".ljust(65, ".")
            + f"{freq}"
        )

    # def _set_power_level(self, power_level: ParameterInput) -> None:
    #     # High - 1, Low - 0
    #     _channel, power_level = power_level
    #     self.instance.write(f"h{power_level}".encode())
    #     logging.info("Windfreak power level set to".ljust(
    #         65, ".") + f"{power_level}")

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def _set_output_on_off(self, on_off: ParameterInput) -> None:
        channel, on_off = on_off
        channel = int(channel)
        on_off = True if on_off == 1 else False
        self.instance[channel].enable = on_off
        logparam = "[ON]" if on_off == 1 else "[OFF]"
        logging.info("WindFreak output set".ljust(65, ".") + logparam)

    def close(self) -> None:
        self.instance.close()
        logging.info("WindFreak instance closed".ljust(65, ".") + "[done]")


class WindFreakSHDMini(SignalSource):

    def __init__(self, address: str, configuration: Dict[str, Any]) -> None:
        self.address = address
        super().__init__(configuration)
        self.instance = serial.Serial(self.address, timeout=1)
        self._set_power_level(1)
        self.attribute_map["power_level"] = self._set_power_level
        self.attribute_map["output_on_off"] = self._set_output_on_off

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_amplitude(self, ampl: ParameterInput) -> None:
        self.instance.write(f"W{ampl}".encode())  # min -13.000 , max 20.000
        logging.info("Windfreak set amplitude to".ljust(65, ".") + f"{ampl}")

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_frequency(self, freq: ParameterInput) -> None:
        freq = freq / 1.0e6  # convert to MHz
        self.instance.write(f"f{round(freq, 8)}".encode())
        logging.info("Windfreak set frequency to [MHz]".ljust(65, ".") + f"{freq}")


    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def _set_power_level(self, power_level: ParameterInput) -> None:
        # High - 1, Low - 0;  only in high power mode the output actually changes with the assigned dBm
        self.instance.write(f"h{power_level}".encode())
        logging.info("Windfreak power level set to".ljust(65, ".") + f"{power_level}")

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def _set_output_on_off(self, on_off: ParameterInput) -> None:
        self.instance.write(f"E{on_off}".encode())
        logparam = "[ON]" if on_off == 1 else "[OFF]"
        logging.info("WindFreak output set".ljust(65, ".") + logparam)


    def __repr__(self) -> str:
        return f"WindFreak SynthHD Mini (address: {self.address})"

    def __str__(self) -> str:
        return f"Signal source of type WindFreak SynthHD Mini (address: {self.address})"

    def close(self) -> None:
        self.instance.close()
        logging.info("WindFreak SynthHD Mini instance closed".ljust(65, ".") + "[done]")
