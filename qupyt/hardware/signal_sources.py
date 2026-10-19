# pylint: disable=logging-format-interpolation
# pylint: disable=logging-not-lazy
"""Create and configure signal sources for measurements."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from time import sleep
from typing import Any, TypeAlias, cast

import serial
from pydantic import validate_call
from windfreak import SynthHD

from qupyt.hardware import visa_handler
from qupyt.mixins import ConfigurationError, ConfigurationMixin, UpdateConfigurationType
from qupyt.utils.decorators import coerce_device_config_shape, loop_inputs
from qupyt.utils.errors import close_on_initialization_error

ParameterInput: TypeAlias = (
    float
    | int
    | str
    | tuple[str, float | int | str]
    | list[tuple[str, float | int | str]]
)

# pylint: disable=too-few-public-methods


class DeviceFactory:
    """Factory for devices used alongside sensors and synchronisers.

    Devices can supply signals or control equipment such as motors, power supplies,
    or lasers. Static devices retain their settings throughout a measurement;
    dynamic devices can change settings between measurement steps.

    Each device exposes an attribute map describing its configurable values.
    Related devices implement a common interface, such as SignalSource.
    Call create_device directly without constructing a factory instance.

    Examples
    --------
    >>> device = DeviceFactory.create_device(
    ...     {"device_type": "Mock", "address": "mock:1", "config": {}}
    ... )
    """

    @staticmethod
    def create_device(device_info: dict[str, Any]) -> SignalSource:
        """Create and configure the requested device.

        Parameters
        ----------
        device_info : dict[str, Any]
            Device type, address, and configuration from the measurement instructions.

        Returns
        -------
        SignalSource
            Instance of the requested signal source.

        Raises
        ------
        ConfigurationError
            If the device type is unsupported.
        """
        known_devices = [
            "WindFreak",
            "WindFreakHDM",
            "Mock",
            "SRS",
            "SMB",
            "SMA",
            "Rigol",
            "TekAWG",
            "TekAFG",
            "WindFreakSNV",
            "WindFreakSHDMini",
        ]
        if device_info["device_type"] not in known_devices:
            logging.error(
                "Unknown device type %r at %s",
                device_info["device_type"],
                device_info.get("address"),
            )
            raise ConfigurationError(
                "the device type", device_info["device_type"], known_devices
            )
        logging.info(
            "Creating device type=%s at %s",
            device_info["device_type"],
            device_info["address"],
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
            if (
                device_info["device_type"] == "SMB"
                or device_info["device_type"] == "SMA"
            ):
                return SMAandSMBVisaSignalSource(
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
            if device_info["device_type"] == "TekAFG":
                return AFGSignalSource(
                    device_info["address"],
                    device_info["device_type"],
                    device_info["config"],
                )

            return VisaSignalSource(
                device_info["address"],
                device_info["device_type"],
                device_info["config"],
            )

        except Exception:
            logging.exception(
                "Could not create device type=%s at %s",
                device_info["device_type"],
                device_info["address"],
            )
            raise


class SignalSource(ABC, ConfigurationMixin):
    attribute_map: UpdateConfigurationType

    def __init__(self, configuration: dict[str, Any]) -> None:
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
        """Set the signal source output frequency."""

    @abstractmethod
    def set_amplitude(self, ampl: ParameterInput) -> None:
        """Set the signal source output amplitude.

        Notes
        -----
        Units depend on the device. Use dBm where the hardware supports it.
        """

    def set_values(self) -> None:
        if self.configuration is not None:
            self._update_from_configuration(self.configuration)

    def update_configuration(self, config: dict[str, Any]) -> None:
        setattr(self, "configuration", config)


class MockSignalSource(SignalSource):
    def __init__(self, address: str, configuration: dict[str, Any]) -> None:
        self.address = address
        super().__init__(configuration)
        self.attribute_map["phase"] = self.set_phase

    def __repr__(self) -> str:
        return f"MockSignalSource(address: {self.address})"

    def __str__(self) -> str:
        return f"Signal source of type MockSignalSource(address: {self.address})"

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_frequency(self, freq: ParameterInput) -> None:
        channel, freq = cast(tuple[str, float], freq)
        sleep(0.1)
        logging.info(
            f"MOCKING! -> set frequency channel {channel} to".ljust(65, ".") + f"{freq}"
        )

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_amplitude(self, ampl: ParameterInput) -> None:
        channel, ampl = cast(tuple[str, float], ampl)
        sleep(0.1)
        logging.info(
            f"MOCKING! -> set amplitued channel {channel} to".ljust(65, ".") + f"{ampl}"
        )

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_phase(self, phase: ParameterInput) -> None:
        channel, phase = cast(tuple[str, float], phase)
        sleep(0.1)
        logging.info(
            f"MOCKING! -> set phase channel {channel} to".ljust(65, ".") + f"{phase}"
        )

    def close(self) -> None:
        pass


class VisaSignalSource(visa_handler.VisaObject, SignalSource):
    """Signal source controlled through the VISA protocol."""

    @close_on_initialization_error
    def __init__(
        self, address: str, device_type: str, configuration: dict[str, Any]
    ) -> None:
        self.address = address
        visa_handler.VisaObject.__init__(self, address, device_type)
        SignalSource.__init__(self, configuration)

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_amplitude(self, ampl: ParameterInput) -> None:
        channel, ampl = cast(tuple[str, float], ampl)
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
        channel, freq = cast(tuple[str, float], freq)
        self.instance.write(self.command[f"SetFreq{channel}"] + str(freq))
        self.opc_wait()
        logging.info(
            f"{self.s_type} set frequency channel {channel} to".ljust(65, ".")
            + f"{freq}"
        )


class PhaseMixin:
    """Set the output signal phase for the specified channel."""

    attribute_map: UpdateConfigurationType
    command: dict[str, str]
    instance: Any
    s_type: str

    def __init__(self) -> None:
        self.attribute_map["phase"] = self.set_phase

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_phase(self, phase: ParameterInput) -> None:
        channel, phase = cast(tuple[str, float], phase)
        phase_command = self.command.get(f"SetPhase{channel}")
        if phase_command is None:
            raise ValueError(
                f"The configured signal source {repr(self)} currently does not implement setting a phase."
            )
        self.instance.write(phase_command + str(phase))
        # self.opc_wait()
        logging.info(
            f"{self.s_type} set phase channel {channel} to".ljust(65, ".") + f"{phase}"
        )


class AFGSignalSource(VisaSignalSource, PhaseMixin):
    """Special class for TekAFG to enable gating and output phase control."""

    @close_on_initialization_error
    def __init__(
        self, address: str, device_type: str, configuration: dict[str, Any]
    ) -> None:
        VisaSignalSource.__init__(self, address, device_type, configuration)
        PhaseMixin.__init__(self)
        self.attribute_map["gating"] = self._set_gate_mode
        self.attribute_map["phase"] = self.set_phase

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def _set_gate_mode(self, mode: ParameterInput) -> None:
        valids = ["off", "gate"]
        channel, mode = cast(tuple[str, str], mode)
        if channel != "1":
            raise ConfigurationError("TekAFG burst channel", channel, ["channel_1"])
        if mode.lower() not in valids:
            raise ConfigurationError("Burst mode", mode, valids)
        if mode.lower() == "off":
            self.instance.write(self.command[f"SetBurstState{channel}"] + "OFF")
        if mode.lower() == "gate":
            self.instance.write(self.command[f"SetBurstState{channel}"] + "ON")
            self.instance.write(self.command[f"SetBurstMode{channel}"] + "GATed")

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_phase(self, phase: ParameterInput) -> None:
        channel, phase = cast(tuple[str, float], phase)
        if not (0 <= phase <= 360):
            raise ConfigurationError("Output phase", phase, "range 0 to 360 degrees")

        cmd = self.command[f"SetPhase{channel}"]
        self.instance.write(f"{cmd}{phase} DEG")


class RigolSignalSource(VisaSignalSource, PhaseMixin):
    """Rigol DG1022 signal source with gating and phase control in degrees."""

    @close_on_initialization_error
    def __init__(
        self, address: str, device_type: str, configuration: dict[str, Any]
    ) -> None:
        VisaSignalSource.__init__(self, address, device_type, configuration)
        PhaseMixin.__init__(self)
        self.attribute_map["gating"] = self._set_gate_mode
        self.attribute_map["phase"] = self.set_phase
        self.attribute_map["triggered_n_cycles"] = self._set_triggered_n_cycles

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def _set_gate_mode(self, mode: ParameterInput) -> None:
        valids = ["off", "gate"]
        channel, mode = cast(tuple[str, str], mode)
        if mode.lower() not in valids:
            raise ConfigurationError("Burst mode", mode, valids)
        if mode.lower() == "off":
            self.instance.write(self.command[f"SetBurstState{channel}"] + "OFF")
        if mode.lower() == "gate":
            self.instance.write(self.command[f"SetBurstState{channel}"] + "ON")
            self.instance.write(self.command[f"SetBurstMode{channel}"] + "GAT")

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_phase(self, phase: ParameterInput) -> None:
        channel, phase = cast(tuple[str, float], phase)
        if not (-360 <= phase <= 360):
            raise ConfigurationError("Burst phase", phase, "range -360 to 360 degrees")

        cmd = self.command[f"SetBurstPhase{channel}"]
        self.instance.write(f"{cmd}{phase}")

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def _set_triggered_n_cycles(self, n_cycles: ParameterInput) -> None:
        """Configure a Rigol burst to emit the requested number of waveform cycles.

        Notes
        -----
        Enable burst mode, select TRIG mode, and set the number of cycles. Each external
        rising-edge trigger then produces that number of periods. Trigger source and
        slope retain their configured values.

        Examples
        --------
        Supported measurement YAML forms::

            triggered_n_cycles: 43
            triggered_n_cycles: ["1", 43]
            triggered_n_cycles: ["channel_1", 43]
        """
        channel, n_cycles = cast(tuple[str, float], n_cycles)
        n_cycles = int(n_cycles)

        if n_cycles < 1:
            raise ValueError("triggered_n_cycles must be >= 1")

        self.instance.write(self.command[f"SetBurstState{channel}"] + "ON")
        self.instance.write(self.command[f"SetBurstMode{channel}"] + "TRIG")
        self.instance.write(self.command[f"SetBurstNCycles{channel}"] + str(n_cycles))
        self.opc_wait()

        logging.info(
            f"{self.s_type} set triggered N-cycle burst channel {channel} to".ljust(
                65, "."
            )
            + f"{n_cycles} cycles"
        )


class SMAandSMBVisaSignalSource(VisaSignalSource):
    """VISA signal source with a configurable frequency sweep list."""

    @close_on_initialization_error
    def __init__(
        self, address: str, device_type: str, configuration: dict[str, Any]
    ) -> None:
        self.slist_frequencies: list[float] = []
        self.slist_amplitudes: list[float] = []
        super().__init__(address, device_type, configuration)
        self.attribute_map["slist_frequencies"] = self._set_slist_frequencies
        self.attribute_map["slist_amplitudes"] = self._set_slist_amplitudes

    def _set_slist_frequencies(self, slist_frequencies: list[float]) -> None:
        self.slist_frequencies = slist_frequencies
        if len(self.slist_frequencies) != 0 and (
            len(self.slist_amplitudes) == len(self.slist_frequencies)
        ):
            self._configure_slist()

    def _set_slist_amplitudes(self, slist_amplitudes: list[float]) -> None:
        self.slist_amplitudes = slist_amplitudes
        if len(self.slist_amplitudes) != 0 and (
            len(self.slist_amplitudes) == len(self.slist_frequencies)
        ):
            self._configure_slist()

    def _configure_slist(self) -> None:
        self.instance.write("*RST")
        self.opc_wait()
        self.instance.write("OUTP:STAT ON")
        self.opc_wait()
        self.instance.write("SOURce:FREQ:MODE CW")
        self.opc_wait()
        # Delete list if exists.
        if "SyncList" in self.instance.query("SOURce:LIST:CAT?"):
            self.instance.write('SOURce:LIST:DEL "SyncList"')
        # create list
        self.instance.write('SOURce:LIST:SEL "SyncList"')
        self.opc_wait()

        # set dwell time
        self.opc_wait()

        # write frequency to list first row in arg first one being the NV
        # second one the overhauser-frequency
        set_slist_frequencies = f"SOURce:LIST:FREQ {self.slist_frequencies[0]} Hz"
        for slist_freq in self.slist_frequencies[1:]:
            set_slist_frequencies += f", {slist_freq} Hz"
        self.instance.write(set_slist_frequencies)
        self.opc_wait()

        # write amp to list first row in arg first one being the NV
        # second one the overhauser-amp
        set_slist_amplitudes = f"SOURce:LIST:POW {self.slist_amplitudes[0]} dBm"
        for slist_ampl in self.slist_amplitudes[1:]:
            set_slist_amplitudes += f", {slist_ampl} dBm"
        self.instance.write(set_slist_amplitudes)
        self.opc_wait()

        # set list mode to step not auto
        self.instance.write("SOURce:LIST:MODE STEP")
        self.opc_wait()
        # set trigger type to external
        self.instance.write("SOURce:LIST:TRIG:SOUR EXT")
        self.opc_wait()
        self.instance.write("SOURce:FREQ:MODE LIST")
        self.opc_wait()
        logging.info("%s[done]", "SMB or SMA set slist values.".ljust(65, "."))


class WindFreakSNV(SignalSource):

    @close_on_initialization_error
    def __init__(self, address: str, configuration: dict[str, Any]) -> None:
        self.address = address
        super().__init__(configuration)
        try:
            self.instance = serial.Serial(self.address, timeout=1)
            logging.info(
                f"Connected to WindFreak on {address}".ljust(65, ".") + "[done]"
            )
        except Exception:
            logging.exception(
                f"Connection to WindFreak on {address} failed".ljust(65, ".")
                + "[failed]"
            )
            raise
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
        _channel, ampl = cast(tuple[str, float], ampl)
        self.instance.write(f"a{ampl}".encode())  # min 0 , max 63
        logging.info("Windfreak set amplitude to".ljust(65, ".") + f"{ampl}")

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_frequency(self, freq: ParameterInput) -> None:
        _channel, freq = cast(tuple[str, float], freq)
        freq = freq / 1.0e6  # convert to MHz
        self.instance.write(f"f{round(freq, 1)}".encode())
        logging.info("Windfreak set frequency to [MHz]".ljust(65, ".") + f"{freq}")

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def _set_power_level(self, power_level: ParameterInput) -> None:
        # High - 1, Low - 0
        _channel, power_level = cast(tuple[str, float], power_level)
        self.instance.write(f"h{power_level}".encode())
        logging.info("Windfreak power level set to".ljust(65, ".") + f"{power_level}")

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def _set_output_on_off(self, on_off: ParameterInput) -> None:
        _channel, on_off = cast(tuple[str, float], on_off)
        self.instance.write(f"o{on_off}".encode())
        logparam = "[ON]" if on_off == 1 else "[OFF]"
        logging.info("WindFreak output set".ljust(65, ".") + logparam)

    def close(self) -> None:
        if getattr(self, "instance", None) is None:
            return
        self.instance.close()
        self.instance = None
        logging.info("WindFreak instance closed".ljust(65, ".") + "[done]")


class WindFreakHDM(SignalSource):

    @close_on_initialization_error
    def __init__(self, address: str, configuration: dict[str, Any]) -> None:
        self.address = address
        super().__init__(configuration)
        self.instance = serial.Serial(self.address, timeout=1)
        logging.info("Connected to WindFreak HDM at %s", self.address)
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
        channel, ampl = cast(tuple[str, float], ampl)
        self.instance.write(f"C{channel}".encode())
        self.instance.write(f"W{ampl}".encode())  # min 0 , max 63
        logging.info("Windfreak set amplitude to".ljust(65, ".") + f"{ampl}")

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_frequency(self, freq: ParameterInput) -> None:
        channel, freq = cast(tuple[str, float], freq)
        self.instance.write(f"C{channel}".encode())
        freq = freq / 1.0e6  # convert to MHz
        self.instance.write(f"f{round(freq, 8)}".encode())
        logging.info("Windfreak set frequency to [MHz]".ljust(65, ".") + f"{freq}")

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def _set_power_level(self, power_level: ParameterInput) -> None:
        # High - 1, Low - 0
        _channel, power_level = cast(tuple[str, float], power_level)
        self.instance.write(f"h{power_level}".encode())
        logging.info("Windfreak power level set to".ljust(65, ".") + f"{power_level}")

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def _set_output_on_off(self, on_off: ParameterInput) -> None:
        _channel, on_off = cast(tuple[str, float], on_off)
        self.instance.write(f"o{on_off}".encode())
        logparam = "[ON]" if on_off == 1 else "[OFF]"
        logging.info("WindFreak output set".ljust(65, ".") + logparam)

    def close(self) -> None:
        if getattr(self, "instance", None) is None:
            return
        self.instance.close()
        self.instance = None
        logging.info("WindFreak instance closed".ljust(65, ".") + "[done]")


class WindFreakOfficial(SignalSource):

    @close_on_initialization_error
    def __init__(self, address: str, configuration: dict[str, Any]) -> None:
        self.address = address
        super().__init__(configuration)
        self.instance = SynthHD(self.address)
        self.instance.init()
        logging.info("Connected to WindFreak SynthHD at %s", self.address)
        # self._set_power_level(1)
        # self.attribute_map["power_level"] = self._set_power_level
        self.attribute_map["output_on_off"] = self._set_output_on_off
        self._set_output_on_off(("channel_0", True))
        self._set_output_on_off(("channel_1", True))

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
        channel, ampl = cast(tuple[str, float], ampl)
        channel_index = int(channel)
        ampl = float(ampl)
        self.instance[channel_index].power = ampl
        logging.info(
            f"Windfreak set amplitude channel{channel_index} to".ljust(65, ".")
            + f"{ampl}"
        )

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_frequency(self, freq: ParameterInput) -> None:
        channel, freq = cast(tuple[str, float], freq)
        channel_index = int(channel)
        freq = float(freq)
        # might need rouding
        self.instance[channel_index].frequency = freq
        logging.info(
            f"Windfreak set channel {channel_index} frequency to [Hz]".ljust(65, ".")
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
        channel, on_off = cast(tuple[str, float], on_off)
        channel_index = int(channel)
        enabled = on_off == 1
        self.instance[channel_index].enable = enabled
        logparam = "[ON]" if enabled else "[OFF]"
        logging.info("WindFreak output set".ljust(65, ".") + logparam)

    def close(self) -> None:
        if getattr(self, "instance", None) is None:
            return
        self.instance.close()
        self.instance = None
        logging.info("WindFreak instance closed".ljust(65, ".") + "[done]")


class WindFreakSHDMini(SignalSource):

    @close_on_initialization_error
    def __init__(self, address: str, configuration: dict[str, Any]) -> None:
        self.address = address
        super().__init__(configuration)
        self.instance = serial.Serial(self.address, timeout=1)
        logging.info("Connected to WindFreak SynthHD Mini at %s", self.address)
        self._set_power_level(1)
        self.attribute_map["power_level"] = self._set_power_level
        self.attribute_map["output_on_off"] = self._set_output_on_off

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_amplitude(self, ampl: ParameterInput) -> None:
        channel, ampl = cast(tuple[str, float], ampl)
        _channel_index = int(channel)
        ampl = float(ampl)
        self.instance.write(f"W{ampl}".encode())  # min -13.000, max 20.000
        logging.info("Windfreak set amplitude to".ljust(65, ".") + f"{ampl}")

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def set_frequency(self, freq: ParameterInput) -> None:
        channel, freq = cast(tuple[str, float], freq)
        _channel_index = int(channel)
        freq = float(freq)
        freq = freq / 1.0e6  # convert to MHz
        self.instance.write(f"f{round(freq, 8)}".encode())
        logging.info("Windfreak set frequency to [MHz]".ljust(65, ".") + f"{freq}")

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def _set_power_level(self, power_level: ParameterInput) -> None:
        _channel, power_level = cast(tuple[str, float], power_level)
        # High - 1, Low - 0;  only in high power mode the output actually changes with the assigned dBm
        self.instance.write(f"h{power_level}".encode())
        logging.info("Windfreak power level set to".ljust(65, ".") + f"{power_level}")

    @validate_call
    @coerce_device_config_shape
    @loop_inputs
    def _set_output_on_off(self, on_off: ParameterInput) -> None:
        _channel, on_off = cast(tuple[str, float], on_off)
        self.instance.write(f"E{on_off}".encode())
        logparam = "[ON]" if on_off == 1 else "[OFF]"
        logging.info("WindFreak output set".ljust(65, ".") + logparam)

    def __repr__(self) -> str:
        return f"WindFreak SynthHD Mini (address: {self.address})"

    def __str__(self) -> str:
        return f"Signal source of type WindFreak SynthHD Mini (address: {self.address})"

    def close(self) -> None:
        if getattr(self, "instance", None) is None:
            return
        self.instance.close()
        self.instance = None
        logging.info("WindFreak SynthHD Mini instance closed".ljust(65, ".") + "[done]")
