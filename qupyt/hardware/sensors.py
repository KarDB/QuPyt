# pylint: disable=logging-format-interpolation
# pylint: disable=logging-not-lazy
"""Create, configure, and acquire data from measurement sensors."""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from time import monotonic, time
from typing import Any

import numpy as np
from harvesters.core import Harvester
from numpy.typing import NDArray
from pypylon import pylon

try:
    from egrabber import (
        BUFFER_INFO_BASE,
        BUFFER_INFO_CUSTOM_NUM_PARTS,
        BUFFER_INFO_CUSTOM_PART_SIZE,
        BUFFER_INFO_TIMESTAMP,
        INFO_DATATYPE_PTR,
        INFO_DATATYPE_SIZET,
        INFO_DATATYPE_UINT64,
        Buffer,
        EGenTL,
        EGrabber,
        EGrabberDiscovery,
    )
except ImportError:
    logging.warning(
        "Could not load egrabber library".ljust(65, ".")
        + "[failed]\nIf you are not using a Phantom S710 (or similar) camera you do not need this!"
    )

from qupyt.hardware.synchronisers import Synchroniser
from qupyt.mixins import ConfigurationError, ConfigurationMixin, UpdateConfigurationType
from qupyt.utils.decorators import log_sensor_acquisition
from qupyt.utils.errors import (
    cleanup_after_error,
    cleanup_on_exit,
    close_on_initialization_error,
)

# Imports for HeliCam
if sys.platform == "win32":
    # from msvcrt import getch
    prgPath = r"C:\Program Files"
    sys.path.insert(0, prgPath + r"\Heliotis\heliCam\Python\wrapper")
else:
    # from getch import getch
    sys.path.insert(0, r"/usr/share/libhelic/python/wrapper")
try:
    import libHeLIC as heli
except ImportError:
    logging.warning(
        "Could not load HeliCam library".ljust(65, ".")
        + "[failed]\nIf you are not using a HeliCam you do not need this!"
    )
try:
    import nidaqmx
    from nidaqmx.constants import (
        AcquisitionType,
        Edge,
        TerminalConfiguration,
        VoltageUnits,
    )
except ImportError:
    logging.warning(
        "Could not load NI-DAQ library".ljust(65, ".")
        + "[failed]\nIf you are not using a NI-DAQ you do not need this!"
    )

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


# pylint: disable=too-few-public-methods
class SensorFactory:
    """Factory for sensors that implement the common Sensor interface.

    Call create_sensor directly without constructing a factory instance.

    Examples
    --------
    >>> cam = SensorFactory.create_sensor("MockCam", {"number_measurements": 10})
    """

    @staticmethod
    def create_sensor(sensor_type: str, configuration: dict[str, Any]) -> Sensor:
        """Create and configure the requested sensor.

        Parameters
        ----------
        sensor_type : str
            Sensor model identifier, such as Basler1920, DAQ, or MockCam.
        configuration : dict[str, Any]
            Parameters supported by the selected sensor. See its class documentation.

        Returns
        -------
        Sensor
            Configured instance of the requested sensor.

        Raises
        ------
        ValueError
            If the sensor type is unknown.
        ImportError
            If the selected sensor's optional driver is unavailable.
        """
        logging.info("Creating sensor type=%s", sensor_type)
        try:
            sensor_types: dict[str, Callable[[dict[str, Any]], Sensor]] = {
                "Basler1920": BaslerCam,
                "EoSense1.1CXP": GenICamHarvester,
                "PhantomS710": GenICamPhantom,
                "HeliC3": HeliCam,
                "DAQ": DAQ,
                "MockCam": MockCam,
            }
            if sensor_type not in sensor_types:
                raise ValueError(f"Requested sensor type {sensor_type} does not exists")
            dependencies = {"PhantomS710": "EGenTL", "HeliC3": "heli", "DAQ": "nidaqmx"}
            dependency = dependencies.get(sensor_type)
            if dependency is not None and globals().get(dependency) is None:
                raise ImportError(
                    f"Sensor {sensor_type} requires the unavailable {dependency} driver"
                )
            sensor = sensor_types[sensor_type](configuration)
        except Exception:
            logging.exception("Could not create sensor type=%s", sensor_type)
            raise
        logging.info("Created sensor type=%s", sensor_type)
        return sensor


class Sensor(ABC, ConfigurationMixin):
    """Abstract interface for interchangeable measurement sensors.

    Parameters
    ----------
    configuration : dict[str, Any]
        Configuration for the concrete sensor. Common keys are number_measurements
        and target_data_type; subclasses may accept additional keys.

    Attributes
    ----------
    roi_shape : list[int]
        Dimensions of one sensor readout, derived from the concrete sensor settings.
    target_data_type : type
        Acquisition data type, chosen to be compatible with the data container.
    number_measurements : int
        Number of images or samples acquired in one burst.

    Notes
    -----
    Configure sensors through the configuration dictionary rather than setting
    these attributes directly.
    """

    attribute_map: UpdateConfigurationType

    def __init__(
        self, configuration: dict[str, Any]
    ) -> None:  # pylint: disable=unused-argument
        # configuration is not used in the ABC, however
        # all child classes must take it as input.
        self.roi_shape: list[int]
        self.number_measurements: int = 2
        self.target_data_type: type
        self.attribute_map = {
            "number_measurements": lambda x: setattr(self, "number_measurements", x),
            "target_data_type": lambda x: setattr(self, "target_data_type", x),
        }

    @abstractmethod
    def open(self) -> None:
        """Discover the sensor and establish its connection."""

    @abstractmethod
    def acquire_data(self, synchroniser: Synchroniser | None = None) -> NDArray[Any]:
        """Acquire one measurement burst.

        Parameters
        ----------
        synchroniser : Synchroniser or None, optional
            Prepared synchroniser to trigger after the sensor is ready. None permits
            acquisition without a synchroniser.

        Returns
        -------
        numpy.ndarray
            Measurements with shape ``(number_measurements, *roi_shape)``.

        Notes
        -----
        Prepare the synchroniser before the sensor. The sensor can then trigger the
        sequence when acquisition is ready.
        """

    @abstractmethod
    def close(self) -> None:
        """Close the connection and release the sensor instance."""


class GenICamPhantom(Sensor):
    """Phantom S710 sensor using the Euresys EGrabber library.

    Parameters
    ----------
    configuration : dict[str, Any]
        Sensor configuration extending the common Sensor settings.

        Supported keys:

        * exposure_time: exposure in microseconds.
        * image_roi: [height, width, x_offset, y_offset]. Height supports 32 to
          800 in steps of 8; width supports 128 to 1280 in steps of 128. Offsets
          are ignored because this camera supports only a centered ROI.
        * pixel_bits: Mono8, Mono12, or Mono16; Bayer formats are unsupported.
        * trigger_mode: On or Off, with the supported capitalization variants.
        * trigger_source: camera trigger input, such as GPIO0 or GPIO1.

    Raises
    ------
    ConfigurationError
        If a sensor setting is invalid.

    Notes
    -----
    Install the Euresys egrabber software and Python wheel. This implementation
    requires a compatible frame grabber and camera.
    """

    @close_on_initialization_error
    def __init__(self, configuration: dict[str, Any]) -> None:
        if globals().get("EGenTL") is None:
            raise ImportError("GenICamPhantom requires the egrabber driver")
        self.cam, self.cam_instance = self._discover_and_setup()
        super().__init__(configuration)
        # self.cam.remote_device.node_map.OffsetX.value = 0
        # self.cam.remote_device.node_map.OffsetY.value = 0
        self.cam.remote.set("Height", 200)
        self.cam.remote.set("Width", 1280)
        self.cam.remote.set("PixelFormat", "Mono8")
        self.image_dtype: type[np.unsignedinteger[Any]] = np.uint8
        self.cam.remote.set("TriggerSource", "GPIO0")
        self.cam.remote.set("TriggerMode", "TriggerModeOn")
        self.cam.remote.set("ExposureTime", 20)
        self.cam.stream.set("UnpackingMode", "Lsb")
        self.cam.stream.set("UnpackingMode", "Lsb")
        self.cam.remote.set("FlatFieldCorrection", "FlatFieldCorrectionOff")
        self.roi_shape = [
            self.cam.remote.get("Height") * 4,
            self.cam.remote.get("Width"),
        ]
        self.attribute_map["exposure_time"] = self._set_exposure_time
        self.attribute_map["image_roi"] = self._set_roi
        # self.attribute_map['gain'] = self._set_gain
        self.attribute_map["pixel_bits"] = self._set_pixel_bits
        self.attribute_map["trigger_mode"] = self._set_trigger_mode
        self.attribute_map["trigger_source"] = self._set_trigger_source
        self.initial_configuration_dict = configuration
        if configuration is not None:
            self._update_from_configuration(configuration)

    def __repr__(self) -> str:
        return f"GenICamPhantom(configuration: {self.initial_configuration_dict})"

    def __str__(self) -> str:
        return f"Phantom S710 camera instance: GenICamPhantom(configuration: {self.initial_configuration_dict})"

    def _discover_and_setup(self) -> Any:
        gentl = EGenTL()
        discovery = EGrabberDiscovery(gentl)
        discovery.discover()
        cam = discovery.cameras[0]
        self.grabber = EGrabber(cam)
        return self.grabber, cam

    def _set_trigger_source(self, trigger_source: str) -> None:
        """Set the camera trigger input.

        Parameters
        ----------
        trigger_source : str
            Camera input selector, such as GPIO0 or GPIO1.
        """
        self.cam.remote.set("TriggerSource", trigger_source)

    def _set_trigger_mode(self, trigger_mode: str) -> None:
        """Set whether the camera uses an external trigger.

        Parameters
        ----------
        trigger_mode : str
            On or Off, with supported capitalization variants.

        Raises
        ------
        ConfigurationError
            If the trigger mode is unsupported.
        """
        if trigger_mode.lower() == "on":
            self.cam.remote.set("TriggerMode", "TriggerModeOn")
        elif trigger_mode.lower() == "off":
            self.cam.remote.set("TriggerMode", "TriggerModeOff")
        else:
            raise ConfigurationError("trigger_mode", trigger_mode, ["on", "off"])

    def _set_pixel_bits(self, pixel_bits: str) -> None:
        """Set the monochrome pixel format.

        Parameters
        ----------
        pixel_bits : str
            One of Mono8, Mono12, or Mono16.

        Raises
        ------
        ConfigurationError
            If the pixel format is unsupported.
        """
        if pixel_bits.lower() == "mono8":
            self.image_dtype = np.uint8
        elif pixel_bits.lower() in ["mono12", "mono16"]:
            self.image_dtype = np.uint16
        else:
            #  The 'bayer' pixel format is not supported
            raise ConfigurationError(
                "pixel_bits", pixel_bits, ["mono8", "mono12", "mono16"]
            )
        self.cam.remote.set("PixelFormat", pixel_bits)

    def _set_exposure_time(self, exposure_time: int) -> None:
        self.cam.remote.set("ExposureTime", exposure_time)

    # def _set_gain(self, gain: int) -> None:
    #     self.cam.remote_device.node_map.Gain.value = gain

    def _set_roi(self, roi_shape_and_offset: list[int]) -> None:
        """Set the centered region of interest on the Phantom camera.

        Parameters
        ----------
        roi_shape_and_offset : list[int]
            [height, width, x_offset, y_offset]; offsets are ignored.

        Notes
        -----
        Height supports 32 to 800 in steps of 8. Some heights in steps of 4 may be
        accepted by the driver but adjusted during acquisition. Width supports 128 to
        1280 in steps of 128. Nonzero offsets produce a warning.
        """
        roi_shape_h_and_w = roi_shape_and_offset[:2]
        if roi_shape_and_offset[2] or roi_shape_and_offset[3]:
            logging.warning(
                f"Offset X:{roi_shape_and_offset[2]} and Y:{roi_shape_and_offset[3]} are ignored, as the Phantom S710\n"
                + "does not support ROI offset and is always centered. Consider resizing the ROI to 0,0.".ljust(
                    65, "."
                )
                + "[warning]"
            )

        # Checking size compatibility of chosen FOV.
        if roi_shape_h_and_w[0] % 8 != 0:
            raise ValueError(
                "ROI Height for the Phantom S710 has to be a multiple of 8"
            )

        roi_shape_h_and_w[0] = int(np.round(roi_shape_h_and_w[0] / 4))

        if roi_shape_h_and_w[1] % 128 != 0:
            raise ValueError(
                f"ROI Width for the Phantom S710 has to be a multiple of 128px; {roi_shape_h_and_w[1]}px was specified"
            )
        if roi_shape_h_and_w[1] < 128:
            raise ValueError(
                f"ROI Width for the Phantom S710 has to be at least 128px; {roi_shape_h_and_w[1]}px was specified"
            )
        if roi_shape_h_and_w[1] > 1280:
            raise ValueError(
                f"ROI Width for the Phantom S710 has to be at most 1280px; {roi_shape_h_and_w[1]}px was specified"
            )

        if roi_shape_h_and_w[0] > 200:
            raise ValueError(
                f"ROI Height for the Phantom S710 has to be at most 800px; {roi_shape_h_and_w[0]*4}px  ({roi_shape_h_and_w[0]}px per Sensor) was specified"
            )
        if roi_shape_h_and_w[0] < 8:
            raise ValueError(
                f"ROI Height for the Phantom S710 has to be at least 32px; {roi_shape_h_and_w[0]*4}px  ({roi_shape_h_and_w[0]}px per Sensor) was specified"
            )

        try:
            self.cam.remote.set("Height", roi_shape_h_and_w[0])
            self.cam.remote.set("Width", roi_shape_h_and_w[1])
            self.roi_shape = [roi_shape_h_and_w[0] * 4, roi_shape_h_and_w[1]]
            logging.info(
                f"Set Sensor roi to height: {roi_shape_h_and_w[0]*4} and width: {roi_shape_h_and_w[1]}\n".ljust(
                    65, "."
                )
                + "[done]"
            )
        except Exception:
            logging.exception(
                f"Failed to set roi of height: {roi_shape_h_and_w[0]*4} and width: {roi_shape_h_and_w[1]}\n".ljust(
                    65, "."
                )
                + "[failed]"
            )
            raise

    @log_sensor_acquisition
    def acquire_data(self, synchroniser: Synchroniser | None = None) -> NDArray[Any]:
        """See :meth:`Sensor.acquire_data`."""
        height = self.cam.remote.get("Height")
        width = self.cam.remote.get("Width")

        self.cam.realloc_buffers(self.number_measurements)
        with cleanup_on_exit(self.cam.stop, "Phantom acquisition stop"):
            self.cam.start()
            data = np.zeros(
                (self.number_measurements, height * 4, width), dtype=np.uint32
            )
            timesteps = np.zeros(self.number_measurements, dtype=np.float64)
            if synchroniser is not None:
                synchroniser.trigger()
            for i in range(self.number_measurements):
                with Buffer(self.cam) as buffer:
                    buffer_ptr, image_size, part_num, timestep = self._grab_frame_info(
                        buffer
                    )
                    raw_frame = self._move_frame_from_pool(
                        buffer_ptr, image_size, part_num
                    )
                    data[i] += raw_frame
                    timesteps[i] += timestep

        if self.number_measurements > 1 and timesteps[-1] > timesteps[0]:
            logging.debug(
                "Camera measured framerate: %s Hz",
                (self.number_measurements - 1)
                / ((timesteps[-1] - timesteps[0]) * 1e-6),
            )

        return data

    def _grab_frame_info(self, buffer: Any) -> tuple[int, int, int, int]:
        buffer_ptr = buffer.get_info(BUFFER_INFO_BASE, INFO_DATATYPE_PTR)
        image_size = buffer.get_info(BUFFER_INFO_CUSTOM_PART_SIZE, INFO_DATATYPE_SIZET)
        part_num = buffer.get_info(BUFFER_INFO_CUSTOM_NUM_PARTS, INFO_DATATYPE_SIZET)
        time_stamp = buffer.get_info(BUFFER_INFO_TIMESTAMP, INFO_DATATYPE_UINT64)

        return buffer_ptr, image_size, part_num, time_stamp

    def _move_frame_from_pool(
        self, buffer_ptr: int | ctypes.c_void_p, image_size: int, part_num: int
    ) -> NDArray[Any]:
        frame = np.empty(
            (part_num * self.roi_shape[0], self.roi_shape[1]), dtype=self.image_dtype
        )
        frame_ptr = frame.ctypes.data_as(ctypes.c_void_p)
        ctypes.memmove(frame_ptr, buffer_ptr, image_size * part_num)
        return frame

    def open(self) -> None:
        """Keep the camera connection established during initialization."""
        logging.info("GenICamPhantom camera ready (configured during initialization)")

    def close(self) -> None:
        """Destroy the camera instance and release its exclusive connection.

        A new instance can be created after the current connection is released.
        """
        if all(
            getattr(self, name, None) is None
            for name in ("cam", "cam_instance", "grabber")
        ):
            return
        self.cam = None
        self.cam_instance = None
        self.grabber = None
        gc.collect()
        logging.info("Closed GenICam camera connection".ljust(65, ".") + "[done]")


class GenICamHarvester(Sensor):
    """GenICam sensor using the Harvester library and a supplied GenTL producer.

    Parameters
    ----------
    configuration : dict[str, Any]
        Sensor configuration extending the common Sensor settings.

        Supported keys:

        * exposure_time: exposure in microseconds.
        * image_roi: [height, width, x_offset, y_offset].
        * GenTL_producer_cti: path to the GenTL producer CTI file.

    Raises
    ------
    FileNotFoundError
        If the configured CTI file is missing.

    Notes
    -----
    Supply a GenTL producer from the camera or frame-grabber manufacturer. See
    `Harvester <https://github.com/genicam/harvesters>`_ for producer guidance.

    The implementation preconfigures CXP12_X4, Mono10, trigger source Line0,
    and trigger selector FrameStart.
    """

    @close_on_initialization_error
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.harvester = Harvester()
        self.cti_file = configuration["GenTL_producer_cti"]
        try:
            self.harvester.add_file(self.cti_file)
        except OSError:
            logging.exception(
                "Could not find GenTL producer file (.cti)".ljust(65, ".") + "[failed]"
            )
            logging.exception("Tried to locate file at %s", self.cti_file)
            raise
        self.harvester.update()
        self.cam = self.harvester.create()
        super().__init__(configuration)
        self.cam.remote_device.node_map.CxpLinkConfiguration.value = "CXP12_X4"
        self.cam.remote_device.node_map.PixelFormat.value = "Mono10"
        self.cam.remote_device.node_map.TriggerMode.value = "On"
        self.cam.remote_device.node_map.TriggerSource.value = "Line0"
        self.cam.remote_device.node_map.TriggerSelector.value = "FrameStart"
        self.cam.remote_device.node_map.ExposureTime.value = 20
        self.cam.remote_device.node_map.OffsetX.value = 0
        self.cam.remote_device.node_map.OffsetY.value = 0
        self.cam.remote_device.node_map.Height.value = 800
        self.cam.remote_device.node_map.Width.value = 1200
        self.roi_shape = [
            self.cam.remote_device.node_map.Height.value,
            self.cam.remote_device.node_map.Width.value,
        ]
        self.attribute_map["exposure_time"] = self._set_exposure_time
        self.attribute_map["image_roi"] = self._set_roi
        self.attribute_map["gain"] = self._set_gain
        self.initial_configuration_dict = configuration
        # GenTL needs to be set above.
        self.attribute_map["GenTL_producer_cti"] = self._throw_away_cti
        if configuration is not None:
            self._update_from_configuration(configuration)

    def __repr__(self) -> str:
        return f"GenICamHarvester(configuration: {self.initial_configuration_dict})"

    def __str__(self) -> str:
        return f"GenICam Harvester compliant camera instance: GenICamHarvester(configuration: {self.initial_configuration_dict})"

    def _throw_away_cti(self, cti_file: str) -> None:
        """Keep the GenTL producer chosen during initialization."""

    def _set_exposure_time(self, exposure_time: int) -> None:
        self.cam.remote_device.node_map.ExposureTime.value = exposure_time

    def _set_gain(self, gain: int) -> None:
        self.cam.remote_device.node_map.Gain.value = gain

    def _set_roi(self, roi_shape_and_offset: list[int]) -> None:
        """Set the camera region of interest and offsets.

        Parameters
        ----------
        roi_shape_and_offset : list[int]
            [height, width, x_offset, y_offset].

        Raises
        ------
        Exception
            If the driver rejects an ROI setting; the failure is logged and propagated.
        """
        roi_shape_h_and_w = roi_shape_and_offset[:2]
        roi_offset_x_and_y = roi_shape_and_offset[2:]
        try:
            self.cam.remote_device.node_map.Height.value = roi_shape_h_and_w[0]
            self.cam.remote_device.node_map.Width.value = roi_shape_h_and_w[1]
            self.cam.remote_device.node_map.OffsetX.value = roi_offset_x_and_y[0]
            self.cam.remote_device.node_map.OffsetY.value = roi_offset_x_and_y[1]
            self.roi_shape = roi_shape_h_and_w
            logging.info(
                "Set Sensor roi to height: {} and width: {}\n\
                          with offset X: {} and Y: {}".format(
                    roi_shape_h_and_w[0],
                    roi_shape_h_and_w[1],
                    roi_offset_x_and_y[0],
                    roi_offset_x_and_y[1],
                ).ljust(
                    65, "."
                )
                + "[done]"
            )
        except Exception:
            logging.exception(
                "Failed to set roi of height: {} and width: {}\n\
                               with offset X: {} and Y: {}".format(
                    roi_shape_h_and_w[0],
                    roi_shape_h_and_w[1],
                    roi_offset_x_and_y[0],
                    roi_offset_x_and_y[1],
                ).ljust(
                    65, "."
                )
                + "[failed]"
            )
            raise

    @log_sensor_acquisition
    def acquire_data(self, synchroniser: Synchroniser | None = None) -> NDArray[Any]:
        """See :meth:`Sensor.acquire_data`."""
        time_1 = time()
        height = self.cam.remote_device.node_map.Height.value
        width = self.cam.remote_device.node_map.Width.value
        with cleanup_on_exit(self.cam.stop, "Harvester acquisition stop"):
            self.cam.start()
            data = np.zeros((self.number_measurements, height * width), dtype=np.uint32)
            if synchroniser is not None:
                synchroniser.trigger()
            for i in range(self.number_measurements):
                with self.cam.fetch() as buffer:
                    component = buffer.payload.components[0]
                    data[i] = component.data
        time_2 = time()
        logging.debug(
            f"Data acquisition took {time_2-time_1} s".ljust(65, ".") + "[done]"
        )
        return data.reshape((self.number_measurements, height, width))

    def open(self) -> None:
        """Keep the camera connection established during initialization."""
        logging.info("GenICamHarvester camera ready (configured during initialization)")

    def close(self) -> None:
        """Destroy the camera instance and reset Harvester.

        Release the exclusive camera connection so a new instance can be created.
        """
        cam = getattr(self, "cam", None)
        harvester = getattr(self, "harvester", None)

        def reset_harvester() -> None:
            if harvester is not None:
                harvester.reset()
                self.harvester = None

        with cleanup_on_exit(reset_harvester, "Harvester reset"):
            if cam is not None:
                cam.destroy()
                self.cam = None
        logging.info("Closed GenICam camera connection".ljust(65, ".") + "[done]")


class BaslerCam(Sensor):
    """Basler monochrome sensor using the pypylon library.

    Parameters
    ----------
    configuration : dict[str, Any]
        Sensor configuration extending the common Sensor settings.

        Supported keys:

        * exposure_time: exposure in microseconds.
        * image_roi: [height, width, x_offset, y_offset].
        * binning_horizontal and binning_vertical: supported binning factors,
          typically 1, 2, 3, or 4, depending on the camera model.
        * binning_mode_horizontal and binning_mode_vertical: sum or average.

    Notes
    -----
    Install Basler's pylon software and the
    `pypylon <https://github.com/basler/pypylon>`_ package.
    GenICam-compatible Basler cameras can also use GenICamHarvester.
    """

    @close_on_initialization_error
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.cam = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateFirstDevice()
        )
        self.cam.Open()
        self._configure_defaults()
        self._configure_const()
        super().__init__(configuration)
        self.attribute_map["exposure_time"] = self._set_exposure_time
        self.attribute_map["binning_horizontal"] = self._set_binning_horizontal
        self.attribute_map["trigger_line"] = self._set_trigger_line
        self.attribute_map["trigger_mode"] = self._set_trigger_mode
        self.attribute_map["binning_vertical"] = self._set_binning_vertical
        self.attribute_map["binning_mode_horizontal"] = (
            self._set_mode_binning_horizontal
        )
        self.attribute_map["binning_mode_vertical"] = self._set_mode_binning_vertical
        self.attribute_map["image_roi"] = self._set_roi
        self.initial_configuration_dict = configuration
        if configuration is not None:
            self._update_from_configuration(configuration)

    def __repr__(self) -> str:
        return f"BaslerCam(configuration: {self.initial_configuration_dict})"

    def __str__(self) -> str:
        return f"Basler pylon camera instance: BaslerCam(configuration: {self.initial_configuration_dict})"

    def _configure_defaults(self) -> None:
        self.cam.BinningHorizontal.SetValue(1)
        self.cam.BinningVertical.SetValue(1)
        self.cam.BinningHorizontalMode.SetValue("Sum")
        self.cam.BinningVerticalMode.SetValue("Sum")
        self.cam.OffsetX.SetValue(0)
        self.cam.OffsetY.SetValue(0)
        self.cam.Height.SetValue(540)
        self.cam.Width.SetValue(220)
        self.cam.ExposureTime.SetValue(700)

    def _configure_const(self) -> None:
        self.cam.TriggerActivation.SetValue("RisingEdge")
        self.cam.Gain.SetValue(0)
        self.cam.TriggerSelector.SetValue("FrameStart")
        self.cam.TriggerMode.SetValue("On")
        self.cam.TriggerSource.SetValue("Line3")
        self.cam.LineSelector.SetValue("Line3")
        self.cam.LineInverter.SetValue(False)
        self.cam.LineMode.SetValue("Input")
        self.cam.PixelFormat.SetValue("Mono12")
        self.cam.ExposureMode.SetValue("Timed")

    @log_sensor_acquisition
    def acquire_data(self, synchroniser: Synchroniser | None = None) -> NDArray[Any]:
        """See :meth:`Sensor.acquire_data`."""
        time_1 = time()
        arr = np.zeros((self.number_measurements, *self.roi_shape))
        with cleanup_on_exit(self.cam.StopGrabbing, "Basler acquisition stop"):
            self.cam.StartGrabbingMax(self.number_measurements)
            if synchroniser is not None:
                synchroniser.trigger()
            for i in range(self.number_measurements):
                grab_result = self.cam.RetrieveResult(
                    int(5000), pylon.TimeoutHandling_ThrowException
                )
                with cleanup_on_exit(grab_result.Release, "Basler frame release"):
                    if not grab_result.GrabSucceeded():
                        raise RuntimeError(
                            f"Basler frame grab failed: {grab_result.ErrorDescription}"
                        )
                    frame = np.asarray(grab_result.Array)
                    arr[i] = frame
        time_2 = time()
        logging.debug(
            f"Basler data acquisition took {time_2-time_1} s".ljust(65, ".") + "[done]"
        )
        return arr

    def _set_trigger_line(self, trigger_line: str) -> None:
        self.cam.TriggerSelector.SetValue("FrameStart")
        self.cam.TriggerMode.SetValue("On")
        self.cam.TriggerSource.SetValue(trigger_line)
        self.cam.LineSelector.SetValue(trigger_line)
        self.cam.LineInverter.SetValue(False)
        self.cam.LineMode.SetValue("Input")

    def _set_trigger_mode(self, trigger_mode: str) -> None:
        if trigger_mode.lower() == "on":
            self.cam.TriggerMode.SetValue("On")
        elif trigger_mode.lower() == "off":
            self.cam.TriggerMode.SetValue("Off")
        else:
            raise ConfigurationError("trigger_mode", trigger_mode, ["on", "off"])

    def _set_exposure_time(self, exposure_time: int) -> None:
        self.cam.ExposureTime.SetValue(exposure_time)

    def _set_binning_horizontal(self, binning_horizontal: int) -> None:
        self.cam.BinningHorizontal.SetValue(binning_horizontal)

    def _set_binning_vertical(self, binning_vertical: int) -> None:
        self.cam.BinningVertical.SetValue(binning_vertical)

    def _set_mode_binning_horizontal(self, mode_binning_horizontal: str) -> None:
        self.cam.BinningHorizontalMode.SetValue(mode_binning_horizontal)

    def _set_mode_binning_vertical(self, mode_binning_vertical: str) -> None:
        self.cam.BinningVerticalMode.SetValue(mode_binning_vertical)

    def _set_roi(self, roi_shape_and_offset: list[int]) -> None:
        """Set the camera region of interest and offsets.

        Parameters
        ----------
        roi_shape_and_offset : list[int]
            [height, width, x_offset, y_offset].

        Raises
        ------
        Exception
            If the driver rejects an ROI setting; the failure is logged and propagated.
        """
        roi_shape_h_and_w = roi_shape_and_offset[:2]
        roi_offset_x_and_y = roi_shape_and_offset[2:]
        try:
            self._set_roi_shape(roi_shape_h_and_w)
            self._set_roi_offset_x_and_y(roi_offset_x_and_y)
            self.roi_shape = roi_shape_h_and_w
            logging.info(
                "Set Sensor roi to height: {} and width: {}\n\
                          with offset X: {} and Y: {}".format(
                    roi_shape_h_and_w[0],
                    roi_shape_h_and_w[1],
                    roi_offset_x_and_y[0],
                    roi_offset_x_and_y[1],
                ).ljust(
                    65, "."
                )
                + "[done]"
            )
        except Exception:
            logging.exception(
                "Failed to set roi of height: {} and width {}\n\
                               with offset X: {} and Y: {}".format(
                    roi_shape_h_and_w[0],
                    roi_shape_h_and_w[1],
                    roi_offset_x_and_y[0],
                    roi_offset_x_and_y[1],
                ).ljust(
                    65, "."
                )
                + "[failed]"
            )
            raise

    def _set_roi_shape(self, roi_shape_h_and_w: list[int]) -> None:
        self.cam.Height.SetValue(roi_shape_h_and_w[0])
        self.cam.Width.SetValue(roi_shape_h_and_w[1])

    def _set_roi_offset_x_and_y(self, roi_offset_x_and_y: list[int]) -> None:
        self.cam.OffsetX.SetValue(roi_offset_x_and_y[0])
        self.cam.OffsetY.SetValue(roi_offset_x_and_y[1])

    def open(self) -> None:
        """Open the camera connection."""
        logging.info("Opened Basler camera".ljust(65, ".") + "[done]")

    def close(self) -> None:
        """Close the camera connection.

        A new camera instance can be created after the connection is released.
        """
        if getattr(self, "cam", None) is None:
            return
        with cleanup_on_exit(self.cam.Close, "Basler connection close"):
            if self.cam.IsGrabbing():
                self.cam.StopGrabbing()
        self.cam = None
        logging.info("Closed Basler camera connection".ljust(65, ".") + "[done]")


class HeliCam(Sensor):
    """Heliotis heliCam C3 sensor operating in raw readout mode.

    Parameters
    ----------
    configuration : dict[str, Any]
        Sensor configuration extending the common Sensor settings.

        Supported keys:

        * exposure_time: exposure in microseconds, converted to rounded camera
          clock cycles. Integer values avoid implying finer timing precision.
        * SensNavM2: required Heliotis-specific list of integers; see the
          manufacturer's documentation for its meaning.
        * number_measurements: even number of frames. Each native frame contains
          two subframes, which are flattened to the common sensor array format.

    Notes
    -----
    See `Heliotis <https://www.heliotis.com>`_ for camera and SDK documentation.
    """

    @close_on_initialization_error
    def __init__(self, configuration: dict[str, Any]) -> None:
        self._is_open = False
        if globals().get("heli") is None:
            raise ImportError("HeliCam requires the libHeLIC driver")
        self.settings = {
            "CamMode": 0,
            "SensNavM2": 6,
            "SensTqp": 24,
            "SensNFrames": 1,
            "BSEnable": 0,
            "DdsGain": 2,
            "TrigFreeExtN": 0,
            "ExtTqp": 1,
            "EnTrigOnPos": 0,
            "EnSynFOut": 0,
            "AcqStop": 0,
        }
        super().__init__(configuration)
        self.attribute_map["SensNavM2"] = self._set_SensNavM2
        self.attribute_map["exposure_time"] = self._set_SensTqp
        self.attribute_map["number_measurements"] = self._set_number_measurements
        self.initial_configuration_dict = configuration
        if configuration is not None:
            self._update_from_configuration(configuration)
        self.he_sys = heli.LibHeLIC()
        self.roi_shape = [300, 300]

    def __repr__(self) -> str:
        return f"HeliCam(configuration: {self.initial_configuration_dict})"

    def __str__(self) -> str:
        return f"HeliCam C3 camera instance: HeliCam(configuration: {self.initial_configuration_dict})"

    def _set_SensTqp(self, exposure_time: int) -> None:
        # Clockcyle of camera is 35 MHz -> 35 cycles per mus
        SensTqp = exposure_time * 35 - 11
        self.settings["SensTqp"] = int(SensTqp)

    def _set_SensNavM2(self, SensNavM2: int) -> None:
        self.settings["SensNavM2"] = int(SensNavM2)

    def _set_number_measurements(self, number_measurements: int) -> None:
        self.settings["SensNFrames"] = int(number_measurements / 2)
        self.number_measurements = int(number_measurements)

    def open(self) -> None:
        """Open the camera, load firmware, and prepare acquisition settings.

        Flush the camera buffer and initialize the camera attribute map.
        """
        if self._is_open:
            return
        try:
            self.he_sys.Open(0, sys="c3cam_sl70")
            self._is_open = True
            self.he_sys.SetTimeout(10000)
            deadline = monotonic() + 10
            res = 1
            while res > 0:
                if monotonic() >= deadline:
                    raise TimeoutError(
                        "HeliCam buffer flush did not finish within 10 seconds"
                    )
                res = self.he_sys.Acquire()
            for key, value in self.settings.items():
                setattr(self.he_sys.map, key, value)
            self.he_sys.AllocCamData(1, heli.LibHeLIC.CamDataFmt["DF_I16Q16"], 0, 0, 0)
        except BaseException:
            logging.exception("HeliCam setup failed")
            cleanup_after_error(self.close, "HeliCam setup cleanup")
            raise
        logging.info("Heli C3 opened".ljust(65, ".") + "[done]")

    @log_sensor_acquisition
    def acquire_data(self, synchroniser: Synchroniser | None = None) -> NDArray[Any]:
        """Acquire frames and flatten the heliCam subframe layout.

        See :meth:`Sensor.acquire_data` for the common acquisition interface.
        """
        time_1 = time()
        if synchroniser is not None:
            synchroniser.trigger()
        res = self.he_sys.Acquire()
        if res < 0:
            raise TimeoutError(
                f"HeliCam acquisition failed or timed out (status={res})"
            )
        _ = self.he_sys.ProcessCamData(1, 0, 0)
        meta = self.he_sys.CamDataMeta()
        img = self.he_sys.GetCamData(1, 0, heli.ct.byref(meta))
        data = img.contents.data
        data = heli.LibHeLIC.Ptr2Arr(
            data, (self.settings["SensNFrames"], 300, 300, 2), heli.ct.c_ushort
        )
        raw_frames = np.asarray((data), dtype="int32")
        return_frames = np.zeros((2 * self.settings["SensNFrames"], 300, 300))
        return_frames[::2] = raw_frames[:, :, :, 0]
        return_frames[1::2] = raw_frames[:, :, :, 1]
        time_2 = time()
        logging.debug(
            f"HeliCam C3 data acquisition took {time_2-time_1} s".ljust(65, ".")
            + "[done]"
        )
        return return_frames

    def close(self) -> None:
        """Close the camera connection.

        A new camera instance can be created after the connection is released.
        """
        if not getattr(self, "_is_open", False):
            return
        self.he_sys.Close()
        self._is_open = False
        logging.info("HeliCam C3 closed".ljust(65, ".") + "[done]")


class DAQ(Sensor):
    """National Instruments analog-input sensor using nidaqmx.

    Parameters
    ----------
    configuration : dict[str, Any]
        Sensor configuration extending the common Sensor settings.

        Supported keys:

        * number_measurements: number of samples acquired in one burst.
        * min_voltage and max_voltage: analog input limits in volts; consult the
          device manual for supported ranges.
        * apd_input: device and input port, defaulting to Dev3/ai0. The device
          identifier is assigned when the DAQ is connected.
        * sample_clk: sampling-clock input, defaulting to PFI1.
        * start_trig: acquisition start-trigger input, defaulting to PFI0.
        * max_samp_rate: maximum sampling rate in samples per second.

    Notes
    -----
    Requires the `nidaqmx <https://github.com/ni/nidaqmx-python/>`_ library and
    National Instruments drivers.
    """

    def __init__(self, configuration: dict[str, Any]) -> None:
        self.type: str = "DAQ"
        self.daq_task: Any = None
        if globals().get("nidaqmx") is None:
            raise ImportError("DAQ requires the nidaqmx driver")
        # configure various input ports ->
        # see inside NI-DAQ hardware for assignment.
        self.daq_apd_input: str = "Dev3/ai0"
        self.daq_sample_clk: str = "PFI1"
        self.daq_start_trig: str = "PFI0"
        self.daq_max_sampling_rate: float = 625e3
        # AI (analog input) in volts
        self.min_voltage: float = -1.0
        self.max_voltage: float = 1.0
        self.daq_timeout: int = 10  # / s
        super().__init__(configuration)
        self.roi_shape = [1]
        self.attribute_map["min_voltage"] = self._set_min_voltage
        self.attribute_map["max_voltage"] = self._set_max_voltage
        self.attribute_map["apd_input"] = self._set_apd_input
        self.attribute_map["sample_clk"] = self._set_sample_clk
        self.attribute_map["start_trig"] = self._set_start_trig
        self.attribute_map["max_samp_rate"] = self._set_max_sampling_rate
        self.initial_configuration_dict = configuration
        if configuration is not None:
            self._update_from_configuration(configuration)

    def __repr__(self) -> str:
        return f"DAQ(configuration: {self.initial_configuration_dict})"

    def __str__(self) -> str:
        return f"NI-DAQ sensor instance (nidaqmx): DAQ(configuration: {self.initial_configuration_dict})"

    def open(self) -> None:
        """Configure the analog input channel, sampling clock, and start trigger."""
        # Release an earlier task when opening the next pulse-sequence step.
        self.close()
        try:
            self.NsampsPerDAQread: float = self.number_measurements
            self.daq_task = nidaqmx.Task()
            self._create_analog_input_channel()
            self._configure_sample_clock()
            self._configure_analog_input_trigger()
            logging.info("DAQ opened and created read task".ljust(65, "."))
        except BaseException:
            logging.exception(
                "Failed to open or configure DAQ input %s", self.daq_apd_input
            )
            cleanup_after_error(self.close, "DAQ setup cleanup")
            raise

    def _create_analog_input_channel(self) -> None:
        # create analog channel to measure voltage
        _ = self.daq_task.ai_channels.add_ai_voltage_chan(
            self.daq_apd_input,
            "",
            TerminalConfiguration.RSE,
            self.min_voltage,
            self.max_voltage,
            VoltageUnits.VOLTS,
        )

    def _configure_analog_input_trigger(self) -> None:
        # Configure convert clock
        # Specifies the terminal of the signal to use
        # as the AI Convert Clock.
        self.daq_task.timing.ai_conv_src = self.daq_sample_clk
        self.daq_task.timing.ai_conv_active_edge = Edge.RISING
        read_start_trig = self.daq_task.triggers.start_trigger
        # Configures the task to start acquiring samples
        # on the active edge of a digital signal.
        read_start_trig.cfg_dig_edge_start_trig(self.daq_start_trig, Edge.RISING)

    def _configure_sample_clock(self) -> None:
        # Configure sample clock : Sets the clock source, the clock rate,
        # active clock edge, sample mode and the number of
        # samples to acquire.
        self.daq_task.timing.cfg_samp_clk_timing(
            self.daq_max_sampling_rate,
            self.daq_sample_clk,
            Edge.RISING,
            AcquisitionType.FINITE,
            self.NsampsPerDAQread,
        )

    @log_sensor_acquisition
    def acquire_data(self, synchroniser: Synchroniser | None = None) -> NDArray[Any]:
        """Read the configured number of analog samples in one burst.

        See :meth:`Sensor.acquire_data` for the common acquisition interface.
        """
        if self.daq_task is None:
            raise RuntimeError("DAQ must be opened before acquiring data")
        stage = "task start"
        try:
            with cleanup_on_exit(self.daq_task.stop, "DAQ acquisition stop"):
                if synchroniser is not None:
                    self.daq_task.start()
                    stage = "trigger"
                    synchroniser.trigger()
                stage = "read"
                samples = self.daq_task.read(self.NsampsPerDAQread, self.daq_timeout)
                stage = "task stop"
        except Exception:
            logging.exception(
                "DAQ acquisition failed during %s: input=%s, samples=%s, timeout=%ss",
                stage,
                self.daq_apd_input,
                self.NsampsPerDAQread,
                self.daq_timeout,
            )
            raise
        logging.debug("DAQ acquisition completed: samples=%s", len(samples))
        return np.asarray(samples).reshape((len(samples), 1))

    def _set_min_voltage(self, min_voltage: float) -> None:
        self.min_voltage = float(min_voltage)

    def _set_max_voltage(self, max_voltage: float) -> None:
        self.max_voltage = float(max_voltage)

    def _set_start_trig(self, start_trig: str) -> None:
        self.daq_start_trig = start_trig

    def _set_max_sampling_rate(self, max_sampling_rate: float) -> None:
        self.daq_max_sampling_rate = max_sampling_rate

    def _set_apd_input(self, apd_input: str) -> None:
        self.daq_apd_input = apd_input

    def _set_sample_clk(self, sample_clk: str) -> None:
        self.daq_sample_clk = sample_clk

    def close(self) -> None:
        """Close the DAQ task if a task is active."""
        if getattr(self, "daq_task", None) is None:
            return
        self.daq_task.close()
        self.daq_task = None
        logging.info("DAQ closed".ljust(65, "."))


class MockCam(Sensor):
    """Mock sensor that produces noise without communicating with hardware.

    Parameters
    ----------
    configuration : dict[str, Any]
        Sensor configuration extending the common Sensor settings.

        Supported keys:

        * image_roi: [height, width, x_offset, y_offset]. Only height and width
          affect the returned noise array.
    """

    def __init__(self, configuration: dict[str, Any]) -> None:
        super().__init__(configuration)
        self.roi_shape = [200, 200]
        self.attribute_map["image_roi"] = self._set_roi
        self.initial_configuration_dict = configuration
        if configuration is not None:
            self._update_from_configuration(configuration)

    def __repr__(self) -> str:
        return f"MockCam(configuration: {self.initial_configuration_dict})"

    def __str__(self) -> str:
        return f"MockCam sensor instance: DAQ(configuration: {self.initial_configuration_dict})"

    def _set_roi(self, roi_shape_and_offset: list[int]) -> None:
        """Extract the mock image dimensions from the ROI configuration.

        Parameters
        ----------
        roi_shape_and_offset : list[int]
            [height, width, x_offset, y_offset]. Offsets are ignored.
        """
        self.roi_shape = roi_shape_and_offset[:2]
        _ = roi_shape_and_offset[2:]

    def open(self) -> None:
        """Mark the mock sensor ready for acquisition."""
        logging.info("MockCam sensor ready")

    @log_sensor_acquisition
    def acquire_data(self, synchroniser: Synchroniser | None = None) -> NDArray[Any]:
        """Trigger the synchroniser and return Poisson-distributed mock images.

        Parameters
        ----------
        synchroniser : Synchroniser or None, optional
            Synchroniser to trigger, or None for acquisition without a synchroniser.

        Returns
        -------
        numpy.ndarray
            Noise with shape (number_measurements, height, width) and Poisson mean
            15,000, using the configured image dimensions.
        """
        if synchroniser is not None:
            synchroniser.trigger()
        noise = np.random.poisson(
            15_000, size=(self.number_measurements, *self.roi_shape)
        )
        return noise

    def close(self) -> None:
        """Record completion of mock sensor cleanup."""
        logging.info("MockCam sensor cleanup completed")
