# pylint: disable=logging-format-interpolation
# pylint: disable=logging-not-lazy
"""
Sensor Module handling the creation and usage of sensors.
"""
from __future__ import annotations
import sys
import os
import gc
import logging
import traceback
from time import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import ctypes

import numpy as np
from pypylon import pylon
from harvesters.core import Harvester

try:
    from egrabber import (
        EGenTL,
        Buffer,
        EGrabber,
        EGrabberDiscovery,
        BUFFER_INFO_BASE,
        INFO_DATATYPE_PTR,
        BUFFER_INFO_CUSTOM_PART_SIZE,
        BUFFER_INFO_CUSTOM_NUM_PARTS,
        INFO_DATATYPE_SIZET,
        BUFFER_INFO_TIMESTAMP,
        INFO_DATATYPE_UINT64,
    )
except ImportError:
    logging.warning(
        "Could not load egrabber library".ljust(65, ".")
        + "[failed]\nIf you are not using a Phantom S710 (or similar) camera you do not need this!"
    )

from qupyt.hardware.synchronisers import Synchroniser
from qupyt.mixins import ConfigurationMixin, UpdateConfigurationType, ConfigurationError

# Imports for HeliCam
if sys.platform == "win32":
    # from msvcrt import getch
    prgPath = os.environ["PROGRAMFILES"]
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
        Edge,
        AcquisitionType,
        VoltageUnits,
        TerminalConfiguration,
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
    """
    Sensor Factory responsible for creating and returning an instance of the
    requested sensor. All sensors created from this factory will adhere to
    an interface specified in :class:`Sensor`. This ensures, that all sensors
    can be used interchangeably in the measurement code.

    Methods:
        - :meth:`create_sensor`: This method creates the appropriate sensor
          instance and configures it. **Note for non programmers**: create_sensor
          is a static method. This means you don't have to create a class
          instance to call it.

    Example:
        >>> cam = SensorFactory.create_sensor('EoSense1.1CXP', {'number_measurements_referenced': 10})
    """

    @staticmethod
    def create_sensor(sensor_type: str, configuration: Dict[str, Any]) -> Sensor:
        """
        :param sensor_type: Sensor model identifier e.g. 'Baserl1920'.
        :type sensor_type: string
        :param configuration: configuration parameters for the sensor.
         Provided parameters need to match those available for the sensor.
         Please check the specific sensors for more information.
        :type configuration: dict
        :return: Instance of the requested sensor.
        :rtype: Sensor
        :raises ValueError:
        """
        try:
            if sensor_type == "Basler1920":
                return BaslerCam(configuration)
            if sensor_type == "EoSense1.1CXP":
                return GenICamHarvester(configuration)
            if sensor_type == "PhantomS710":
                return GenICamPhantom(configuration)
            if sensor_type == "HeliC3":
                return HeliCam(configuration)
            if sensor_type == "DAQ":
                return DAQ(configuration)
            if sensor_type == "MockCam":
                return MockCam(configuration)
            raise ValueError(
                f"Requested sensor type {sensor_type} does not exists")
        except Exception as exc:
            logging.exception(
                "Could not open desired camera".ljust(65, ".") + "[failed]"
            )
            traceback.print_exc()
            raise exc


class Sensor(ABC, ConfigurationMixin):
    """
    Abstract Base Class for all sensors. All sensors implemented in QuPyt
    should inherit from this class. This helps ensure compliance with the
    sensor API.

    Note:
        The attributes listed below are never explicitly set by the user.
        Please use the ``configuration`` constructor argument to configure the sensor.

    Arguments:
        - **configuration** (dict): Configuration dictionary. Keys will be used
          to select setter methods from an attribute map dicionary to set
          associated values.

          Possible configuration values:
            - **number_measurements** (int, even): Set the number of
              images or measurements the sensor will acquire in one go.

          Concrete sensor classes may have additional configuration values.

    Attributes:
        - **roi_shape** (list[int]): Specifies the shape (Region Of Interest)
          thats read out from the sensor. This attribute is present to
          communicate the sensor dimensions to other parts of the code.
          It should not be actively set by the user.
          Check the configuration options for the individual sensors.
        - **target_data_type** (int): Datatype returned by the sensor.
          It should match or at least be compatible with the datatype of the
          data container. Not settable for all sensors.
        - **number_measurements** (int): Set by the number_measurements
          attribute in the configuration dict.
    """

    attribute_map: UpdateConfigurationType

    def __init__(
        self, configuration: Dict[str, Any]
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
        """
        Implements the discovery and establishes the connection
        to a sensor.
        """

    @abstractmethod
    def acquire_data(self, synchroniser: Optional[Synchroniser]) -> np.ndarray:
        """
        Returns measurements in an array of shape
        [self.number_measurements, \\*self.roi_shape].

        :param synchroniser: Synchronisers instance.
         Sensor and Synchroniser are in most cases the devices that
         need the most amount of configuration. To avoid problems
         with one device beeing ready first, the synchronisers
         is prepared first. Next the sensor is made ready. Once
         ready, the sensor sends a trigger signal to the
         synchronisers to kick-off the measurement.
        :type synchroniser: Optional[Synchroniser]
        """

    @abstractmethod
    def close(self) -> None:
        """
        Closes and if necessary destroys the sensor instance.
        """


class GenICamPhantom(Sensor):
    """
    Sensor class implementation for the PhantomS710 GenICam compliant camera.
    It builds on the EGrabber library provided by Euresys.
    Therefore, it will only work with the Framegrabber / Camera combination.

    Note:
        You will need to install the egrabber software and the egrabber python
        wheel provided on the Euresys website for this class to work.

    Arguments:
        - **configuration** (dict): Configuration dictionary. Keys will be used
          to select setter methods from an attribute map dicionary to set
          associated values.

          Possible configuration values:
            - **exposure_time** (int, µs)
            - **image_roi** (list[int]):
              **The S710 only supports size adjustment, but does not support a roi that is off-center
              Region Of Interest of the sensor in
              the following format: [height, width, x_offset, y_offset]. **offset is left but ignored for now.
              The code is not removed to ensure it is better reusable for future devices and nothing else breaks** 
              The roi_shape attribute of the :class:`Sensor` base class will
              be derived from this.
            - **pixel_bits** (string): Sets number of bits per pixel.
              E.g. 'Mono8', 'Mono12' or 'Mono16'.
              Note that we currently do not support the "bayer" pixel format.
            - **trigger_mode** (string): Sets camera in triggered mode.
              ['On', 'on', 'ON', 'Off', 'off', "OFF"]
            - **trigger_source** (string): Sets input for the camera trigger.
              E.g. 'GPIO0', 'GPIO1'

          Note that these configuration attributes extend those from the
          :class:`Sensor` base class.

    Raises (__init__):

        - ConfigurationError
    """

    def __init__(self, configuration: Dict[str, Any]) -> None:
        self.cam, self.cam_instance = self._discover_and_setup()
        super().__init__(configuration)
        # self.cam.remote_device.node_map.OffsetX.value = 0
        # self.cam.remote_device.node_map.OffsetY.value = 0
        self.cam.remote.set("Height", 200)
        self.cam.remote.set("Width", 1280)
        self.cam.remote.set("PixelFormat", "Mono8")
        self.image_dtype = np.uint8
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

    def _discover_and_setup(self) -> EGrabber:
        gentl = EGenTL()
        discovery = EGrabberDiscovery(gentl)
        discovery.discover()
        cam = discovery.cameras[0]
        self.grabber = EGrabber(cam)
        return self.grabber, cam

    def _set_trigger_source(self, trigger_source: str) -> None:
        """e.g. GPIO0, GPIO1, ..."""
        self.cam.remote.set("TriggerSource", trigger_source)

    def _set_trigger_mode(self, trigger_mode: str) -> None:
        """
        e.g. On of Off

        Raises:
            - ConfigurationError
        """
        if trigger_mode.lower() == "on":
            self.cam.remote.set("TriggerMode", "TriggerModeOn")
        elif trigger_mode.lower() == "off":
            self.cam.remote.set("TriggerMode", "TriggerModeOff")
        else:
            raise ConfigurationError(
                "trigger_mode", trigger_mode, ["on", "off"])

    def _set_pixel_bits(self, pixel_bits: str) -> None:
        """
        e.g. Mono8, Mono12 or Mono16

        Raises:
            - ConfigurationError
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

    def _set_roi(self, roi_shape_and_offset: List[int]) -> None:
        roi_shape_h_and_w = roi_shape_and_offset[:2]
        if roi_shape_and_offset[2] or roi_shape_and_offset[3]:
            logging.warning(f"Offset X:{roi_shape_and_offset[2]}+Y:{roi_shape_and_offset[3]} set, but the S710 does not support ROI offset.\n Consider resizing the ROI to get the Region in a center-focused way\n".ljust(
                    65, "."
                )
                + "[warning]" )
        #this code checks if the sizes are okay on a per-sensor-basis and not in the total camera size, so the divisison is performed here
        if roi_shape_h_and_w[0] % 8 != 0:
            raise Exception("Picture Size has to be a multiple of 8")

        
        roi_shape_h_and_w[0] = int(np.round(roi_shape_h_and_w[0] / 4))
        #ensure the roi has a compliant size with the specification.
        if roi_shape_h_and_w[1] % 128 != 0:
            raise Exception(f"ROI Width for the S710 has to be a multiple of 128px; {roi_shape_h_and_w[1]}px was specified")
        if roi_shape_h_and_w[1] < 128:
            raise Exception(f"ROI Width for the S710 has to at least 128px; {roi_shape_h_and_w[1]}px was specified")
        if roi_shape_h_and_w[1] > 1280:
            raise Exception(f"ROI Width for the S710 has to at most 1280px; {roi_shape_h_and_w[1]}px was specified")

        if roi_shape_h_and_w[0] > 200:
            raise Exception(f"ROI Height for the S710 has to at most 800px; {roi_shape_h_and_w[0]*4}px  ({roi_shape_h_and_w[0]}px per Sensor) was specified")
        if roi_shape_h_and_w[0] < 8:
            raise Exception(f"ROI Height for the S710 has to at least 32px; {roi_shape_h_and_w[0]*4}px  ({roi_shape_h_and_w[0]}px per Sensor) was specified")

        try:
            self.cam.remote.set("Height", roi_shape_h_and_w[0])
            self.cam.remote.set("Width", roi_shape_h_and_w[1])
            height_real = self.cam.remote.get("Height")
            width_real = self.cam.remote.get("Width")
            logging.info(
            f"Real camera height: {height_real} and width: {width_real}\n".ljust(
                    65, "."
                )
                + "[done]"
            )

            ### testing has shown, that .9*x AFR.Max is more stable than 1.0x or .95x 
            ### This line has to be used if the Framerate must be increased in the code.
            ### However, it is removed since it is likley unneeded. In case of reintroduction, see:
            ### https://github.com/KarDB/QuPyt/pull/33#discussion_r2239747736 ###
            # self.grabber.remote.set("AcquisitionFrameRate",0.9*int(self.grabber.remote.get("AcquisitionFrameRate.Max")))
            self.roi_shape = [roi_shape_h_and_w[0]*4, roi_shape_h_and_w[1]]
            logging.info(
                f"Set Sensor roi to height: {roi_shape_h_and_w[0]*4} and width: {roi_shape_h_and_w[1]}\n".ljust(
                    65, "."
                )
                + "[done]"
            )
        except Exception as exc:
            logging.exception(
                f"Failed to set roi of height: {roi_shape_h_and_w[0]*4} and width: {roi_shape_h_and_w[1]}\n".ljust(
                    65, "."
                )
                + "[failed]"
            )
            raise exc

    def acquire_data(self, synchroniser: Optional[Synchroniser] = None) -> np.ndarray:
        """
        See :meth:`Sensor.acquire_data`.
        """
        time_1 = time()
        height = self.cam.remote.get("Height")
        width = self.cam.remote.get("Width")

        logging.info(f"Sizes are {height}, {width}, {self.roi_shape} .".ljust(
                65, ".") + "[info]")
        self.cam.realloc_buffers(self.number_measurements)
        self.cam.start()
        data = np.zeros((self.number_measurements,
                        height * 4, width), dtype=np.uint32)
        timesteps = np.zeros(self.number_measurements, dtype=np.float64)
        if synchroniser is not None:
            synchroniser.trigger()
        for i in range(self.number_measurements):
            with Buffer(self.cam) as buffer:
                buffer_ptr, image_size, part_num, timestep = self._grab_frame_info(
                    buffer)
                raw_frame = self._move_frame_from_pool(
                    buffer_ptr, image_size, part_num)
                data[i] += raw_frame
                timesteps[i] += timestep
        self.cam.stop()
        time_2 = time()

        logging.info(f"Framerate is {1/((timesteps[self.number_measurements-1]-timesteps[0])*1e-6/(self.number_measurements-1))} Hz".ljust(
                65, ".") + "[info]")
        logging.info(f"Timesteps are {timesteps-timesteps[0]} .".ljust(
                65, ".") + "[info]")
        logging.info(
            f"Data acquisition took {time_2-time_1} s".ljust(
                65, ".") + "[done]"
        )
        return data

    def _grab_frame_info(self, buffer: Buffer):
        buffer_ptr = buffer.get_info(BUFFER_INFO_BASE, INFO_DATATYPE_PTR)
        image_size = buffer.get_info(
            BUFFER_INFO_CUSTOM_PART_SIZE, INFO_DATATYPE_SIZET)
        part_num = buffer.get_info(
            BUFFER_INFO_CUSTOM_NUM_PARTS, INFO_DATATYPE_SIZET)
        time_stamp = buffer.get_info(
            BUFFER_INFO_TIMESTAMP, INFO_DATATYPE_UINT64)

        return buffer_ptr, image_size, part_num, time_stamp

    def _move_frame_from_pool(self, buffer_ptr, image_size, part_num) -> np.array:
        frame = np.empty(
            (part_num * self.roi_shape[0], self.roi_shape[1]), dtype=self.image_dtype
        )
        frame_ptr = frame.ctypes.data_as(ctypes.c_void_p)
        ctypes.memmove(frame_ptr, buffer_ptr, image_size * part_num)
        return frame

    def open(self) -> None:
        """
        This function simply passes, since all necessary setup happens
        in __init__. This is necessary to configure the camera.
        """

    def close(self) -> None:
        """
        Destroys the camera instance.
        Without this, you won't be able to make a new camera instance,
        as the camera will be exclusively owned by this one.
        """
        self.cam = None
        self.cam_instance = None
        self.grabber = None
        gc.collect()
        logging.info("Closed GenICam camera connection".ljust(
            65, ".") + "[done]")


class GenICamHarvester(Sensor):
    """
    Sensor class implementation for all simple GenICam compliant cameras.
    This class relies on the excellent
    `harvesters <https://github.com/genicam/harvesters>`_ library.
    You will need to provide your own GenTL producer file for this to work.
    Check the recommendations `here <https://github.com/genicam/harvesters>`_
    or get it from your GenICam compliant camera manufacturer.

    Arguments:
        - **configuration** (dict): Configuration dictionary. Keys will be used
          to select setter methods from an attribute map dicionary to set
          associated values.

          Possible configuration values:
            - **exposure_time** (int, µs)
            - **image_roi** (list[int]): Region Of Interest of the sensor in
              the following format: [height, width, x_offset, y_offset].
              The roi_shape attribute of the :class:`Sensor` base class will
              be derived from this.
            - **GenTL_producer_cti** (string): Path to the GenTL producer (cti)
              file on your computer.

          Note that these configuration attributes extend those from the
          :class:`Sensor` base class.

    Note:
        Currently this class preconfigures the following:
          - **CXP link configuration**: CXP12_X4. This means the frame grabber expects
            to communicate via CoaXPress 12 on 4 lanes.
          - **pixel_format**: Mono10
          - **trigger_source**: Line0
          - **trigger_mode**: FrameStart

    Raises (__init__):
        - **FileNotFoundError**
    """

    def __init__(self, configuration: Dict[str, Any]) -> None:
        self.harvester = Harvester()
        self.cti_file = configuration["GenTL_producer_cti"]
        try:
            self.harvester.add_file(self.cti_file)
        except (FileNotFoundError, OSError) as err:
            logging.exception(
                "Could not find GenTL producer file (.cti)".ljust(
                    65, ".") + "[failed]"
            )
            logging.exception(f"Tried to locate file at {self.cti_file}")
            raise err
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
        """cti must be set at the start. It is therefore ignored here."""

    def _set_exposure_time(self, exposure_time: int) -> None:
        self.cam.remote_device.node_map.ExposureTime.value = exposure_time

    def _set_gain(self, gain: int) -> None:
        self.cam.remote_device.node_map.Gain.value = gain

    def _set_roi(self, roi_shape_and_offset: List[int]) -> None:
        """
        Set the region of interest (ROI) on the camera sensor.

        This method configures the ROI on the camera sensor by setting the height, width,
        and the X and Y offsets. The ROI shape and offsets are specified in a list.

        Args:
            roi_shape_and_offset (List[int]): A list containing four integers:
                - The first two integers specify the height and width of the ROI.
                - The last two integers specify the X and Y offsets of the ROI.

        Raises:
            Exception: If setting any of the ROI parameters on the camera fails, an exception is logged and raised.

        Example:
            roi_shape_and_offset = [height, width, offsetX, offsetY]
            _set_roi(roi_shape_and_offset)

        The method logs the outcome of setting the ROI:
            - On success: Logs the height, width, and offsets along with a "[done]" status.
            - On failure: Logs the height, width, and offsets along with a "[failed]" status and raises the exception.
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
        except Exception as exc:
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
            raise exc

    def acquire_data(self, synchroniser: Optional[Synchroniser] = None) -> np.ndarray:
        """
        See :meth:`Sensor.acquire_data`.
        """
        time_1 = time()
        height = self.cam.remote_device.node_map.Height.value
        width = self.cam.remote_device.node_map.Width.value
        self.cam.start()
        data = np.zeros((self.number_measurements,
                        height * width), dtype=np.uint32)
        if synchroniser is not None:
            synchroniser.trigger()
        for i in range(self.number_measurements):
            with self.cam.fetch() as buffer:
                component = buffer.payload.components[0]
                data[i] = component.data
        self.cam.stop()
        time_2 = time()
        logging.info(
            f"Data acquisition took {time_2-time_1} s".ljust(
                65, ".") + "[done]"
        )
        return data.reshape((self.number_measurements, height, width))

    def open(self) -> None:
        """
        This function simply passes, since all necessary setup happens
        in __init__. This is necessary to configure the camera.
        """

    def close(self) -> None:
        """
        Destroys the camera instance, and resets Harvester.
        Without this, you won't be able to make a new camera instance,
        as the camera will be exclusively owned by this one.
        """
        self.cam.destroy()
        self.harvester.reset()
        logging.info("Closed GenICam camera connection".ljust(
            65, ".") + "[done]")


class BaslerCam(Sensor):
    """
    Sensor class implementation for all Basler Mono cameras.
    While (most or all) Basler cameras are fully GenICam compliant
    and will work with the :class:`GenICam_Harvester` class, some
    users might prefer to use `pypylon <https://github.com/basler/pypylon>`_,
    which is provieded by Basler direclty. Note that Baser strongly
    recommends you install
    `pylon <https://www.baslerweb.com/en/products/basler-pylon-camera-software-suite/>`_
    from their website.

    Arguments:
        - **configuration** (dict): Configuration dictionary. Keys will be used
          to select setter methods from an attribute map dicionary to set
          associated values.

          Possible configuration values:
            - **exposure_time** (int, µs)
            - **image_roi** (list[int]): Region Of Interest of the sensor in
              the following format: [height, width, x_offset, y_offset].
              The roi_shape attribute of the :class:`Sensor` base class will
              be derived from this.
            - **binning_horizontal** (string): Options typically are 1, 2, 3 or 4.
              However, this might depend on your specific camera model.
            - **binning_vertical** (string): Options are typically 1, 2, 3 or 4.
              However, this might depend on your specific camera model.
            - **binning_mode_horizontal** (string): Options are `'sum'` or `'average'`
            - **binning_mode_vertical** (string): Options are `'sum'` or `'average'`

          Note that these configuration attributes extend those from the
          :class:`Sensor` base class.
    """

    def __init__(self, configuration: Dict[str, Any]) -> None:
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

    def acquire_data(self, synchroniser: Optional[Synchroniser] = None) -> np.ndarray:
        """
        See :meth:`Sensor.acquire_data`.
        """
        time_1 = time()
        arr = np.zeros((self.number_measurements, *self.roi_shape))
        self.cam.StartGrabbingMax(self.number_measurements)
        if synchroniser is not None:
            synchroniser.trigger()
        for i in range(self.number_measurements):
            grab_result = self.cam.RetrieveResult(
                int(5000), pylon.TimeoutHandling_ThrowException
            )
            frame = np.asarray(grab_result.Array)
            arr[i] = frame
            grab_result.Release()
        time_2 = time()
        logging.info(
            f"Basler data acquisition took {time_2-time_1} s".ljust(
                65, ".") + "[done]"
        )
        return arr

    def _set_trigger_line(self, trigger_line):
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
            raise ConfigurationError(
                "trigger_mode", trigger_mode, ["on", "off"])

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

    def _set_roi(self, roi_shape_and_offset: List[int]) -> None:
        """
        Set the region of interest (ROI) on the camera sensor.

        This method configures the ROI on the camera sensor by setting the height, width,
        and the X and Y offsets. The ROI shape and offsets are specified in a list.

        Args:
            roi_shape_and_offset (List[int]): A list containing four integers:
                - The first two integers specify the height and width of the ROI.
                - The last two integers specify the X and Y offsets of the ROI.

        Raises:
            Exception: If setting any of the ROI parameters on the camera fails, an exception is logged and raised.

        Example:
            roi_shape_and_offset = [height, width, offsetX, offsetY]
            _set_roi(roi_shape_and_offset)

        The method logs the outcome of setting the ROI:
            - On success: Logs the height, width, and offsets along with a "[done]" status.
            - On failure: Logs the height, width, and offsets along with a "[failed]" status and raises the exception.
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
        except Exception as exc:
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
            raise exc

    def _set_roi_shape(self, roi_shape_h_and_w: List[int]) -> None:
        self.cam.Height.SetValue(roi_shape_h_and_w[0])
        self.cam.Width.SetValue(roi_shape_h_and_w[1])

    def _set_roi_offset_x_and_y(self, roi_offset_x_and_y: List[int]) -> None:
        self.cam.OffsetX.SetValue(roi_offset_x_and_y[0])
        self.cam.OffsetY.SetValue(roi_offset_x_and_y[1])

    def open(self) -> None:
        """Opens camera."""
        logging.info("Opening Balser".ljust(65, ".") + "[done]")

    def close(self) -> None:
        """Closes the the camera.
        A new camera instance may now be created."""
        self.cam.Close()
        logging.info("Closed Basler camera connection".ljust(
            65, ".") + "[done]")


class HeliCam(Sensor):
    """
    `Heliotis <https://www.heliotis.com>`_ builds highly specialised lock-in
    cameras which are very powerfull for specific applications.
    This class focusses on the implementation of their heliCam C3 and its
    raw readout mode.

    Arguments:
        - **configuration** (dict): Configuration dictionary. Keys will be used
          to select setter methods from an attribute map dicionary to set
          associated values.

          Possible configuration values:
            - **exposure_time** (int/float, µs): This value will get converted
              to heliCam clock cycles. While float values are possible, and may
              provide more fine grained controll, the number of clock cycles will
              be rounded. Therefore, I advise to stick with integer values.
            - **SensNavM2** (list[int]): This is a special parameter used in
              Heliotis cameras. This parameters must be set, as it is vital to
              the operation of the camera. However, we refer to Helioti's
              documentation for the parameters explanation.
            - **number_measurements** (int, even): HeliCam overwrites
              the base :class:`Sensor` class's
              set_number_measurements logic, because every frame produced
              by the heliCam C3 contains two sub frames. We reshape this nested
              frame structure into a flat array of frames in accordance to all
              other camers. However, this implied that internally, the
              configuration for the number of frames works slighly differently.

          Note that these configuration attributes extend those from the
          :class:`Sensor` base class.
    """

    def __init__(self, configuration: Dict[str, Any]) -> None:
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
        """
        Opens the camera and loads the firmware.
        Flushes the camera buffer, and sets the cameras attribute map.
        """
        self.he_sys.Open(0, sys="c3cam_sl70")
        res = 1
        while res > 0:
            res = self.he_sys.Acquire()
        for key, value in self.settings.items():
            setattr(self.he_sys.map, key, value)
        self.he_sys.AllocCamData(
            1, heli.LibHeLIC.CamDataFmt["DF_I16Q16"], 0, 0, 0)
        self.he_sys.SetTimeout(10000)
        logging.info("Heli C3 opened".ljust(65, ".") + "[done]")

    def acquire_data(self, synchroniser: Optional[Synchroniser] = None) -> np.ndarray:
        """
        Reads frames from the camera and reshuffels from the
        heliCam specific format to the structure
        we expect from other sensors.

        See :meth:`Sensor.acquire_data`.
        """
        time_1 = time()
        if synchroniser is not None:
            synchroniser.trigger()
        res = self.he_sys.Acquire()
        if res < 0:
            logging.warning(
                "Your camera appears to have timed out".ljust(
                    65, ",") + "[done]"
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
        logging.info(
            f"HeliCam C3 data acquisition took {time_2-time_1} s".ljust(
                65, ".")
            + "[done]"
        )
        return return_frames

    def close(self) -> None:
        """Closes the the camera.
        A new camera instance may now be created."""
        self.he_sys.Close()
        logging.info("HeliCam C3 closed".ljust(65, ".") + "[done]")


class DAQ(Sensor):
    """
    This class interfaces with Data Acquisition Units from National Instruments.
    It specifically focusses on devices that use the
    `nidaqmx <https://github.com/ni/nidaqmx-python/>`_ library.

    Arguments:
        - **configuration** (dict): Configuration dictionary. Keys will be used
          to select setter methods from an attribute map dicionary to set
          associated values.

          Possible configuration values:
            - **number_measurements** (int, even): HeliCam overwrites
              the base :class:`Sensor` class's
              set_number_measurements logic, because every frame produced
              by the heliCam C3 contains two sub frames. We reshape this nested
              frame structure into a flat array of frames in accordance to all
              other camers. However, this implied that internally, the
              configuration for the number of frames works slighly differently.
            - **min_voltage** (float): Mininmum voltage level for the DAQ to digitise.
              For the specific range of your device please consult your manual.
            - **max_voltage** (float): Maximum voltage level for the DAQ to digitise.
              For the specific range of your device please consult your manual.
            - **apd_input** (str): Descriptor of the device and input port to use.
              Default is "Dev3/ai0". The Dev3 part is negotiated when you plug in
              your device. Please consult e.g. the NI software to find this value.
              the "ai0" selects the channel on your device which you will read from.
            - **sample_clk** (str): Selects the channel on the DAQ board used to
              time acquisition. e.g. (default) "PFI1". PFI stands for Programmable
              Function Interface.
            - **start_trig** (str): Selects the channel on the DAQ board to recieve
              a trigger to start the measurement process ('default "PFI0").
            - **max_samp_rate** (float): Sets the max sampling rate of the DAQ device.

          Note that these configuration attributes extend those from the
          :class:`Sensor` base class.
    """

    def __init__(self, configuration: Dict[str, Any]) -> None:
        self.type: str = "DAQ"
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
        """
        Configures an analog input voltage channel,
        sample clock and start trigger.
        """
        try:
            self.NsampsPerDAQread: float = self.number_measurements
            self.daq_task = nidaqmx.Task()
            self._create_analog_input_channel()
            self._configure_sample_clock()
            self._configure_analog_input_trigger()
            logging.info("DAQ opened and created read task".ljust(65, "."))
        except Exception:
            self.close()
            logging.exception("An error occured while trying opening the DAQ")
            traceback.print_exc()
            raise Exception(
                "An error Occured while opening and configureing the DAQ.\
                        Please make sure the DAQ is connected and powered on"
            )

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
        read_start_trig.cfg_dig_edge_start_trig(
            self.daq_start_trig, Edge.RISING)

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

    def acquire_data(self, synchroniser: Optional[Synchroniser] = None) -> np.ndarray:
        """
        Reads all samples at onec from the configure sensor.

        See :meth:`Sensor.acquire_data`.
        """
        if synchroniser is not None:
            self.daq_task.start()
            synchroniser.trigger()
        try:
            samples = self.daq_task.read(
                self.NsampsPerDAQread, self.daq_timeout)
            self.daq_task.stop()
        except Exception as excpt:
            print(
                """Error: could not read DAQ.
                Please check your DAQ's connections.
                Exception details:""",
                type(excpt).__name__,
                ".",
                excpt,
            )
            sys.exit()
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
        """
        See :meth:`Sensor.close`
        """
        self.daq_task.close()
        logging.info("DAQ closed".ljust(65, "."))


class MockCam(Sensor):
    """
    To ensure users can test their code and the general logic without having
    to buy expensive lab equipment, most devices in QuPyt implement
    mocking behoviour. Mock devices do not talk to any real hardware.
    They implement a dummy interface, where API functions either pass,
    sleep or return noise.
    This is the mock variant of a sensor.

    Arguments:
        - **configuration** (dict): Configuration dictionary. Keys will be used
          to select setter methods from an attribute map dicionary to set
          associated values.

          Possible configuration values:
            - **image_roi** (list[int]): Region Of Interest of the sensor in
              the following format: [height, width, x_offset, y_offset].
              The roi_shape attribute of the :class:`Sensor` base class will
              be derived from this.

          Possible configuration values:

          Note that these configuration attributes extend those from the
          :class:`Sensor` base class.
    """

    def __init__(self, configuration: Dict[str, Any]) -> None:
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

    def _set_roi(self, roi_shape_and_offset: List[int]) -> None:
        """
        This function emulates the current behaviour of the GenICam and Basler camera sensor.
        ROI values are extracted from combined input of the ROI and offset values.
        """
        self.roi_shape = roi_shape_and_offset[:2]
        _ = roi_shape_and_offset[2:]

    def open(self) -> None:
        """
        passes since there is no device to open.
        """

    def acquire_data(self, synchroniser: Optional[Synchroniser] = None) -> np.ndarray:
        """
        Sends tigger signal to synchroniser if there is one.
        Returns an array of shape ``[number_measrurements, height, witdh]``
        as specified in configuration. Array contains Poisson distributed values
        with k=15000

        See :meth:`Sensor.acquire_data`.
        """
        if synchroniser is not None:
            synchroniser.trigger()
        noise = np.random.poisson(
            15_000,
            size=(self.number_measurements, *self.roi_shape),
        )
        return noise

    def close(self) -> None:
        """
        Passes since there is no device to close.
        """
