import logging
from typing import Dict, Any, List
import numpy as np
from qupyt.hardware.sensors import Sensor
from qupyt.mixins import ConfigurationMixin, UpdateConfigurationType


class Data(ConfigurationMixin):
    attribute_map: UpdateConfigurationType

    def __init__(self, configuration: Dict[str, Any]) -> None:
        self.roi_shape: list[int]
        self.averaging_mode: str
        self.number_dynamic_steps: int
        self.number_measurements: int
        self.data_type: type
        self.compress: bool = False
        self.live_compression: bool = False
        self.save_in_chunks: int = 0
        self.reference_channels: int=2
        self.data: np.ndarray
        self.attribute_map = {
            "dynamic_steps": self._set_number_dynamic_steps,
            "averaging_mode": self._set_averaging_mode,
            "number_measurements": self._set_number_measurements,
            "roi_shape": self._set_roi_shape,
            "compress": self._set_compress_mode,
            "live_compression": self._set_live_compression,
            "save_in_chunks": self._set_save_chunk_size,
            "reference_channels": self._set_reference_channels
        }
        self._update_from_configuration(configuration)

    def _set_save_chunk_size(self, save_in_chunks: int) -> None:
        self.save_in_chunks = save_in_chunks
    
    def _set_reference_channels(self, reference_channels: int) -> None:
        self.reference_channels = int(reference_channels)

    def _set_compress_mode(self, compression_value: bool) -> None:
        self.compress = compression_value

    def _set_live_compression(self, live_compression: bool) -> None:
        self.live_compression = live_compression

    def _set_dtype_from_sensor(self, sensor: Sensor) -> None:
        self.data_type = sensor.target_data_type

    def _set_number_dynamic_steps(self, number_dynamic_steps: int) -> None:
        self.number_dynamic_steps = int(number_dynamic_steps)

    def _set_averaging_mode(self, averaging_mode: str) -> None:
        self.averaging_mode = str(averaging_mode)

    def _set_number_measurements(self, number_measurements: int) -> None:
        self.number_measurements = number_measurements

    def _set_roi_shape(self, roi_shape: List[int]) -> None:
        self.roi_shape = roi_shape

    def set_dims_from_sensor(self, sensor: Sensor) -> None:
        self._set_ROI_from_sensor(sensor)
        self._set_number_measurements_from_sensor(sensor)
        if hasattr(sensor, "target_data_type"):
            self._set_dtype_from_sensor(sensor)

    def _set_ROI_from_sensor(self, sensor: Sensor) -> None:
        self.roi_shape = sensor.roi_shape

    def _set_number_measurements_from_sensor(self, sensor: Sensor) -> None:
        self.number_measurements = sensor.number_measurements

    def create_array(self) -> None:
        # HeliCam gets number_images, whereas other get number_images / 2.
        # I suggest fixing this as number_images
        # Specifics have then to be dealt with in each class.
        if self.live_compression:
            self.roi_shape = [1]
        if self.averaging_mode == "sum":
            data_array_dim = [self.reference_channels, self.number_dynamic_steps, 1, *self.roi_shape]
        elif self.averaging_mode == "spread":
            data_array_dim = [
                self.reference_channels,
                self.number_dynamic_steps,
                int(self.number_measurements / 2),
                *self.roi_shape,
            ]
        else:
            logging.info(
                f"Failed to crearte dara array for mode {self.averaging_mode}".ljust(
                    65, "."
                )
                + "[failed]"
            )
            raise ValueError(f"averaging_mode {self.averaging_mode} not available")
        logging.info(
            f"Created data array of shape {data_array_dim}".ljust(65, ".") + "[done]"
        )
        self.data = np.zeros(data_array_dim, dtype=getattr(self, "data_type", float))

    def update_data(self, data: np.ndarray, dynamic_step: int, avg_step: int) -> None:
        if self.save_in_chunks != 0 and avg_step % self.save_in_chunks == 0:
            self.save(f"save_chunk_{avg_step}.npy")
            self.create_array()
        if self.live_compression:
            self._update_data_compressed(data, dynamic_step)
        else:
            self._update_data_full(data, dynamic_step)

    def _update_data_full(self, data: np.ndarray, dynamic_step: int) -> None:
        if self.averaging_mode == "sum":
            for i in range(self.reference_channels):
                self.data[i, dynamic_step] += data[i::self.reference_channels].sum(axis=0)
            np.save("C:/Users/ge54vec/.qupyt/data", self.data)
        elif self.averaging_mode == "spread":
            for i in range(self.reference_channels):
                self.data[i, dynamic_step] += data [i::self.reference_channels]
                np.save("C:/Users/ge54vec/.qupyt/data", self.data)


    def _update_data_compressed(self, data: np.ndarray, dynamic_step: int) -> None:
        if self.averaging_mode == "sum":
            self.data[0, dynamic_step] += data[::2].mean(axis=(1, 2)).sum(axis=0)
            self.data[1, dynamic_step] += data[1::2].mean(axis=(1, 2)).sum(axis=0)
        elif self.averaging_mode == "spread":
            self.data[0, dynamic_step] += data[::2].mean(axis=(1, 2)).reshape(-1, 1)
            self.data[1, dynamic_step] += data[1::2].mean(axis=(1, 2)).reshape(-1, 1)

    def save(self, filename: str) -> None:
        """
        Save data (without metadata) to a .npy file.
        :param filename: Name of the resulting data file.
         During normal usage as part of QuPyt, this will be assigned by the
         main measurement loop.
        :type filename: str
        """
        if self.compress:
            np.save(filename, self.data.mean(axis=(3, 4)))
        else:
            np.save(filename, self.data)
