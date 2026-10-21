from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from qupyt._version import __version__ as qupyt_version
from qupyt.hardware.sensors import Sensor
from qupyt.mixins import ConfigurationMixin, UpdateConfigurationType
from qupyt.utils.errors import record_failure


class Data(ConfigurationMixin):
    attribute_map: UpdateConfigurationType

    def __init__(self, configuration: dict[str, Any]) -> None:
        self.roi_shape: list[int]
        self.averaging_mode: str
        self.number_dynamic_steps: int
        self.number_measurements: int
        self.data_type: type
        self.live_compression: bool = False
        self.number_pulse_sequences: int = 1
        self.save_in_chunks: int = 0
        self.reference_channels: int = 2
        self.data: NDArray[Any]
        self.chunk_data: NDArray[Any]
        self.chunk_save_index: int = 0
        self.filename: tuple[str, str]
        self.attribute_map = {
            "dynamic_steps": self._set_number_dynamic_steps,
            "ps_steps": self._set_number_pulse_sequences,
            "averaging_mode": self._set_averaging_mode,
            "number_measurements": self._set_number_measurements,
            "roi_shape": self._set_roi_shape,
            "compress": self._set_compress_mode,
            "live_compression": self._set_live_compression,
            "save_in_chunks": self._set_save_chunk_size,
            "reference_channels": self._set_reference_channels,
        }
        self._update_from_configuration(configuration)

    def _set_save_chunk_size(self, save_in_chunks: int) -> None:
        self.save_in_chunks = save_in_chunks

    def _set_reference_channels(self, reference_channels: int) -> None:
        self.reference_channels = int(reference_channels)

    def _set_compress_mode(self, compression_value: bool) -> None:
        logging.warning(
            "The 'compress' parameter is deprecated; use 'live_compression' instead"
        )
        self.live_compression = compression_value

    def _set_live_compression(self, live_compression: bool) -> None:
        self.live_compression = live_compression

    def _set_dtype_from_sensor(self, sensor: Sensor) -> None:
        self.data_type = sensor.target_data_type

    def _set_number_dynamic_steps(self, number_dynamic_steps: int) -> None:
        if number_dynamic_steps < 1:
            raise ValueError(
                """Cannot have fewer than 1 dynamic steps. This will result in an empty measurement.
            If you measure just one step, this is still ONE dynamic step. The first of one."""
            )
        self.number_dynamic_steps = int(number_dynamic_steps)

    def _set_number_pulse_sequences(self, number_pulse_sequences: int) -> None:
        self.number_pulse_sequences = int(number_pulse_sequences)

    def _set_averaging_mode(self, averaging_mode: str) -> None:
        self.averaging_mode = str(averaging_mode)

    def _set_number_measurements(self, number_measurements: int) -> None:
        self.number_measurements = number_measurements

    def _set_roi_shape(self, roi_shape: list[int]) -> None:
        self.roi_shape = roi_shape

    def _set_filename(self, filename: tuple[str, str]) -> None:
        self.filename = filename

    def set_filename(self, filename: tuple[str, str]) -> None:
        self._set_filename(filename)

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
        if (
            min(
                self.reference_channels,
                self.number_pulse_sequences,
                self.number_dynamic_steps,
                self.number_measurements,
            )
            < 1
        ):
            raise ValueError("Data dimensions and reference_channels must be positive")
        if self.save_in_chunks < 0:
            raise ValueError("save_in_chunks must be zero or a positive chunk size")
        self._input_roi_shape = tuple(self.roi_shape)
        if self.live_compression:
            self.roi_shape = [1]
            self.data_type = float
        if self.averaging_mode == "sum":
            data_array_dim = [
                self.reference_channels,
                self.number_pulse_sequences,
                self.number_dynamic_steps,
                1,
                *self.roi_shape,
            ]
        elif self.averaging_mode == "spread":
            measurements_per_channel = (
                self.number_measurements / self.reference_channels
            )
            if measurements_per_channel != int(measurements_per_channel):
                raise ValueError(
                    f"""Your number of measurements {self.number_measurements} is not divisible by the number of channels {self.reference_channels}.
                Please make sure the number of measurements you want to recored can be distributed accross the reference channels. (I.e. number_measurements is divisible by reference_cannels"""
                )
            data_array_dim = [
                self.reference_channels,
                self.number_pulse_sequences,
                self.number_dynamic_steps,
                int(measurements_per_channel),
                *self.roi_shape,
            ]
        else:
            logging.error(
                "Cannot create data array: unsupported averaging_mode=%s",
                self.averaging_mode,
            )
            raise ValueError(f"averaging_mode {self.averaging_mode} not available")
        self.data = np.zeros(data_array_dim, dtype=getattr(self, "data_type", float))
        self.chunk_data = np.zeros_like(self.data)
        logging.info(
            "Created data arrays: shape=%s, dtype=%s, averaging_mode=%s, "
            "live_compression=%s, chunk_size=%s, allocated_bytes=%s",
            self.data.shape,
            self.data.dtype,
            self.averaging_mode,
            self.live_compression,
            self.save_in_chunks,
            self.data.nbytes + self.chunk_data.nbytes,
        )

    def update_data(
        self, data: NDArray[Any], ps_step: int, dynamic_step: int, avg_step: int
    ) -> None:
        expected_shape = (self.number_measurements, *self._input_roi_shape)
        if data.shape != expected_shape:
            raise ValueError(
                f"Sensor returned data shape {data.shape}; expected {expected_shape}"
            )
        if self.live_compression:
            self._update_data_compressed(data, ps_step, dynamic_step)
            if self.save_in_chunks != 0:
                self._update_data_compressed(
                    data, ps_step, dynamic_step, target_array=self.chunk_data
                )
        else:
            self._update_data_full(data, ps_step, dynamic_step)
            if self.save_in_chunks != 0:
                self._update_data_full(
                    data, ps_step, dynamic_step, target_array=self.chunk_data
                )
        if self.save_in_chunks != 0 and (avg_step + 1) % self.save_in_chunks == 0:
            self.save(
                self.filename[0]
                + "_ch-"
                + f"{self.chunk_save_index}_"
                + self.filename[1],
                source_data=self.chunk_data,
                average_count=self.save_in_chunks,
            )
            self.chunk_save_index += 1
            self.chunk_data.fill(0)
        logging.debug(
            "Accumulated data: pulse_sequence=%s, dynamic_step=%s, average=%s",
            ps_step + 1,
            dynamic_step + 1,
            avg_step + 1,
        )

    def _update_data_full(
        self,
        data: NDArray[Any],
        ps_step: int,
        dynamic_step: int,
        target_array: NDArray[Any] | None = None,
    ) -> None:
        if target_array is None:
            target_array = self.data
        if self.averaging_mode == "sum":
            for i in range(self.reference_channels):
                target_array[i, ps_step, dynamic_step] += data[
                    i :: self.reference_channels
                ].sum(axis=0)
        elif self.averaging_mode == "spread":
            for i in range(self.reference_channels):
                target_array[i, ps_step, dynamic_step] += data[
                    i :: self.reference_channels
                ]

    def _update_data_compressed(
        self,
        data: NDArray[Any],
        ps_step: int,
        dynamic_step: int,
        target_array: NDArray[Any] | None = None,
    ) -> None:
        if target_array is None:
            target_array = self.data
        if self.averaging_mode == "sum":
            for i in range(self.reference_channels):
                ndim = data.ndim
                target_array[i, ps_step, dynamic_step] += (
                    data[i :: self.reference_channels]
                    .mean(axis=tuple(range(1, ndim)))
                    .sum(axis=0)
                )
        elif self.averaging_mode == "spread":
            for i in range(self.reference_channels):
                ndim = data.ndim
                target_array[i, ps_step, dynamic_step] += (
                    data[i :: self.reference_channels]
                    .mean(axis=tuple(range(1, ndim)))
                    .reshape(-1, 1)
                )

    def save(
        self,
        filename: str,
        source_data: NDArray[Any] | None = None,
        average_count: int | None = None,
    ) -> None:
        """Save measurement data to a NumPy file.

        Parameters
        ----------
        filename : str
            Output filename, normally assigned by the measurement loop. The .npy
            extension is appended if absent.
        source_data : numpy.ndarray or None, optional
            Data to save, or this container's data when None.
        average_count : int or None, optional
            Number of completed averages to use for normalization. None or a
            nonpositive value saves the data without normalization.

        Notes
        -----
        Metadata is saved separately.
        """
        if source_data is None:
            source_data = self.data
        output_path = str(Path(filename).absolute())
        if not output_path.endswith(".npy"):
            output_path += ".npy"
        logging.debug("Saving data to %s", output_path)
        try:
            if average_count is not None and average_count > 0:
                np.save(filename, source_data / average_count)
            else:
                np.save(filename, source_data)
        except Exception:
            logging.exception("Failed to save data to %s", output_path)
            raise
        logging.info(
            "Saved data to %s: shape=%s, source_dtype=%s, average_count=%s, chunk=%s",
            output_path,
            source_data.shape,
            source_data.dtype,
            average_count,
            source_data is getattr(self, "chunk_data", None),
        )


def save_measurement_results(
    data_container: Data | None,
    params: dict[str, Any],
    filename: str | None,
    average_count: int | None = None,
) -> str:
    """Save available data and metadata, recording persistence failures in the status."""
    params["qupyt_version"] = qupyt_version
    if (
        filename is not None
        and data_container is not None
        and hasattr(data_container, "data")
    ):
        try:
            if average_count is None:
                logging.warning(
                    "Saving partial, unaveraged data for failed measurement %s",
                    filename,
                )
            data_container.save(filename, average_count=average_count)
        except Exception as error:
            params["measurement_status"] = "failed"
            record_failure(params, error, "data saving")
            logging.exception("Measurement %s: failed to save data", filename)
    else:
        logging.warning(
            "Measurement %s has no allocated data to save",
            filename or params.get("experiment_type"),
        )
    if filename is not None:
        try:
            with open(filename + ".yaml", "w", encoding="utf-8") as file:
                yaml.dump(params, file)
            logging.info("Saved measurement metadata to %s.yaml", filename)
        except Exception as error:
            params["measurement_status"] = "failed"
            record_failure(params, error, "metadata saving")
            logging.exception("Measurement %s: failed to save metadata", filename)
    return str(params["measurement_status"])
