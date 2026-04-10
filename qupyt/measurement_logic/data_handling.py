import logging
from typing import Dict, Any, List
import numpy as np
from qupyt.hardware.sensors import Sensor
from qupyt.mixins import ConfigurationMixin, UpdateConfigurationType
import time
import os
import multiprocessing


class Data(ConfigurationMixin):
    attribute_map: UpdateConfigurationType

    def __init__(self, configuration: Dict[str, Any], measurement_file_name:str) -> None:
        self.roi_shape: list[int]
        self.averaging_mode: str
        self.number_dynamic_steps: int
        self.number_measurements: int
        self.data_type: type
        self.live_compression: bool = False
        self.save_in_chunks: int = 0
        self.reference_channels: int = 2
        self.data: np.ndarray
        self.memory_map: np.memmap
        self.memory_map_current_length = 0
        
        
        self.measurement_file_name = measurement_file_name
        self.attribute_map = {
            "dynamic_steps": self._set_number_dynamic_steps,
            "averaging_mode": self._set_averaging_mode,
            "number_measurements": self._set_number_measurements,
            "roi_shape": self._set_roi_shape,
            "compress": self._set_compress_mode,
            "live_compression": self._set_live_compression,
            "save_in_chunks": self._set_save_chunk_size,
            "reference_channels": self._set_reference_channels,
            "time_stamping": self._set_time_stamping,
            "gui_live_view": self._set_gui_live_view,
            "temporary_saving": self._set_temporary_saving
        }
        
        self._update_from_configuration(configuration)
        
    def _set_save_chunk_size(self, save_in_chunks: int) -> None:
        self.save_in_chunks = save_in_chunks
    def _set_temporary_saving(self, temporary_saving: bool) -> None:
        self.temporary_saving = temporary_saving
    def _set_reference_channels(self, reference_channels: int) -> None:
        self.reference_channels = int(reference_channels)
    def _set_time_stamping(self, time_stamping: bool) -> None:
        self.time_stamping = time_stamping
    def _set_gui_live_view(self, gui_live_view: bool) -> None:
        self.gui_live_view = gui_live_view
    def _set_compress_mode(self, compression_value: bool) -> None:
        logging.warning("Depracation warning! The paremter 'compress' is beeing deprecated.\nPlase use 'live_compression' instead")
        self.live_compression = compression_value

    def _set_live_compression(self, live_compression: bool) -> None:
        self.live_compression = live_compression

    def _set_dtype_from_sensor(self, sensor: Sensor) -> None:
        self.data_type = sensor.target_data_type

    def _set_number_dynamic_steps(self, number_dynamic_steps: int) -> None:
        if number_dynamic_steps < 1:
            raise ValueError("""Cannot have fewer than 1 dynamic steps. This will result in an empty measurement.
            If you measure just one step, this is still ONE dynamic step. The first of one.""")
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
            self.data_type = float
        if self.averaging_mode == "sum":
            data_array_dim = [
                self.reference_channels,
                self.number_dynamic_steps,
                1,
                *self.roi_shape,
            ]
        elif self.averaging_mode == "spread":
            measurements_per_channel = self.number_measurements / self.reference_channels
            if measurements_per_channel != int(measurements_per_channel):
                raise ValueError(f"""Your number of measurements {self.number_measurements} is not divisible by the number of channels {self.reference_channels}.
                Please make sure the number of measurements you want to recored can be distributed accross the reference channels. (I.e. number_measurements is divisible by reference_cannels""")
            data_array_dim = [
                self.reference_channels,
                self.number_dynamic_steps,
                int(measurements_per_channel),
                *self.roi_shape,
            ]
        else:
            logging.info(
                f"Failed to create dara array for mode {self.averaging_mode}".ljust(
                    65, "."
                )
                + "[failed]"
            )
            raise ValueError(f"averaging_mode {self.averaging_mode} not available")
        logging.info(
            f"Created data array of shape {data_array_dim}".ljust(65, ".") + "[done]"
        )
        self.data = np.zeros(data_array_dim, dtype=getattr(self, "data_type", float))
        
    def create_memory_map(self, file_name: str, averages: int) -> None:
        
        mmap_width = 2+self.number_measurements
        mmap_length = averages * self.number_dynamic_steps
        self.memory_map = np.memmap(f"C:/Users/ge54vec/.qupyt/temp/{file_name}_temp.dat",dtype=float, mode="w+",
                                    shape=(mmap_length,mmap_width))
        self.memory_map[:,:] = 0.0
        self.memory_map.flush()
        print(f"Memory map dimensions are: {self.memory_map.shape}, {self.memory_map.size}")
    def update_data(self, data: np.ndarray, dynamic_step: int, avg_step: int, t_begin: float, t_end: float) -> None:
        #print(data)
        if self.save_in_chunks != 0 and avg_step % self.save_in_chunks == 0:
            time_0 = time.time()
            os.makedirs(f"{self.measurement_file_name}_chunks", exist_ok=True)
            print(f"directory creation time: {(time.time() - time_0)} ms")
            self.save(f"{self.measurement_file_name}_chunks/{self.measurement_file_name}_save_chunk_{avg_step}.npy")
            print(f"chunk writing time: {(time.time() - time_0)*1000} ms")
            self.create_array()
       
        if self.temporary_saving:
            self.save_temporary_file(data, t_begin, t_end) 
        if self.live_compression:
            self._update_data_compressed(data, dynamic_step)    
        else:
            self._update_data_full(data, dynamic_step)

    def _update_data_full(self, data: np.ndarray, dynamic_step: int) -> None:
        #print(f"Data Stream:{data}")
        #time_0 = time.time()
        if self.averaging_mode == "sum":
            for i in range(self.reference_channels):
                self.data[i, dynamic_step] += data[i :: self.reference_channels].sum(
                    axis=0
                )
            if self.gui_live_view:
                np.save("C:/Users/ge54vec/.qupyt/data", self.data)
        elif self.averaging_mode == "spread":
            for i in range(self.reference_channels):
                self.data[i, dynamic_step] += data[i :: self.reference_channels]
                if self.gui_live_view:
                    np.save("C:/Users/ge54vec/.qupyt/data", self.data)
        #print(f"updating data_full time: {(time.time() - time_0)*1000} ms ")

    def _update_data_compressed(self, data: np.ndarray, dynamic_step: int) -> None:
        if self.averaging_mode == "sum":
            for i in range(self.reference_channels):
                ndim = data.ndim
                self.data[i, dynamic_step] += (
                    data[i :: self.reference_channels].mean(axis=tuple(range(1, ndim))).sum(axis=0)
                )
        elif self.averaging_mode == "spread":
            for i in range(self.reference_channels):
                ndim = data.ndim
                self.data[i, dynamic_step] += (
                    data[i :: self.reference_channels].mean(axis=tuple(range(1, ndim))).reshape(-1, 1)
                )

    def save(self, filename: str) -> None:
        """
        Save data (without metadata) to a .npy file.
        :param filename: Name of the resulting data file.
         During normal usage as part of QuPyt, this will be assigned by the
         main measurement loop.
        :type filename: str
        """
        #Multiprocessing parallelizes the process since it moves it to another kernel
        a = multiprocessing.Process(target=np.save,args=(filename,self.data),)
        a.start()
        #np.save(filename, self.data)

    def save_temporary_file(self, data: np.ndarray, t_begin: float, t_end: float) -> None:
       #flatten data
        reshaped_data = np.array(data.flatten("F"), copy=True, dtype=float)

        # Write timestamps
        time_stamps = np.array([t_begin, t_end], dtype=float, copy=True)
        #conca data
        data_stream = np.concatenate((time_stamps, reshaped_data,),axis=None)
        # Save to memmap
        self.memory_map[self.memory_map_current_length] = data_stream
        self.memory_map.flush()
    
        print(f"Saved record {self.memory_map_current_length + 1}: t0={t_begin}, t1={t_end}")
        #print( self.memory_map[self.memory_map_current_length])
        self.memory_map_current_length += 1
