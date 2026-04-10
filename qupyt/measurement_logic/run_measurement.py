"""
Main measurement loop.
"""

import logging
from time import sleep, time
from datetime import datetime
from typing import Dict, Any
import gc

import yaml
from tqdm import tqdm

from qupyt.hardware.device_handler import DeviceHandler, DynamicDeviceHandler
from qupyt.measurement_logic.data_handling import Data
from qupyt.hardware.synchronisers import Synchroniser
from qupyt.hardware.sensors import Sensor
from qupyt._version import __version__ as qupyt_version
import numpy as np
import os
import shutil



def run_measurement(
    static_devices: DeviceHandler,
    dynamic_devices: DynamicDeviceHandler,
    sensor: Sensor,
    synchroniser: Synchroniser,
    params: Dict[str, Any],
) -> str:
    static_devices.set_all_params()
    iterator_size = int(params.get("dynamic_steps", 1))
    mid = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    return_status = "all_fail"
    try:
        # Create the params file - BD
        if "file_name" not in params:
            params["file_name"] = params["experiment_type"] + "_" + mid
        params["measurement_status"] = return_status
        params["qupyt_version"] = qupyt_version
        if "time_stamping" in params["data"] and params["data"]["time_stamping"] == True:
            time_stamping = True
        else:
            time_stamping = False
        update_param_file(params)
        data_container = Data(params["data"], params["file_name"])
        data_container.set_dims_from_sensor(sensor)
        data_container.create_array()
        if "temporary_saving" in params["data"] and params["data"]["temporary_saving"] == True:
            data_container.create_memory_map(params["file_name"], params["averages"])
        synchroniser.open()
        synchroniser.stop()
        synchroniser.load_sequence()
        synchroniser.run()
        sleep(0.1)
        sensor.open()
        sleep(0.5)
        for itervalue in tqdm(range(iterator_size)):
            dynamic_devices.next_dynamic_step()
            sleep(0.1)
            for avg in tqdm(
                range(int(params["averages"])), leave=itervalue == (iterator_size - 1)
            ):
                sleep(float(params.get("sleep", 0)))
                t_begin = time()
                time_0 = time() 
                data = sensor.acquire_data(synchroniser)
                #print(f"data_acquiration time: {(time()-time_0)*1000} ms")
                time_1 = time()
                t_end = time()
                if time_stamping:
                    time_2 = time()
                    append_timestamp(params["file_name"], t_begin, t_end)
                    print(f"Time_stamps_update: {(time() - time_2)*1000} ms")
                time_3 = time()
                
                data_container.update_data(data, itervalue, avg, t_begin, t_end)
                #print(f"data_container file update: {(time() - time_3)*1000} ms")
                #print(f"Whole Data update process time: {(time() - time_1)*1000} ms")
                
        return_status = "success"
    except Exception as e:
        print(f"exc {e}")
        logging.exception("An error occured during the measurement!")
        return_status = "failed"
    finally:
        sensor.close()
        synchroniser.close()
        print("sensor closed")
        data_container.save(params["file_name"])
       
        
        
    #copy and delete the temporary file
        if "temporary_saving" in params["data"] and params["data"]["temporary_saving"] == True:
            #print(data_container.memory_map)
            shutil.copy(f"C:/Users/ge54vec/.qupyt/temp/{params['file_name']}_temp.dat",
                        f"{params['file_name']}_temp.dat")
        del data_container
        gc.collect()
        sleep(1)
        os.remove(f"C:/Users/ge54vec/.qupyt/temp/{params['file_name']}_temp.dat")
    return return_status
# This function updates the params file after each measurement - BD
def update_param_file(params: Dict[str, any]) -> None:
    with open(params["file_name"] + ".yaml", "w", encoding="utf-8") as file:
        yaml.dump(params, file)
    
def append_timestamp(file_name, t_begin, t_end):
    with open(file_name + "_timestamps.csv", "a") as f:
        f.write(f"{t_begin},{t_end}\n")
