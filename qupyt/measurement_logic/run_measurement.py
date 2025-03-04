"""
Main measurement loop.
"""

import logging
from time import sleep
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
        data_container = Data(params["data"])
        data_container.set_dims_from_sensor(sensor)
        data_container.create_array()

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
                data = sensor.acquire_data(synchroniser)
                data_container.update_data(data, itervalue, avg)
        return_status = "success"
    except Exception as e:
        print(f"exc {e}")
        logging.exception("An error occured during the measurement!")
        return_status = "failed"
    finally:
        sensor.close()
        synchroniser.close()
        print("sensor closed")
        params["filename"] = params["experiment_type"] + "_" + mid
        params["measurement_status"] = return_status
        params["qupyt_version"] = qupyt_version

        data_container.save(params["filename"])
        with open(params["filename"] + ".yaml", "w", encoding="utf-8") as file:
            yaml.dump(params, file)
        del data_container
        gc.collect()
    return return_status
