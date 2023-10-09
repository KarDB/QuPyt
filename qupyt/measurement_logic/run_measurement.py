"""
Main measurement loop.
"""
import logging
from time import sleep
from datetime import datetime
from typing import Dict, Any

import yaml
from tqdm import tqdm

import qupyt.hardware.device_handler as dh
from qupyt.measurement_logic.data_handling import Data
from qupyt.hardware.synchronisers import Synchroniser
from qupyt.hardware.sensors import Sensor


def run_measurement(static_devices: Dict[str, Any],
                    dynamic_devices: Dict[str, Any],
                    sensor: Sensor,
                    synchroniser: Synchroniser,
                    params: Dict[str, Any]) -> str:

    dh.make_sweep_lists(dynamic_devices, int(params.get('dynamic_steps')))
    dh.set_all_static_params(static_devices)
    iterator_size = dh.get_iterator_size(dynamic_devices)
    mid = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    return_status = 'all_fail'
    try:
        sensor.open()
        print('sensor opened')

        data_container = Data(params['data'])
        data_container.set_dims_from_sensor(sensor)
        data_container.create_array()

        synchroniser.open()
        synchroniser.stop()
        synchroniser.load_sequence()
        synchroniser.run()
        sleep(0.5)
        for itervalue in tqdm(range(iterator_size)):
            if dynamic_devices:
                dh.set_all_dynamic_params(dynamic_devices, itervalue)
            sleep(0.1)
            for _ in tqdm(range(int(params["averages"])),
                          leave=itervalue == (iterator_size - 1)):
                sleep(2.5)
                data = sensor.acquire_data(synchroniser)
                data_container.update_data(data, itervalue)
        return_status = 'success'
    except Exception as e:
        print(f"exc {e}")
        logging.exception('An error occured during the measurement!')
        return_status = 'failed'
    finally:
        sensor.close()
        print('sensor closed')
        params['filename'] = params['experiment_type'] + "_" + mid
        params['measurement_status'] = return_status

        data_container.save(params['filename'])
        with open(params['filename'] + '.yaml', 'w', encoding='utf-8') as file:
            yaml.dump(params, file)
    return return_status
