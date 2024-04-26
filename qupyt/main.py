# pylint: disable=logging-format-interpolation
# pylint: disable=logging-not-lazy
"""
Main programm. Measurement loop is started from here.
"""
import argparse
import logging
import threading
import traceback
import os
import platform
from datetime import date
from time import sleep
from queue import Queue
from typing import Dict
from pathlib import Path

import yaml
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler, FileModifiedEvent, FileClosedEvent

import qupyt.hardware.device_handler as dh
from qupyt.pulse_sequences.pulse_sequence_handler import write_user_ps, update_params_dict
from qupyt.hardware.synchronisers import SynchroniserFactory
from qupyt.hardware.sensors import SensorFactory
from qupyt.measurement_logic.run_measurement import run_measurement
from qupyt.hardware.signal_sources import SignalSource
from qupyt.set_up import get_waiting_room, make_userdirs, get_log_dir, get_home_dir

qupyt_logo_text = '''                                                                                                        
                                                                                                         
     QQQQQQQQQ                       PPPPPPPPPPPPPPPPP                                     tttt          
   QQ:::::::::QQ                     P::::::::::::::::P                                 ttt:::t          
 QQ:::::::::::::QQ                   P::::::PPPPPP:::::P                                t:::::t          
Q:::::::QQQ:::::::Q                  PP:::::P     P:::::P                               t:::::t          
Q::::::O   Q::::::Quuuuuu    uuuuuu    P::::P     P:::::Pyyyyyyy           yyyyyyyttttttt:::::ttttttt    
Q:::::O     Q:::::Qu::::u    u::::u    P::::P     P:::::P y:::::y         y:::::y t:::::::::::::::::t    
Q:::::O     Q:::::Qu::::u    u::::u    P::::PPPPPP:::::P   y:::::y       y:::::y  t:::::::::::::::::t    
Q:::::O     Q:::::Qu::::u    u::::u    P:::::::::::::PP     y:::::y     y:::::y   tttttt:::::::tttttt    
Q:::::O     Q:::::Qu::::u    u::::u    P::::PPPPPPPPP        y:::::y   y:::::y          t:::::t          
Q:::::O     Q:::::Qu::::u    u::::u    P::::P                 y:::::y y:::::y           t:::::t          
Q:::::O  QQQQ:::::Qu::::u    u::::u    P::::P                  y:::::y:::::y            t:::::t          
Q::::::O Q::::::::Qu:::::uuuu:::::u    P::::P                   y:::::::::y             t:::::t    tttttt
Q:::::::QQ::::::::Qu:::::::::::::::uuPP::::::PP                  y:::::::y              t::::::tttt:::::t
 QQ::::::::::::::Q  u:::::::::::::::uP::::::::P                   y:::::y               tt::::::::::::::t
   QQ:::::::::::Q    uu::::::::uu:::uP::::::::P                  y:::::y                  tt:::::::::::tt
     QQQQQQQQ::::QQ    uuuuuuuu  uuuuPPPPPPPPPP                 y:::::y                     ttttttttttt  
             Q:::::Q                                           y:::::y                                   
              QQQQQQ                                          y:::::y                                    
                                                             y:::::y                                     
                                                            y:::::y                                      
                                                           yyyyyyy                                       
                                                                                                         
'''

print('Welcome to')
print(qupyt_logo_text)

make_userdirs()
parser = argparse.ArgumentParser(description='Start QuPyt measurement')
parser.add_argument('--verbose', action="store_true",
                    help='deactivate logging output to screen')
args = parser.parse_args()

logfile = get_log_dir() / f"log_{date.today()}.log"
handlers: list[logging.Handler] = [logging.FileHandler(logfile)]
if args.verbose:
    handlers.append(logging.StreamHandler())
logging.basicConfig(handlers=handlers,
                    # filename='logfile.log',
                    format="%(levelname)s\t%(asctime)s %(message)s",
                    datefmt='%Y/%m/%d %I:%M:%S %p',
                    encoding='utf-8',
                    level=logging.INFO,
                    force=True)

queue: Queue[str]
event_thread: threading.Event


def _on_closed(event: FileClosedEvent) -> None:
    queue.put(event.src_path)
    event_thread.set()
    event_thread.clear()


def _on_modified(event: FileModifiedEvent) -> None:
    queue.put(event.src_path)
    sleep(0.1)
    event_thread.set()
    event_thread.clear()


def _set_busy() -> None:
    with open(get_home_dir() / 'status.txt', 'w', encoding='utf-8') as file:
        file.write('busy')


def _set_ready() -> None:
    with open(get_home_dir() / 'status.txt', 'w', encoding='utf-8') as file:
        file.write('ready')


def parse_input() -> None:
    static_devices: Dict[str, SignalSource] = {}
    dynamic_devices: Dict[str, SignalSource] = {}
    while True:
        if queue.empty():
            static_devices_requested: Dict[str, SignalSource] = {}
            dynamic_devices_requested: Dict[str, SignalSource] = {}
            dh.close_superfluous_devices(
                static_devices, static_devices_requested)
            dh.close_superfluous_devices(
                dynamic_devices, dynamic_devices_requested)
            _set_ready()
            event_thread.wait()
        try:
            _set_busy()
            logging.info('STARTED NEW MEASUREMENT'.ljust(65, '=') + '[START]')
            instruction_file = queue.get()
            with open(instruction_file, "r", encoding='utf-8') as file:
                params = yaml.safe_load(file)
            os.rename(instruction_file, instruction_file + "_running")
            parameter_update = write_user_ps(Path(params['ps_path']),
                                             params['pulse_sequence'])
            update_params_dict(params, parameter_update)
            synchroniser = SynchroniserFactory.create_synchroniser(
                params['synchroniser']['type'],
                params['synchroniser']['config'],
                params['synchroniser']['channel_mapping']
            )
            sensor = SensorFactory.create_sensor(
                params['sensor']['type'],
                params['sensor']['config']
            )

            static_devices_requested, dynamic_devices_requested = dh.get_device_dicts(
                params
            )
            dh.open_new_requested_devices(
                static_devices, static_devices_requested)
            dh.open_new_requested_devices(
                dynamic_devices, dynamic_devices_requested)

            dh.close_superfluous_devices(
                static_devices, static_devices_requested)
            dh.close_superfluous_devices(
                dynamic_devices, dynamic_devices_requested)

            success_status = run_measurement(
                static_devices, dynamic_devices, sensor, synchroniser, params
            )
            if success_status == "success":
                os.remove(instruction_file + "_running")
            elif success_status == "failed":
                os.rename(instruction_file + "_running",
                          instruction_file + "_failed")
        except Exception:
            logging.exception("Excpetion in main measurement loop")
            traceback.print_exc()


def _get_observer_event_hanlder() -> PatternMatchingEventHandler:
    patterns = ["*.yaml"]
    ignore_patterns = None
    ignore_directories = True
    case_sensitive = True
    event_handler = PatternMatchingEventHandler(
        patterns, ignore_patterns, ignore_directories, case_sensitive
    )
    if platform.system() == 'Windows':
        event_handler.on_modified = _on_modified
    else:
        event_handler.on_closed = _on_closed
    return event_handler


def _get_observer(event_handler: PatternMatchingEventHandler) -> Observer:
    path = get_waiting_room()
    go_recursively = False
    my_observer = Observer()
    my_observer.schedule(event_handler, path, recursive=go_recursively)
    return my_observer


def main() -> None:
    """
    Start the main measurement loop.
    """
    logging.info('Started Program')
    global event_thread
    event_thread = threading.Event()
    global queue
    queue = Queue()
    event_handler = _get_observer_event_hanlder()
    observer = _get_observer(event_handler)
    thread = threading.Thread(target=parse_input)
    observer.start()
    thread.start()
    try:
        while True:
            sleep(0.5)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()
        thread.join()
        queue.join()


if __name__ == "__main__":
    main()
