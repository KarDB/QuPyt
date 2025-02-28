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
from typing import Optional
from pathlib import Path

import yaml
from watchdog.observers import Observer
from watchdog.events import (
    PatternMatchingEventHandler,
    FileModifiedEvent,
    FileClosedEvent,
)

from qupyt.hardware.device_handler import DeviceHandler, DynamicDeviceHandler
from qupyt.pulse_sequences.pulse_sequence_handler import (
    write_user_ps,
    update_params_dict,
)
from qupyt.hardware.synchronisers import SynchroniserFactory
from qupyt.hardware.sensors import SensorFactory
from qupyt.measurement_logic.run_measurement import run_measurement
from qupyt.hardware.signal_sources import SignalSource
from qupyt.set_up import get_waiting_room, make_userdirs, get_log_dir, get_home_dir

qupyt_logo_text = """                                                                                                        
                                                                                                         
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
                                                                                                         
"""

print("\nWelcome to")
print(qupyt_logo_text)

make_userdirs()
parser = argparse.ArgumentParser(description="Start QuPyt measurement")
parser.add_argument(
    "--verbose", action="store_true", help="deactivate logging output to screen"
)
args = parser.parse_args()

logfile = get_log_dir() / f"log_{date.today()}.log"
handlers: list[logging.Handler] = [logging.FileHandler(logfile)]
if args.verbose:
    handlers.append(logging.StreamHandler())
logging.basicConfig(
    handlers=handlers,
    # filename='logfile.log',
    format="%(levelname)s\t%(asctime)s %(message)s",
    datefmt="%Y/%m/%d %I:%M:%S %p",
    encoding="utf-8",
    level=logging.INFO,
    force=True,
)

queue: Queue[str]
event_thread: threading.Event


def _set_busy() -> None:
    with open(get_home_dir() / "status.txt", "w", encoding="utf-8") as file:
        file.write("busy")


def _set_ready() -> None:
    with open(get_home_dir() / "status.txt", "w", encoding="utf-8") as file:
        file.write("ready")


def parse_input() -> None:
    static_devices = DeviceHandler({})
    dynamic_devices = DynamicDeviceHandler({}, number_dynamic_steps=1)
    while True:
        if queue.empty():
            static_devices.update_devices({})
            dynamic_devices.update_devices({})
            _set_ready()
            event_thread.wait()
        try:
            _set_busy()
            logging.info("STARTED NEW MEASUREMENT".ljust(65, "=") + "[START]")
            instruction_file = queue.get()
            with open(instruction_file, "r", encoding="utf-8") as file:
                params = yaml.safe_load(file)
            os.rename(instruction_file, instruction_file + "_running")
            parameter_update = write_user_ps(
                Path(params["ps_path"]), params["pulse_sequence"]
            )
            update_params_dict(params, parameter_update)
            synchroniser = SynchroniserFactory.create_synchroniser(
                params["synchroniser"]["type"],
                params["synchroniser"]["config"],
                params["synchroniser"]["channel_mapping"],
            )
            sensor = SensorFactory.create_sensor(
                params["sensor"]["type"], params["sensor"]["config"]
            )
            static_devices.update_devices(params["static_devices"])
            dynamic_devices.number_dynamic_steps = int(params["dynamic_steps"])
            dynamic_devices.update_devices(params["dynamic_devices"])
            success_status = run_measurement(
                static_devices, dynamic_devices, sensor, synchroniser, params
            )
            if success_status == "success":
                os.remove(instruction_file + "_running")
            elif success_status == "failed":
                os.rename(instruction_file + "_running", instruction_file + "_failed")
        except Exception:
            logging.exception("Excpetion in main measurement loop")
            traceback.print_exc()


class WaitingRoomEventHandler(PatternMatchingEventHandler):
    """
    A custom event handler for monitoring changes in specific files.

    This handler extends PatternMatchingEventHandler to react to file modifications
    and closures. It uses different methods based on the operating system:
    - On Windows, it handles file modifications.
    - On other systems, it handles file closures.

    Attributes:
        patterns (list): List of file patterns to include in monitoring.
        ignore_patterns (list): List of file patterns to exclude from monitoring.
        ignore_directories (bool): Whether to ignore changes in directories.
        case_sensitive (bool): Whether file pattern matching is case sensitive.
    """

    def __init__(
        self,
        patterns: Optional[list[str]] = None,
        ignore_patterns: Optional[list[str]] = None,
        ignore_directories: bool = False,
        case_sensitive: bool = False,
    ):
        """
        Initialize the event handler with the specified parameters.

        Args:
            patterns (list, optional): List of file patterns to include in monitoring.
            ignore_patterns (list, optional): List of file patterns to exclude from monitoring.
            ignore_directories (bool, optional): Whether to ignore changes in directories. Defaults to False.
            case_sensitive (bool, optional): Whether file pattern matching is case sensitive. Defaults to False.
        """
        super().__init__(patterns, ignore_patterns, ignore_directories, case_sensitive)
        if platform.system() == "Windows":
            self.on_modified = self._on_modified
        if platform.system() == "Linux":
            self.on_closed = self._on_closed
        else:  # macOS or other
            self.on_created = self._on_created
            self.on_modified = self._on_modified
            self.on_moved = self._on_moved

    def _on_closed(self, event: FileClosedEvent) -> None:
        """
        Handle the event when a monitored file is closed.

        This method is called when a file matching the specified patterns is closed.
        It puts the file path in a queue and triggers the event thread.

        Args:
            event (FileClosedEvent): The event object containing information about the closed file.
        """
        queue.put(event.src_path)
        event_thread.set()
        event_thread.clear()

    def _on_modified(self, event: FileModifiedEvent) -> None:
        """
        Handle the event when a monitored file is modified.

        This method is called when a file matching the specified patterns is modified.
        It puts the file path in a queue, waits briefly, and triggers the event thread.

        Args:
            event (FileModifiedEvent): The event object containing information about the modified file.
        """
        queue.put(event.src_path)
        sleep(0.1)
        event_thread.set()
        event_thread.clear()

    def _on_created(self, event) -> None:
        """
        Handle the event when a monitored file is created.

        This method is called when a file matching the specified patterns is created.
        It puts the file path in a queue, waits briefly, and triggers the event thread.

        Args:
            event: The event object containing information about the modified file.
        """
        queue.put(event.src_path)
        sleep(0.1)
        event_thread.set()
        event_thread.clear()

    def _on_moved(self, event) -> None:
        """
        Handle the event when a monitored file is moved.

        This method is called when a file matching the specified patterns is moved.
        It puts the file path in a queue, waits briefly, and triggers the event thread.

        Args:
            event: The event object containing information about the modified file.
        """
        queue.put(event.src_path)
        sleep(0.1)
        event_thread.set()
        event_thread.clear()


def _get_observer_event_hanlder() -> WaitingRoomEventHandler:
    patterns = ["*.yaml"]
    ignore_patterns = None
    ignore_directories = True
    case_sensitive = True
    return WaitingRoomEventHandler(
        patterns, ignore_patterns, ignore_directories, case_sensitive
    )


def _get_observer(event_handler: WaitingRoomEventHandler) -> Observer:
    path = get_waiting_room()
    go_recursively = False
    my_observer = Observer()
    my_observer.schedule(event_handler, path, recursive=go_recursively)
    return my_observer


def main() -> None:
    """
    Start the main measurement loop.
    """
    logging.info("Started Program")
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
