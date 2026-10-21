# pylint: disable=logging-format-interpolation
# pylint: disable=logging-not-lazy
"""Start and manage the QuPyt measurement service."""

from __future__ import annotations

import argparse
import logging
import os
import platform
import sys
import threading
import time
from collections.abc import Callable
from datetime import date
from functools import partial
from pathlib import Path
from queue import Empty, Queue
from time import sleep
from typing import Any, cast

from qupyt.set_up import get_home_dir, get_log_dir, get_waiting_room, make_userdirs

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

parser = argparse.ArgumentParser(description="Start QuPyt measurement")
parser.add_argument(
    "--verbose", action="store_true", help="also show full logging output on screen"
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="include detailed acquisition and configuration logs",
)
args = parser.parse_args()

logfile = get_log_dir() / f"log_{date.today()}.log"


class _ConsoleErrorFormatter(logging.Formatter):
    """Show the cause on the terminal while keeping tracebacks in the log file."""

    def format(self, record: logging.LogRecord) -> str:
        message = f"{record.levelname}: {record.getMessage()}"
        if record.exc_info is not None and record.exc_info[1] is not None:
            error = record.exc_info[1]
            message += f"\n{type(error).__name__}: {error}"
        return f"{message}\nFull details: {logfile}"


try:
    make_userdirs()
    handlers: list[logging.Handler] = [logging.FileHandler(logfile, encoding="utf-8")]
except OSError as error:
    print(
        f"ERROR: Cannot create QuPyt directories or open the log file {logfile}.\n"
        f"{type(error).__name__}: {error}\n"
        "Check directory permissions and available disk space.",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(1) from error

console_handler = logging.StreamHandler()
if not args.verbose:
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(_ConsoleErrorFormatter())
handlers.append(console_handler)
logging.basicConfig(
    handlers=handlers,
    format="%(levelname)s\t%(asctime)s [%(threadName)s] %(module)s:%(lineno)d %(message)s",
    datefmt="%Y/%m/%d %I:%M:%S %p",
    level=logging.DEBUG if args.debug else logging.INFO,
    force=True,
)
logging.info(
    "Logging initialized: file=%s, level=%s", logfile, "DEBUG" if args.debug else "INFO"
)

# Initialize logging before imports so dependency failures also reach the terminal.
try:
    import yaml
    from watchdog.events import (
        FileClosedEvent,
        FileCreatedEvent,
        FileModifiedEvent,
        FileMovedEvent,
        PatternMatchingEventHandler,
    )
    from watchdog.observers import Observer
    from watchdog.observers.api import BaseObserver

    from qupyt.hardware.device_handler import DeviceHandler, DynamicDeviceHandler
    from qupyt.hardware.sensors import Sensor, SensorFactory
    from qupyt.hardware.synchronisers import Synchroniser, SynchroniserFactory
    from qupyt.measurement_logic.run_measurement import run_measurement
    from qupyt.pulse_sequences.pulse_sequence_handler import (
        update_params_dict,
        write_user_ps,
    )
    from qupyt.utils.errors import cleanup_after_error, cleanup_all, cleanup_on_exit
except Exception as error:
    logging.exception(
        "Cannot load QuPyt components; check the installed dependencies and drivers"
    )
    raise SystemExit(1) from error

queue: Queue[str]
event_thread: threading.Event


def _show_service_status(message: str) -> None:
    """Keep service startup and shutdown visible without enabling verbose logs."""
    logging.info(message)
    if not args.verbose:
        print(message, file=sys.stderr, flush=True)


def _set_busy() -> None:
    with open(get_home_dir() / "status.txt", "w", encoding="utf-8") as file:
        file.write("busy")
    logging.debug("Measurement service status changed to busy")


def _set_ready() -> None:
    with open(get_home_dir() / "status.txt", "w", encoding="utf-8") as file:
        file.write("ready")
    logging.debug("Measurement service status changed to ready")


def _prepare_measurement(
    params: dict[str, Any],
    static_devices: DeviceHandler,
    dynamic_devices: DynamicDeviceHandler,
) -> tuple[Sensor, Synchroniser]:
    """Prepare hardware, releasing only resources acquired after generation succeeds."""
    synchroniser = sensor = None
    preparing_hardware = False
    stage = "pulse sequence generation"
    try:
        parameter_update = write_user_ps(
            Path(params["ps_path"]), params["pulse_sequence"]
        )
        update_params_dict(params, parameter_update)
        preparing_hardware = True
        stage = "synchroniser creation"
        synchroniser = SynchroniserFactory.create_synchroniser(
            params["synchroniser"]["type"],
            params["synchroniser"]["config"],
            params["synchroniser"]["channel_mapping"],
        )
        stage = "sensor creation"
        sensor = SensorFactory.create_sensor(
            params["sensor"]["type"], params["sensor"]["config"]
        )
        stage = "device configuration"
        static_devices.update_devices(params["static_devices"])
        dynamic_devices.number_dynamic_steps = int(params["dynamic_steps"])
        dynamic_devices.update_devices(params["dynamic_devices"])
    except BaseException:
        logging.exception("Measurement preparation failed during %s", stage)
        for name, resource in (("sensor", sensor), ("synchroniser", synchroniser)):
            if resource is not None:
                cleanup_after_error(resource.close, f"{name} preparation cleanup")
        if preparing_hardware:
            for handler in (static_devices, dynamic_devices):
                cleanup_after_error(
                    partial(handler.update_devices, {}),
                    f"{type(handler).__name__} preparation cleanup",
                )
        raise
    logging.info("Hardware preparation completed")
    return sensor, synchroniser


def _process_instruction(
    instruction_file: str | Path,
    static_devices: DeviceHandler,
    dynamic_devices: DynamicDeviceHandler,
) -> None:
    """Claim one instruction and retain it on every preparation or measurement failure."""
    timestamp = time.time_ns()
    running_file = Path(f"{instruction_file}_running_{timestamp}")
    failed_file = Path(f"{instruction_file}_failed_{timestamp}")
    claimed = False
    stage = "claiming the instruction file"
    try:
        os.replace(instruction_file, running_file)
        claimed = True
        logging.info(
            "Claimed instruction file: %s -> %s", instruction_file, running_file
        )
        stage = "reading YAML instructions"
        with running_file.open("r", encoding="utf-8") as file:
            params = yaml.safe_load(file)
        if not isinstance(params, dict):
            raise ValueError("Measurement instructions must contain a YAML mapping")
        stage = "measurement preparation"
        sensor, synchroniser = _prepare_measurement(
            params, static_devices, dynamic_devices
        )
        stage = "measurement execution"
        success_status = run_measurement(
            static_devices, dynamic_devices, sensor, synchroniser, params
        )
        if success_status == "success":
            stage = "removing the completed instruction file"
            running_file.unlink()
            logging.info("Removed completed instruction file %s", running_file)
        elif success_status == "failed":
            # Report the primary cause before attempting recovery, which may fail too.
            detail = params.get("measurement_error", {})
            logging.error(
                "Instruction %s failed during %s: %s: %s",
                instruction_file,
                detail.get("stage", "measurement execution"),
                detail.get("type", "MeasurementError"),
                detail.get(
                    "message", "No failure details returned; check the log file"
                ),
            )
            stage = "retaining the failed instruction file"
            os.replace(running_file, failed_file)
            logging.error("Retained failed instruction file at %s", failed_file)
        else:
            raise RuntimeError(f"Unexpected measurement status {success_status!r}")
        logging.info(
            "Instruction %s finished with status=%s", instruction_file, success_status
        )
    except BaseException:
        logging.exception(
            "Exception while processing measurement instructions %s during %s",
            instruction_file,
            stage,
        )
        if claimed:

            def retain_failed_instruction() -> None:
                if running_file.exists():
                    try:
                        os.replace(running_file, failed_file)
                    except (Exception, SystemExit):
                        logging.exception(
                            "Cannot retain failed instruction as %s; recover it from %s",
                            failed_file,
                            running_file,
                        )
                        raise
                    logging.error("Retained failed instruction file at %s", failed_file)

            cleanup_after_error(
                retain_failed_instruction, "failed instruction retention"
            )
        raise


def _release_idle_devices(
    static_devices: DeviceHandler | None, dynamic_devices: DynamicDeviceHandler | None
) -> None:
    """Attempt both handlers even if one device refuses to close."""
    for handler in (static_devices, dynamic_devices):
        if handler is None:
            continue
        try:
            handler.update_devices({})
        except (Exception, SystemExit):
            logging.exception(
                "Failed to release idle %s devices", type(handler).__name__
            )


def parse_input(stop_event: threading.Event | None = None) -> None:
    static_devices = None
    dynamic_devices = None
    stop_event = stop_event or threading.Event()
    idle = False
    try:
        static_devices = DeviceHandler({})
        dynamic_devices = DynamicDeviceHandler({}, number_dynamic_steps=1)
        while not stop_event.is_set() or not queue.empty():
            if queue.empty():
                if not idle:
                    _release_idle_devices(static_devices, dynamic_devices)
                    _set_ready()
                    idle = True
                    logging.debug(
                        "Waiting for measurement instructions in %s", get_waiting_room()
                    )
                event_thread.wait(timeout=0.5)
                event_thread.clear()
                continue
            try:
                instruction_file = queue.get(timeout=0.5)
            except Empty:
                continue
            idle = False
            try:
                if not Path(instruction_file).exists():
                    logging.warning(
                        "Skipping instruction file that no longer exists: %s",
                        instruction_file,
                    )
                    continue
                _set_busy()
                _process_instruction(instruction_file, static_devices, dynamic_devices)
            except Exception:
                # The instruction helper already recorded the primary traceback.
                # A failure must not terminate the worker or stall queue.join().
                logging.exception(
                    "Instruction %s failed; continuing with the queue. "
                    "QuPyt is still running and accepting instructions.",
                    instruction_file,
                )
            finally:
                queue.task_done()
    except BaseException:
        logging.exception("Measurement worker stopped unexpectedly")
        raise
    finally:
        _release_idle_devices(static_devices, dynamic_devices)


class WaitingRoomEventHandler(PatternMatchingEventHandler):
    """Queue incoming measurement instructions from filesystem events.

    Attributes
    ----------
    patterns : list[str]
        File patterns included in monitoring.
    ignore_patterns : list[str]
        File patterns excluded from monitoring.
    ignore_directories : bool
        Whether directory events are ignored.
    case_sensitive : bool
        Whether pattern matching is case sensitive.

    Notes
    -----
    Windows uses modification events; Linux uses file-closed events. Other systems
    use creation, modification, and move events.
    """

    def __init__(
        self,
        patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        ignore_directories: bool = False,
        case_sensitive: bool = False,
    ) -> None:
        """Initialize the filesystem filters and platform-specific event callbacks.

        Parameters
        ----------
        patterns : list[str] or None, optional
            File patterns included in monitoring, or the watchdog default when None.
        ignore_patterns : list[str] or None, optional
            File patterns excluded from monitoring, or no exclusions when None.
        ignore_directories : bool, optional
            Ignore directory events when True. Defaults to False.
        case_sensitive : bool, optional
            Use case-sensitive pattern matching when True. Defaults to False.
        """
        cast(Callable[..., None], super().__init__)(
            patterns, ignore_patterns, ignore_directories, case_sensitive
        )
        if platform.system() == "Windows":
            setattr(self, "on_modified", self._on_modified)
        elif platform.system() == "Linux":
            setattr(self, "on_closed", self._on_closed)
        else:  # macOS or other
            setattr(self, "on_created", self._on_created)
            setattr(self, "on_modified", self._on_modified)
            setattr(self, "on_moved", self._on_moved)

    def _on_closed(self, event: FileClosedEvent) -> None:
        """Queue an instruction when its monitored file is closed.

        Signal the measurement worker after placing the path in its queue.

        Parameters
        ----------
        event : FileClosedEvent
            Filesystem event containing the instruction's source path.
        """
        logging.debug("Queueing closed instruction file %s", event.src_path)
        queue.put(event.src_path)
        event_thread.set()
        event_thread.clear()

    def _on_modified(self, event: FileModifiedEvent) -> None:
        """Queue an instruction when its monitored file is modified.

        Wait briefly after placing the path in the queue, then signal the worker.

        Parameters
        ----------
        event : FileModifiedEvent
            Filesystem event containing the instruction's source path.
        """
        logging.debug("Queueing modified instruction file %s", event.src_path)
        queue.put(event.src_path)
        sleep(0.1)
        event_thread.set()
        event_thread.clear()

    def _on_created(self, event: FileCreatedEvent) -> None:
        """Queue an instruction when its monitored file is created.

        Wait briefly after placing the path in the queue, then signal the worker.

        Parameters
        ----------
        event : FileCreatedEvent
            Filesystem event containing the instruction's source path.
        """
        logging.debug("Queueing created instruction file %s", event.src_path)
        queue.put(event.src_path)
        sleep(0.1)
        event_thread.set()
        event_thread.clear()

    def _on_moved(self, event: FileMovedEvent) -> None:
        """Queue an incoming YAML instruction at its move destination.

        Parameters
        ----------
        event : FileMovedEvent
            Filesystem event containing the moved file's destination path.

        Notes
        -----
        Ignore destinations used to claim instructions or retain failed measurements.
        """
        # Watchdog may match the source name when we claim a YAML instruction.
        # Only incoming YAML names belong in the queue, not our recovery files.
        if Path(event.dest_path).suffix != ".yaml":
            return
        logging.debug(
            "Queueing moved instruction file from %s to %s",
            event.src_path,
            event.dest_path,
        )
        queue.put(event.dest_path)
        sleep(0.1)
        event_thread.set()
        event_thread.clear()


def _get_observer_event_handler() -> WaitingRoomEventHandler:
    patterns = ["*.yaml"]
    ignore_patterns = None
    ignore_directories = True
    case_sensitive = True
    return WaitingRoomEventHandler(
        patterns, ignore_patterns, ignore_directories, case_sensitive
    )


def _get_observer(event_handler: WaitingRoomEventHandler) -> BaseObserver:
    path = get_waiting_room()
    go_recursively = False
    my_observer = Observer()
    cast(Callable[..., object], my_observer.schedule)(
        event_handler, path, recursive=go_recursively
    )
    return my_observer


def _join_service_thread(thread: threading.Thread, description: str) -> None:
    """Wait for safe cleanup and explain delays without interrupting hardware."""
    if thread.ident is None:
        return
    thread.join(timeout=5)
    while thread.is_alive():
        _show_service_status(
            f"Still waiting for {description} to finish. "
            "Queued measurements and hardware cleanup may take time."
        )
        thread.join(timeout=5)


def main() -> None:
    """Start the main measurement loop."""
    logging.info(
        "Starting QuPyt measurement service; waiting_room=%s", get_waiting_room()
    )
    global event_thread
    event_thread = threading.Event()
    global queue
    queue = Queue()
    observer: BaseObserver | None = None
    thread: threading.Thread | None = None
    stop_event = threading.Event()
    service_failed = False

    def shutdown() -> None:
        stop_event.set()
        event_thread.set()
        operations: list[tuple[str, Callable[[], object]]] = []
        if observer is not None:
            operations.extend(
                [
                    ("observer stop", observer.stop),
                    (
                        "observer join",
                        partial(
                            _join_service_thread, observer, "waiting-room observer"
                        ),
                    ),
                ]
            )
        if thread is not None:
            operations.append(
                (
                    "measurement worker join",
                    partial(_join_service_thread, thread, "measurement worker"),
                )
            )
        cleanup_all(operations)

    try:
        with cleanup_on_exit(shutdown, "measurement service shutdown"):
            try:
                event_handler = _get_observer_event_handler()
                observer = _get_observer(event_handler)
                thread = threading.Thread(
                    target=parse_input, args=(stop_event,), name="QuPyt-measurements"
                )
                cast(Callable[[], None], observer.start)()
                thread.start()
                _show_service_status(
                    f"QuPyt is waiting for measurement instructions in {get_waiting_room()}."
                )
                logging.info("Waiting-room observer and measurement worker started")
                while True:
                    sleep(0.5)
                    if not thread.is_alive():
                        raise RuntimeError(
                            "Measurement worker terminated unexpectedly; see preceding error"
                        )
                    if not observer.is_alive():
                        raise RuntimeError(
                            "Waiting-room observer terminated unexpectedly; "
                            f"QuPyt can no longer accept instructions from {get_waiting_room()}"
                        )
                    for emitter in observer.emitters:
                        if not emitter.is_alive():
                            raise RuntimeError(
                                "Waiting-room filesystem watcher terminated unexpectedly; "
                                f"check that {get_waiting_room()} is accessible"
                            )
            except KeyboardInterrupt:
                _show_service_status(
                    "Shutdown requested; waiting for queued measurements to finish "
                    "and hardware to be released."
                )
            except BaseException:
                # Report before cleanup, which can itself be delayed by a driver.
                service_failed = True
                logging.exception("QuPyt measurement service failed")
                raise
    except (Exception, SystemExit):
        if not service_failed:
            logging.exception("QuPyt measurement service could not shut down cleanly")
        raise
    _show_service_status("QuPyt measurement service stopped")


if __name__ == "__main__":
    main()
