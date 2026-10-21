"""Main measurement loop."""

from __future__ import annotations

import gc
import logging
from datetime import datetime
from time import perf_counter, sleep
from typing import Any

from tqdm import tqdm

from qupyt.hardware.device_handler import DeviceHandler, DynamicDeviceHandler
from qupyt.hardware.sensors import Sensor
from qupyt.hardware.synchronisers import Synchroniser
from qupyt.measurement_logic.data_handling import Data, save_measurement_results
from qupyt.set_up import get_seq_dir
from qupyt.utils.errors import record_failure


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized_value = value.strip().lower()
        if normalized_value in {"true", "yes", "1", "on"}:
            return True
        if normalized_value in {"false", "no", "0", "off"}:
            return False
    return bool(value)


def _close_hardware(
    sensor: Sensor,
    synchroniser: Synchroniser,
    measurement_id: str,
    had_error: bool = False,
    params: dict[str, Any] | None = None,
) -> bool:
    """Attempt both cleanup operations even when a device fails to close."""
    closed_successfully = True
    for resource_name, resource in (("sensor", sensor), ("synchroniser", synchroniser)):
        try:
            logging.debug("Closing %s %s", resource_name, type(resource).__name__)
            resource.close()
        except (Exception, SystemExit) as error:
            closed_successfully = False
            if params is not None:
                detail = record_failure(params, error, f"{resource_name} close")
                params.setdefault("cleanup_errors", []).append(detail)
            logging.exception(
                "%sMeasurement %s: failed to close %s",
                (
                    "Additional cleanup error after measurement failure: "
                    if had_error
                    else ""
                ),
                measurement_id,
                resource_name,
            )
    return closed_successfully


def run_measurement(
    static_devices: DeviceHandler,
    dynamic_devices: DynamicDeviceHandler,
    sensor: Sensor,
    synchroniser: Synchroniser,
    params: dict[str, Any],
) -> str:
    started = perf_counter()
    mid = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    return_status = "failed"
    data_container: Data | None = None
    filename: str | None = None
    averages = 0
    stage = "initialization"
    ps_itervalue = itervalue = avg = None
    acquisition_completed = False
    try:
        params.pop("measurement_error", None)
        params.pop("cleanup_errors", None)
        filename = params["experiment_type"] + "_" + mid
        params["filename"] = filename
        iterator_size = int(params.get("dynamic_steps", 1))
        ps_iterator_size = int(params.get("pulse_sequence_steps", 1))
        averages = int(params["averages"])
        if min(iterator_size, ps_iterator_size, averages) < 1:
            raise ValueError(
                "dynamic_steps, pulse_sequence_steps, and averages must be positive"
            )
        synchroniser_params = params.get("synchroniser", {})
        upload_to_synchroniser = _as_bool(synchroniser_params.get("upload", True))
        has_dynamic_devices = bool(dynamic_devices.devices)
        logging.info(
            "Starting measurement %s: pulse_sequences=%s, dynamic_steps=%s, averages=%s",
            filename,
            ps_iterator_size,
            iterator_size,
            averages,
        )
        stage = "static device configuration"
        static_devices.set_all_params()
        stage = "data allocation"
        data_container = Data(params["data"])
        data_container.set_dims_from_sensor(sensor)
        data_container.create_array()
        data_container.set_filename((params["experiment_type"], mid))

        for ps_itervalue in tqdm(range(ps_iterator_size)):
            itervalue = avg = None
            logging.info(
                "Measurement %s: starting pulse sequence %s/%s",
                filename,
                ps_itervalue + 1,
                ps_iterator_size,
            )
            stage = "synchroniser open"
            synchroniser.open()
            stage = "synchroniser stop"
            synchroniser.stop()
            if upload_to_synchroniser:
                stage = "sequence upload"
                sequence_path = get_seq_dir() / f"sequence_{ps_itervalue}.yaml"
                synchroniser.load_sequence(sequence_path)
            else:
                logging.info(
                    "Skipped synchroniser sequence upload".ljust(65, ".") + "[done]"
                )
            stage = "synchroniser run"
            synchroniser.run()
            sleep(2)
            stage = "sensor open"
            sensor.open()
            sleep(0.1)
            for itervalue in tqdm(
                range(iterator_size), leave=(ps_itervalue == ps_iterator_size - 1)
            ):
                avg = None
                stage = "dynamic device configuration"
                dynamic_devices.next_dynamic_step()
                if has_dynamic_devices:
                    sleep(0.1)
                for avg in tqdm(
                    range(averages),
                    leave=(itervalue == (iterator_size - 1))
                    and (ps_itervalue == (ps_iterator_size - 1)),
                ):
                    sleep(float(params.get("sleep", 0)))
                    stage = "sensor acquisition"
                    data = sensor.acquire_data(synchroniser)
                    stage = "data accumulation"
                    data_container.update_data(data, ps_itervalue, itervalue, avg)
                logging.info(
                    "Measurement %s: dynamic step %s/%s completed (%s averages)",
                    filename,
                    itervalue + 1,
                    iterator_size,
                    averages,
                )
            dynamic_devices.current_dynamic_step = 0
            logging.info(
                "Measurement %s: pulse sequence %s/%s completed",
                filename,
                ps_itervalue + 1,
                ps_iterator_size,
            )
        acquisition_completed = True
        return_status = "success"
    except Exception as error:
        record_failure(
            params,
            error,
            stage,
            pulse_sequence=ps_itervalue,
            dynamic_step=itervalue,
            average=avg,
        )
        logging.exception(
            "Measurement %s failed during %s (zero-based pulse_sequence=%s, dynamic_step=%s, average=%s)",
            filename or mid,
            stage,
            ps_itervalue,
            itervalue,
            avg,
        )
    except (KeyboardInterrupt, SystemExit) as error:
        record_failure(params, error, stage)
        logging.warning("Measurement %s interrupted during %s", filename or mid, stage)
        raise
    finally:
        if not _close_hardware(
            sensor,
            synchroniser,
            filename or mid,
            had_error=not acquisition_completed,
            params=params,
        ):
            return_status = "failed"
        params["measurement_status"] = return_status
        return_status = save_measurement_results(
            data_container,
            params,
            filename,
            average_count=averages if acquisition_completed else None,
        )
        del data_container
        gc.collect()
        log_result = logging.info if return_status == "success" else logging.error
        log_result(
            "Measurement %s finished: status=%s, elapsed=%.3fs",
            filename or mid,
            return_status,
            perf_counter() - started,
        )
    return return_status
