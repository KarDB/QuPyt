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
from qupyt.utils.logging import log_done, log_values


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
                "%s%s: failed to close %s",
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
        log_values(
            "\n" + "=" * 88 + f"\nMEASUREMENT START: {filename}\n" + "=" * 88,
            {
                "pulse_sequences": ps_iterator_size,
                "dynamic_steps": iterator_size,
                "averages": averages,
            },
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
                "%s: starting pulse sequence %s/%s",
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
                log_done("Skipped synchroniser sequence upload")
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
                log_done(
                    f"{filename}: dynamic step {itervalue + 1}/{iterator_size} "
                    f"({averages} average{'s' if averages != 1 else ''})"
                )
            dynamic_devices.current_dynamic_step = 0
            log_done(
                f"{filename}: pulse sequence {ps_itervalue + 1}/{ps_iterator_size}"
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
            "%s failed during %s (zero-based pulse_sequence=%s, dynamic_step=%s, average=%s)",
            filename or mid,
            stage,
            ps_itervalue,
            itervalue,
            avg,
        )
    except (KeyboardInterrupt, SystemExit) as error:
        record_failure(params, error, stage)
        logging.warning("%s interrupted during %s", filename or mid, stage)
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
        log_values(
            f"{filename or mid}: finished",
            {"status": return_status, "elapsed": f"{perf_counter() - started:.3f}s"},
            level=logging.INFO if return_status == "success" else logging.ERROR,
        )
    return return_status
