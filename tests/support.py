"""Types for shared fixtures; the implementations remain in QuPyt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, TypeAlias
from unittest.mock import Mock

from qupyt.hardware.device_handler import DeviceHandler, DynamicDeviceHandler
from qupyt.hardware.sensors import Sensor
from qupyt.hardware.synchronisers import Synchroniser
from qupyt.measurement_logic.data_handling import Data


@dataclass
class MeasurementEnvironment:
    static: DeviceHandler
    dynamic: DynamicDeviceHandler
    sensor: Sensor
    acquisition: Mock
    synchroniser: Synchroniser
    params: dict[str, Any]
    directory: Path
    sensor_close: Mock
    synchroniser_close: Mock
    sequence_upload: Mock | None = None


class NumericDataFactory(Protocol):
    def __call__(
        self, mode: str = "spread", compressed: bool = False, chunk_size: int = 0
    ) -> Data: ...


PipelineFixture: TypeAlias = tuple[ModuleType, dict[str, Any], Mock, Mock, Mock]
SequenceDirectory: TypeAlias = tuple[Path, dict[str, Any], Path]
