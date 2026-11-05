"""Run the provided mock measurement examples through the production pipeline."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import yaml

from qupyt.hardware import signal_sources
from qupyt.hardware.device_handler import DeviceHandler, DynamicDeviceHandler

EXAMPLE_DIRECTORY = Path(__file__).resolve().parents[1] / "source/example_configs"
EXAMPLES = sorted(EXAMPLE_DIRECTORY.glob("*.yaml"))
measurement_module = importlib.import_module("qupyt.measurement_logic.run_measurement")


@pytest.mark.parametrize("example_file", EXAMPLES, ids=lambda path: path.stem)
def test_provided_mock_example_completes_a_measurement(
    example_file: Path,
    service: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    params: dict[str, Any] = yaml.safe_load(example_file.read_text(encoding="utf-8"))
    assert params["sensor"]["type"] == "MockCam"
    assert params["synchroniser"]["type"] == "MockSynchroniser"
    for device_group in ("static_devices", "dynamic_devices"):
        assert all(
            device["device_type"] == "Mock"
            for device in (params[device_group] or {}).values()
        )

    # Resolve the examples' launch-directory-specific Windows paths in the test.
    generator_name = str(params["ps_path"]).replace("\\", "/").rsplit("/", 1)[-1]
    params["ps_path"] = str(
        EXAMPLE_DIRECTORY.parent / "example_pulse_sequences" / generator_name
    )
    # Keep pulse parameters, frame counts, sweeps, and dynamic steps as supplied.
    # Bound acquisition and saved-array costs by using one average and a small ROI.
    params["averages"] = 1
    params["sensor"]["config"]["image_roi"] = [2, 2, 0, 0]
    monkeypatch.setattr(measurement_module, "sleep", lambda _: None)
    monkeypatch.setattr(measurement_module, "tqdm", lambda iterable, **_: iterable)
    monkeypatch.setattr(signal_sources, "sleep", lambda _: None)
    caplog.set_level(logging.WARNING)
    instruction = tmp_path / "instruction.yaml"
    instruction.write_text(yaml.safe_dump(params), encoding="utf-8")
    static = DeviceHandler({})
    dynamic = DynamicDeviceHandler(
        {}, number_dynamic_steps=int(params["dynamic_steps"])
    )
    try:
        service._process_instruction(instruction, static, dynamic)
    finally:
        static.update_devices({})
        dynamic.update_devices({})

    assert not instruction.exists()
    assert not list(tmp_path.glob("instruction.yaml_failed_*")), caplog.text
    metadata_files = list(tmp_path.glob(f"{params['experiment_type']}_*.yaml"))
    assert len(metadata_files) == 1
    metadata = yaml.safe_load(metadata_files[0].read_text(encoding="utf-8"))
    assert metadata["measurement_status"] == "success", caplog.text
    assert "measurement_error" not in metadata
    data = np.load(metadata_files[0].with_suffix(".npy"))
    assert np.all(np.isfinite(data))
    assert np.any(data)
