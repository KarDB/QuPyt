from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.typing import ArrayLike

from qupyt.hardware.sensors import SensorFactory
from qupyt.measurement_logic.data_handling import Data
from tests.support import NumericDataFactory


# Tests if the number_measurements can be distributed accross the reference channels.
# Should pass if number_measurements is divisible by reference_channels.
@pytest.mark.parametrize(
    "reference_channels", [1, 2, 10]
)  # nframes = 10 should be divisible.
def test_set_dims_from_sensor(reference_channels: int) -> None:
    dynamic_steps = 6
    nframes = 10
    roi = [10, 10]
    cam = SensorFactory.create_sensor(
        "MockCam", {"number_measurements": nframes, "image_roi": roi}
    )
    data = Data(
        {
            "averaging_mode": "spread",
            "dynamic_steps": dynamic_steps,
            "live_compression": False,
            "reference_channels": reference_channels,
        }
    )
    data.set_dims_from_sensor(cam)
    data.create_array()
    assert data.data.shape == (
        reference_channels,
        1,
        dynamic_steps,
        int(nframes / reference_channels),
        *roi,
    )


# Tests if the number_measurements can be distributed accross the reference channels.
# Expected to raise ValueError becuase number_measurements not divisible by reference_channels.
@pytest.mark.parametrize("reference_channels", [3, 4])
def test_set_dims_from_sensor_refuse_dist_measurements_channels(
    reference_channels: int,
) -> None:
    dynamic_steps = 6
    nframes = 10
    roi = [10, 10]
    cam = SensorFactory.create_sensor(
        "MockCam", {"number_measurements": nframes, "image_roi": roi}
    )
    data = Data(
        {
            "averaging_mode": "spread",
            "dynamic_steps": dynamic_steps,
            "live_compression": False,
            "reference_channels": reference_channels,
        }
    )
    data.set_dims_from_sensor(cam)
    with pytest.raises(ValueError):
        data.create_array()


# Tests the correct Data container creation for
# various combinations of ROIs / sensor dimensions.
@pytest.mark.parametrize("reference_channels", [1, 2])
@pytest.mark.parametrize("image_roi", [[10, 10], [1], [13], [2, 10, 3, 2]])
def test_set_dims_from_sensor_various_rois(
    reference_channels: int, image_roi: list[int]
) -> None:
    dynamic_steps = 6
    nframes = 10
    cam = SensorFactory.create_sensor(
        "MockCam", {"number_measurements": nframes, "image_roi": image_roi}
    )
    # Bypass the roi parsing mechanism of the Mock Sensor to test the behaviour of the Data cotainer.
    cam.roi_shape = image_roi
    data = Data(
        {
            "averaging_mode": "spread",
            "dynamic_steps": dynamic_steps,
            "live_compression": False,
            "reference_channels": reference_channels,
        }
    )
    data.set_dims_from_sensor(cam)
    data.create_array()
    assert data.data.shape == (
        reference_channels,
        1,
        dynamic_steps,
        int(nframes / reference_channels),
        *image_roi,
    )
    data.update_data(cam.acquire_data(), 0, 0, 0)


# Tests the correct Data container ROI compression for
# various combinations of ROIs / sensor dimensions.
@pytest.mark.parametrize("reference_channels", [1, 2])
@pytest.mark.parametrize("image_roi", [[10, 10], [1], [13], [2, 10, 3, 2]])
def test_set_dims_from_sensor_compression(
    reference_channels: int, image_roi: list[int]
) -> None:
    dynamic_steps = 6
    nframes = 10
    cam = SensorFactory.create_sensor(
        "MockCam", {"number_measurements": nframes, "image_roi": image_roi}
    )
    # Bypass the roi parsing mechanism of the Mock Sensor to test the behaviour of the Data cotainer.
    cam.roi_shape = image_roi
    data = Data(
        {
            "averaging_mode": "spread",
            "dynamic_steps": dynamic_steps,
            "live_compression": True,
            "reference_channels": reference_channels,
        }
    )
    data.set_dims_from_sensor(cam)
    data.create_array()
    assert data.data.shape == (
        reference_channels,
        1,
        dynamic_steps,
        int(nframes / reference_channels),
        1,
    )
    data.update_data(cam.acquire_data(), 0, 0, 0)


# No dimension of the data container can be zero
# => This would reduce the overall size of the array to 0.
def test_data_container_refuse_0_dyn_steps() -> None:
    dynamic_steps = 0
    with pytest.raises(ValueError):
        Data({"averaging_mode": "spread", "dynamic_steps": dynamic_steps})


@pytest.fixture
def numeric_data() -> NumericDataFactory:
    def create(
        mode: str = "spread", compressed: bool = False, chunk_size: int = 0
    ) -> Data:
        data = Data(
            {
                "averaging_mode": mode,
                "dynamic_steps": 2,
                "ps_steps": 2,
                "live_compression": compressed,
                "save_in_chunks": chunk_size,
            }
        )
        sensor = SensorFactory.create_sensor(
            "MockCam", {"number_measurements": 4, "image_roi": [2]}
        )
        data.set_dims_from_sensor(sensor)
        data.create_array()
        data.set_filename(("numeric", "test"))
        return data

    return create


@pytest.mark.parametrize(
    "mode,compressed,expected",
    [
        ("spread", False, [[[2, 4], [6, 8]], [[10, 14], [18, 22]]]),
        ("sum", False, [[[8, 12]], [[28, 36]]]),
        ("spread", True, [[[3], [7]], [[12], [20]]]),
        ("sum", True, [[[10]], [[32]]]),
    ],
)
def test_accumulation_values_and_step_isolation(
    numeric_data: NumericDataFactory, mode: str, compressed: bool, expected: ArrayLike
) -> None:
    data = numeric_data(mode, compressed)
    frames = np.array([[2, 4], [10, 14], [6, 8], [18, 22]], dtype=float)
    original = frames.copy()

    data.update_data(frames, 1, 1, 0)
    np.testing.assert_array_equal(data.data[:, 1, 1], expected)
    data.update_data(frames, 1, 1, 1)
    np.testing.assert_array_equal(data.data[:, 1, 1], np.array(expected) * 2)
    assert not np.any(data.data[:, 0])
    assert not np.any(data.data[:, 1, 0])
    assert not np.any(data.chunk_data)
    np.testing.assert_array_equal(frames, original)


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("shape", [(2, 2), (4, 1), (4, 2, 1)])
def test_wrong_sensor_shape_does_not_change_buffers(
    numeric_data: NumericDataFactory, compressed: bool, shape: tuple[int, ...]
) -> None:
    data = numeric_data(compressed=compressed, chunk_size=2)
    with pytest.raises(ValueError, match="Sensor returned data shape"):
        data.update_data(np.ones(shape), 0, 0, 0)
    assert not np.any(data.data)
    assert not np.any(data.chunk_data)
    assert data.chunk_save_index == 0


@pytest.mark.parametrize("mode", ["sum", "spread"])
@pytest.mark.parametrize("compressed", [False, True])
def test_chunks_are_averaged_independently(
    numeric_data: NumericDataFactory,
    mode: str,
    compressed: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data = numeric_data(mode, compressed, chunk_size=2)
    data.update_data(np.ones((4, 2)), 0, 0, 0)
    assert not list(tmp_path.glob("*.npy"))
    data.update_data(np.full((4, 2), 3.0), 0, 0, 1)
    first = np.load(tmp_path / "numeric_ch-0_test.npy")
    np.testing.assert_array_equal(first[:, 0, 0], 4 if mode == "sum" else 2)
    assert data.chunk_save_index == 1
    assert not np.any(data.chunk_data)

    data.update_data(np.full((4, 2), 5.0), 0, 0, 2)
    data.update_data(np.full((4, 2), 7.0), 0, 0, 3)
    second = np.load(tmp_path / "numeric_ch-1_test.npy")
    np.testing.assert_array_equal(second[:, 0, 0], 12 if mode == "sum" else 6)
    np.testing.assert_array_equal(np.load(tmp_path / "numeric_ch-0_test.npy"), first)
    np.testing.assert_array_equal(data.data[:, 0, 0], 32 if mode == "sum" else 16)
    assert data.chunk_save_index == 2
    assert not np.any(data.chunk_data)


def test_failed_chunk_write_preserves_buffers_for_diagnosis(
    numeric_data: NumericDataFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = numeric_data(chunk_size=2)
    error = OSError("disk full")
    monkeypatch.setattr(np, "save", Mock(side_effect=error))
    data.update_data(np.ones((4, 2)), 0, 0, 0)
    with pytest.raises(OSError) as caught:
        data.update_data(np.full((4, 2), 3.0), 0, 0, 1)
    assert caught.value is error
    assert data.chunk_save_index == 0
    np.testing.assert_array_equal(data.data[:, 0, 0], 4)
    np.testing.assert_array_equal(data.chunk_data, data.data)


@pytest.mark.parametrize("average_count,expected", [(None, 6), (0, 6), (3, 2)])
def test_saving_does_not_modify_accumulated_data(
    numeric_data: NumericDataFactory,
    tmp_path: Path,
    average_count: int | None,
    expected: ArrayLike,
) -> None:
    data = numeric_data()
    data.update_data(np.full((4, 2), 6.0), 0, 0, 0)
    original = data.data.copy()
    path = tmp_path / "result.npy"
    data.save(str(path), average_count=average_count)
    np.testing.assert_array_equal(np.load(path)[:, 0, 0], expected)
    np.testing.assert_array_equal(data.data, original)


@pytest.mark.parametrize(
    "configuration",
    [
        {"ps_steps": 0},
        {"reference_channels": 0},
        {"save_in_chunks": -1},
        {"number_measurements": 0},
        {"averaging_mode": "unknown"},
    ],
)
def test_invalid_data_configuration_rejected_before_allocation(
    configuration: dict[str, Any]
) -> None:
    data = Data(
        {
            "averaging_mode": "spread",
            "dynamic_steps": 1,
            "number_measurements": 2,
            "roi_shape": [1],
            **configuration,
        }
    )
    with pytest.raises(ValueError):
        data.create_array()
    assert not hasattr(data, "data")


def test_sum_accepts_unequal_number_of_frames_per_reference() -> None:
    data = Data(
        {
            "averaging_mode": "sum",
            "dynamic_steps": 1,
            "number_measurements": 3,
            "roi_shape": [1],
        }
    )
    data.create_array()
    data.update_data(np.array([[2.0], [7.0], [3.0]]), 0, 0, 0)
    np.testing.assert_array_equal(data.data[:, 0, 0, 0, 0], [5, 7])


@pytest.mark.parametrize("compressed", [False, True])
def test_sensor_dtype_is_used_and_compression_preserves_fractional_values(
    compressed: bool,
) -> None:
    sensor = SensorFactory.create_sensor(
        "MockCam", {"number_measurements": 2, "image_roi": [2]}
    )
    sensor.target_data_type = np.float32
    data = Data(
        {"averaging_mode": "spread", "dynamic_steps": 1, "live_compression": compressed}
    )
    data.set_dims_from_sensor(sensor)
    data.create_array()
    data.update_data(np.array([[1, 2], [3, 4]], dtype=np.float32), 0, 0, 0)
    if compressed:
        np.testing.assert_array_equal(data.data[:, 0, 0, 0], [[1.5], [3.5]])
        assert data.data.dtype == np.float64
    else:
        np.testing.assert_array_equal(data.data[:, 0, 0, 0], [[1, 2], [3, 4]])
        assert data.data.dtype == np.float32
