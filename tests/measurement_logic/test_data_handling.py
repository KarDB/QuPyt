from qupyt.hardware.sensors import SensorFactory
from qupyt.measurement_logic.data_handling import Data
import pytest


# Tests if the number_measurements can be distributed accross the reference channels.
# Should pass if number_measurements is divisible by reference_channels.
@pytest.mark.parametrize("reference_channels", [1, 2, 10])  # nframes = 10 should be divisible.
def test_set_dims_from_sensor(reference_channels):
    dynamic_steps = 6
    nframes = 10
    roi = [10, 10]
    cam = SensorFactory.create_sensor("MockCam", {"number_measurements": nframes, "image_roi": roi})
    data = Data({"averaging_mode":"spread",
                 "dynamic_steps": dynamic_steps,
                 "live_compression": False,
                 "reference_channels": reference_channels})
    data.set_dims_from_sensor(cam)
    data.create_array()
    assert data.data.shape == (reference_channels, dynamic_steps, int(nframes/reference_channels), *roi)


# Tests if the number_measurements can be distributed accross the reference channels.
# Expected to raise ValueError becuase number_measurements not divisible by reference_channels.
@pytest.mark.parametrize("reference_channels", [3, 4])
def test_set_dims_from_sensor_refuse_dist_measurements_channels(reference_channels):
    dynamic_steps = 6
    nframes = 10
    roi = [10, 10]
    cam = SensorFactory.create_sensor("MockCam", {"number_measurements": nframes, "image_roi": roi})
    data = Data({"averaging_mode": "spread",
                 "dynamic_steps": dynamic_steps,
                 "live_compression": False,
                 "reference_channels": reference_channels})
    data.set_dims_from_sensor(cam)
    with pytest.raises(ValueError):
        data.create_array()


# Tests the correct Data container creation for
# various combinations of ROIs / sensor dimensions.
@pytest.mark.parametrize("reference_channels", [1, 2])
@pytest.mark.parametrize("image_roi", [[10, 10], [1], [13], [2, 10, 3, 2]])
def test_set_dims_from_sensor_various_rois(reference_channels, image_roi):
    dynamic_steps = 6
    nframes = 10
    cam = SensorFactory.create_sensor("MockCam", {"number_measurements": nframes, "image_roi": image_roi})
    # Bypass the roi parsing mechanism of the Mock Sensor to test the behaviour of the Data cotainer. 
    cam.roi_shape = image_roi
    data = Data({"averaging_mode": "spread",
                 "dynamic_steps": dynamic_steps,
                 "live_compression": False,
                 "reference_channels": reference_channels})
    data.set_dims_from_sensor(cam)
    data.create_array()
    assert data.data.shape == (reference_channels, dynamic_steps, int(nframes/reference_channels), *image_roi)
    data.update_data(cam.acquire_data(), 0, 0)


# Tests the correct Data container ROI compression for
# various combinations of ROIs / sensor dimensions.
@pytest.mark.parametrize("reference_channels", [1, 2])
@pytest.mark.parametrize("image_roi", [[10, 10], [1], [13], [2, 10, 3, 2]])
def test_set_dims_from_sensor_compression(reference_channels, image_roi):
    dynamic_steps = 6
    nframes = 10
    cam = SensorFactory.create_sensor("MockCam", {"number_measurements": nframes, "image_roi": image_roi})
    # Bypass the roi parsing mechanism of the Mock Sensor to test the behaviour of the Data cotainer. 
    cam.roi_shape = image_roi
    data = Data({"averaging_mode": "spread",
                 "dynamic_steps": dynamic_steps,
                 "live_compression": True,
                 "reference_channels": reference_channels})
    data.set_dims_from_sensor(cam)
    data.create_array()
    assert data.data.shape == (reference_channels, dynamic_steps, int(nframes/reference_channels), 1)
    data.update_data(cam.acquire_data(), 0, 0)


# No dimension of the data container can be zero
# => This would reduce the overall size of the array to 0.
def test_data_container_refuse_0_dyn_steps():
    dynamic_steps = 0
    with pytest.raises(ValueError):
        data = Data({"averaging_mode": "spread", "dynamic_steps": dynamic_steps})
