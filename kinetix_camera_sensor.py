 from pyvcam import pvc
from pyvcam.camera import Camera

# Initialize the PVCAM library
pvc.init_pvcam()

# Open the camera
cam = Camera()
cam.open()

# Check if camera is open and ready
if cam.is_open():
    print("Kinetix camera initialized successfully.")
else:
    print("Failed to initialize the Kinetix camera.")

