import pyrealsense2 as rs
import numpy as np
import cv2
from time import time

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
for i in reversed(range(90)):
    frames = pipeline.wait_for_frames()
    # cv.imshow("Camera", frame)
    # cv.waitKey(1)
    if frames is None:
        print("No frame received. TERMINATE!")
        sys.exit()
    if not i % 30:
        print(i/30)  # count down 3, 2, 1 sec

# Init timer for FPS computing
start_stamp = time()
frame_counts = 0
frame_rate = 0.

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame is None:
            print("No frame received. TERMINATE!")
            break
        # Convert images to numpy arrays

        color_image = np.asanyarray(color_frame.get_data())

        # Show the stacked images
        cv2.imshow('RealSense', color_image)

        # Log frame rate
        frame_counts += 1
        since_start = time() - start_stamp
        frame_rate = frame_counts / since_start
        print(f"frame rate: {frame_rate}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
