import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline() # type: ignore
config = rs.config() # type: ignore

# Enable depth and color streams
config.enable_stream(rs.stream.color, 400, 300, rs.format.rgb8, 60) # type: ignore

# Start streaming
pipeline.start(config)
for i in reversed(range(180)):
    frames = pipeline.wait_for_frames()
    # cv.imshow("Camera", frame)
    # cv.waitKey(1)
    if frames is None:
        print("No frame received. TERMINATE!")
        sys.exit()
    if not i % 60:
        print(i/60)  # count down 3, 2, 1 sec

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
            pipeline.stop()
            cv2.destroyAllWindows()
            sys.exit()
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Show the stacked images
        cv2.imshow('RealSense', color_image)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
