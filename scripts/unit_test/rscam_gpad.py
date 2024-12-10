"""
Integrated test with controller, pico usb communication, throttle motor.
"""
import sys
import os
import json
from time import time
import pygame
import cv2 as cv
import pyrealsense2 as rs


# SETUP
# Load configs
params_file_path = os.path.join(os.path.dirname(sys.path[0]), 'configs.json')
params_file = open(params_file_path)
params = json.load(params_file)
# Constants
STEERING_AXIS = params['steering_joy_axis']
THROTTLE_AXIS = params['throttle_joy_axis']
RECORD_BUTTON = params['record_btn']
STOP_BUTTON = params['stop_btn_x']
# Init controller
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)
# Init camera
cv.startWindowThread()
cam = rs.pipeline() # type: ignore
config = rs.config() # type: ignore
config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 25) # type: ignore
config.enable_stream(rs.stream.color, 320, 240, rs.format.bgr8, 25) # type: ignore
cam.start(config)
for i in reversed(range(75)):
    frames = cam.wait_for_frames()
    if frames is None:
        print("No frame received. TERMINATE!")
        sys.exit()
    # cv.imshow("Camera", frame)
    # cv.waitKey(1)
    if not i % 25:
        print(i/25)  # count down 3, 2, 1 sec
# Init timer for FPS computing
start_stamp = time()
frame_counts = 0
ave_frame_rate = 0.
# Init joystick axes values
ax_val_st = 0.
ax_val_th = 0.

# MAIN LOOP
try:
    while True:
        frames = cam.wait_for_frames() # read image
        if frames is None:
            print("No frame received. TERMINATE!")
            cv.destroyAllWindows()
            pygame.quit()
            sys.exit()
        cv.imshow('camera', frame)
        for e in pygame.event.get(): # read controller input
            if e.type == pygame.JOYAXISMOTION:
                ax_val_st = round((js.get_axis(STEERING_AXIS)), 2)  # keep 2 decimals
                ax_val_th = round((js.get_axis(THROTTLE_AXIS)), 2)  # keep 2 decimals
            elif e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(RECORD_BUTTON):
                    print("Record Pressed.")
                elif js.get_button(STOP_BUTTON): # emergency stop
                    print("E-STOP PRESSED. TERMINATE!")
                    cv.destroyAllWindows()
                    pygame.quit()
                    sys.exit()
        # Calaculate steering and throttle value
        act_st = ax_val_st
        act_th = -ax_val_th # throttle action: -1: max forward, 1: max backward
        # Log action
        action = [act_st, act_th]
        print(f"action: {action}")
        # Log frame rate
        frame_counts += 1
        since_start = time() - start_stamp
        frame_rate = frame_counts / since_start
        print(f"frame rate: {frame_rate}")
        # Display video
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        image_stack = np.hstack((color_image, depth_colormap))
        cv.imshow("RealSense", image_stack)
        cv.waitKey(1)
        # Press "q" to quit
        if cv.waitKey(1)==ord('q'):
            cv.destroyAllWindows()
            pygame.quit()
            sys.exit()

# Take care terminal signal (Ctrl-c)
except KeyboardInterrupt:
    cv.destroyAllWindows()
    pygame.quit()
    sys.exit()
