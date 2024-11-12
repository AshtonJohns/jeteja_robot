# ROS2 RealSense and Joystick Data Collection

import os
import sys
import json
import csv
from time import time, sleep
from datetime import datetime
import pygame
import pyrealsense2 as rs
import numpy as np
import cv2
import serial

# Load configs
params_file_path = os.path.join(sys.path[0], 'test_config.json')
with open(params_file_path) as params_file:
    params = json.load(params_file)

# Constants
STEERING_AXIS = params['steering_joy_axis']
STEERING_CENTER = params['steering_center']
STEERING_RANGE = params['steering_range']
THROTTLE_AXIS = params['throttle_joy_axis']
THROTTLE_STALL = params['throttle_stall']
THROTTLE_FWD_RANGE = params['throttle_fwd_range']
THROTTLE_REV_RANGE = params['throttle_rev_range']
THROTTLE_LIMIT = params['throttle_limit']
THROTTLE_MIN = params['throttle_min']
RECORD_BUTTON = params['record_btn']
STOP_BUTTON = params['stop_btn']

# Initialize hardware (serial communication and joystick)
try:
    ser_pico = serial.Serial(port='/dev/ttyACM1', baudrate=115200)
except:
    ser_pico = serial.Serial(port='/dev/ttyACM0', baudrate=115200)

# Initialize joystick and directories for saving data
pygame.joystick.init()
js = pygame.joystick.Joystick(0)

data_dir = os.path.join('data', datetime.now().strftime("%Y-%m-%d-%H-%M"))
combined_image_dir = os.path.join(data_dir, 'combined_images/')
label_path = os.path.join(data_dir, 'labels.csv')
os.makedirs(combined_image_dir, exist_ok=True)

# Initialize RealSense camera pipeline for RGB and Depth
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)  # Enable depth stream

# Start streaming from the camera
pipeline.start(config)

# Initialize variables
is_recording = False
frame_counts = 0
ax_val_st = 0.
ax_val_th = 0.

# Initialize Pygame for joystick handling
pygame.init()

try:
    while True:
        # Capture RGB and Depth frames from RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue  # Skip if frames are not ready

        # Convert RGB and depth to numpy arrays and resize to 120x160
        color_image = np.asanyarray(color_frame.get_data())
        resized_color_image = cv2.resize(color_image, (160, 120))
        
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        resized_depth_image = cv2.resize(depth_image_normalized, (160, 120)).astype(np.uint8)

        # Display the combined image
        cv2.imshow('Combined Stream - RGB and Depth', resized_color_image)

        # Handle joystick inputs
        for e in pygame.event.get():
            if e.type == pygame.JOYAXISMOTION:
                ax_val_st = round(js.get_axis(STEERING_AXIS), 2)
                ax_val_th = round(js.get_axis(THROTTLE_AXIS), 2)
            elif e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(RECORD_BUTTON):
                    print("Collecting data")
                    is_recording = not is_recording  # Toggle recording
                elif js.get_button(STOP_BUTTON):
                    print("E-STOP PRESSED. TERMINATE!")
                    msg = ("END,END\n").encode('utf-8')
                    ser_pico.write(msg)
                    raise KeyboardInterrupt

        # Calculate steering and throttle values
        act_st = -ax_val_st
        act_th = -ax_val_th
        duty_st = STEERING_CENTER - STEERING_RANGE + int(STEERING_RANGE * (act_st + 1))
        
        # Refined throttle control with correct forward and reverse mapping
        if act_th > 0:
            # Forward motion with variable speed control
            duty_th = THROTTLE_STALL + int((THROTTLE_FWD_RANGE - THROTTLE_STALL) * act_th)
        elif act_th < 0:
            # Reverse motion with variable speed control
            duty_th = THROTTLE_STALL - int((THROTTLE_STALL - THROTTLE_REV_RANGE) * abs(act_th))
        else:
            # No throttle
            duty_th = THROTTLE_STALL

        # Send control signals to the microcontroller
        ser_pico.write((f"{duty_st},{duty_th}\n").encode('utf-8'))

        # Save data if recording is active
        if is_recording:
            # Save the combined image with RGB and depth
            cv2.imwrite(os.path.join(combined_image_dir, f"{frame_counts}_combined.png"), resized_color_image)

            # Log joystick values with image name
            with open(label_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f"{frame_counts}_combined.png", ax_val_st, ax_val_th])

            frame_counts += 1  # Increment frame counter

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Terminated by user.")

finally:
    # Cleanup
    pipeline.stop()
    pygame.quit()
    ser_pico.close()
    cv2.destroyAllWindows()