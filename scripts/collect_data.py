"""
import os
import sys
import json
import csv
from time import time
from datetime import datetime
from hardware import setup_camera, setup_serial, setup_joystick, setup_led, encode_steering, encode_throttle
import pygame
import cv2 as cv


# SETUP
# Load configs
params_file_path = os.path.join(sys.path[0], 'configs.json')
with open(params_file_path) as params_file:
    params = json.load(params_file)

# Constants
STEERING_AXIS = params['steering_joy_axis']
THROTTLE_AXIS = params['throttle_joy_axis']
RECORD_BUTTON = params['record_btn']
STOP_BUTTON = params['stop_btn']

# Initialize hardware
headlight = setup_led(params['led_pin'])
ser_pico = setup_serial(port='/dev/ttyACM0', baudrate=115200)
cam = setup_camera((120, 160), frame_rate=20)
js = setup_joystick()

# Create data directories
data_dir = os.path.join('data', datetime.now().strftime("%Y-%m-%d-%H-%M"))
image_dir = os.path.join(data_dir, 'images/')
label_path = os.path.join(data_dir, 'labels.csv')
os.makedirs(image_dir, exist_ok=True)

# Initialize variables
is_recording = False
frame_counts = 0
start_time = time()

# MAIN LOOP
try:
    while True:
        ret, frame = cam.read()
        if frame is None:
            print("No frame received. TERMINATE!")
            break

        # Controller input
        for e in pygame.event.get():
            if e.type == pygame.JOYAXISMOTION:
                ax_val_st = round(js.get_axis(STEERING_AXIS), 2)
                ax_val_th = round(js.get_axis(THROTTLE_AXIS), 2)
            elif e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(RECORD_BUTTON):
                    is_recording = not is_recording
                    headlight.toggle()
                elif js.get_button(STOP_BUTTON):
                    print("E-STOP PRESSED. TERMINATE!")
                    break

        # Encode and transmit control signals
        duty_st = int(encode_steering(ax_val_st, params))
        duty_th = int(encode_throttle(ax_val_th, params))
        ser_pico.write(f"{duty_st},{duty_th}\n".encode('utf-8'))

        # Log data
        if is_recording:
            cv.imwrite(os.path.join(image_dir, f"{frame_counts}.jpg"), frame)
            with open(label_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f"{frame_counts}.jpg", ax_val_st, ax_val_th])

        frame_counts += 1

except KeyboardInterrupt:
    print("Terminated by user.")
finally:
    cam.stop()
    pygame.quit()
    ser_pico.close()
    headlight.off()
    cv.destroyAllWindows()
    
"""
# ROS2 RealSense and RP LiDAR Integration 
import os
import sys
import json
import csv
from time import time
from datetime import datetime
import pygame
import pyrealsense2 as rs  # Import the RealSense library
import numpy as np
import cv2
import serial

# Load configs
params_file_path = os.path.join(sys.path[0], 'config_new.json')
with open(params_file_path) as params_file:
    params = json.load(params_file)

# Constants
STEERING_AXIS = params['steering_joy_axis']
THROTTLE_AXIS = params['throttle_joy_axis']
RECORD_BUTTON = params['record_btn']
STOP_BUTTON = params['stop_btn_x']

# Initialize hardware (serial communication and joystick)
ser_pico = serial.Serial(port='/dev/ttyACM0', baudrate=115200)
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)

# Create data directories
data_dir = os.path.join('data', datetime.now().strftime("%Y-%m-%d-%H-%M"))
image_dir = os.path.join(data_dir, 'images/')
label_path = os.path.join(data_dir, 'labels.csv')
os.makedirs(image_dir, exist_ok=True)

# Initialize RealSense camera pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure RealSense streams (set resolution and FPS)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)

# Start streaming from the camera
pipeline.start(config)

# Initialize variables
is_recording = False
frame_counts = 0

# Initialize Pygame for joystick handling
pygame.init()

try:
    while True:
        # Wait for a new set of frames from the camera
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue  # Skip if frames are not ready

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap to depth image for visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        # Display the RGB and depth images side by side
        stacked_images = np.hstack((color_image, depth_colormap))
        cv2.imshow('RealSense Stream', stacked_images)

        # Handle joystick input events
        for e in pygame.event.get():
            if e.type == pygame.JOYAXISMOTION:
                ax_val_st = round(js.get_axis(STEERING_AXIS), 2)
                ax_val_th = round(js.get_axis(THROTTLE_AXIS), 2)
            elif e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(RECORD_BUTTON):
                    is_recording = not is_recording  # Toggle recording
                elif js.get_button(STOP_BUTTON):
                    print("E-STOP PRESSED. TERMINATE!")
                    raise KeyboardInterrupt

        # Send control signals to the microcontroller
        duty_st = int(encode_steering(ax_val_st, params))
        duty_th = int(encode_throttle(ax_val_th, params))
        ser_pico.write(f"{duty_st},{duty_th}\n".encode('utf-8'))

        # Save data if recording is active
        if is_recording:
            # Save the RGB and depth images
            cv2.imwrite(os.path.join(image_dir, f"{frame_counts}_color.jpg"), color_image)
            cv2.imwrite(os.path.join(image_dir, f"{frame_counts}_depth.png"), depth_image)

            # Log the joystick values
            with open(label_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f"{frame_counts}_color.jpg", f"{frame_counts}_depth.png", ax_val_st, ax_val_th])

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
