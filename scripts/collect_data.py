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
STEERING_CENTER = params['steering_center']
STEERING_RANGE = params['steering_range']
THROTTLE_AXIS = params['throttle_joy_axis']
THROTTLE_STALL = params['throttle_stall']
THROTTLE_FWD_RANGE = params['throttle_fwd_range']
THROTTLE_REV_RANGE = params['throttle_rev_range']
THROTTLE_LIMIT = params['throttle_limit']
RECORD_BUTTON = params['record_btn']
STOP_BUTTON = params['stop_btn']

# Initialize hardware (serial communication and joystick)
try:
    ser_pico = serial.Serial(port='/dev/ttyACM1', baudrate=115200)
except:
    ser_pico = serial.Serial(port='/dev/ttyACM0', baudrate=115200)

pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)

# Create data directories
data_dir = os.path.join('data', datetime.now().strftime("%Y-%m-%d-%H-%M"))
image_dir = os.path.join(data_dir, 'images/')
depth_image_dir= os.path.join(data_dir, 'depth_images/')
label_path = os.path.join(data_dir, 'labels.csv')
os.makedirs(image_dir, exist_ok=True)
os.makedirs(depth_image_dir, exist_ok=True)

# Initialize RealSense camera pipeline
pipeline = rs.pipeline() #type:ignore
config = rs.config() #type:ignore

# Configure RealSense streams (set resolution and FPS)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90) #type:ignore
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60) #type:ignore

# Start streaming from the camera
pipeline.start(config)

# Initialize variables
is_recording = False
frame_counts = 0
ax_val_st = 0.
ax_val_th = 0.

# Initialize frame rate tracking variables
prev_time_rgb = time()
prev_time_depth = time()
frame_count_rgb = 0
frame_count_depth = 0
fps_rgb = 0
fps_depth = 0

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

        # Calculate RGB frame rate
        frame_count_rgb += 1
        current_time_rgb = time()
        if current_time_rgb - prev_time_rgb >= 1.0:
            fps_rgb = frame_count_rgb / (current_time_rgb - prev_time_rgb)
            print(f"RGB Frame Rate: {fps_rgb:.2f} FPS")
            prev_time_rgb = current_time_rgb
            frame_count_rgb = 0

        # Calculate Depth frame rate
        frame_count_depth += 1
        current_time_depth = time()
        if current_time_depth - prev_time_depth >= 1.0:
            fps_depth = frame_count_depth / (current_time_depth - prev_time_depth)
            print(f"Depth Frame Rate: {fps_depth:.2f} FPS")
            prev_time_depth = current_time_depth
            frame_count_depth = 0

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
            print(e)
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

        # Calaculate steering and throttle value
        act_st = -ax_val_st
        act_th = -ax_val_th # throttle action: -1: max forward, 1: max backward

        # Encode steering value to dutycycle in nanosecond
        duty_st = STEERING_CENTER - STEERING_RANGE + int(STEERING_RANGE * (act_st + 1))
        # Encode throttle value to dutycycle in nanosecond
        if act_th > 0:
            duty_th = THROTTLE_STALL + int(THROTTLE_FWD_RANGE * min(act_th, THROTTLE_LIMIT))
        elif act_th < 0:
            duty_th = THROTTLE_STALL + int(THROTTLE_REV_RANGE * max(act_th, -THROTTLE_LIMIT))
        else:
            duty_th = THROTTLE_STALL 
        duty_st = round(duty_st, 2)
        duty_th = round(duty_th, 2)
        msg = (str(duty_st) + "," + str(duty_th) + "\n").encode('utf-8')

        # Send control signals to the microcontroller
        ser_pico.write(msg)

        # Save data if recording is active
        if is_recording:
            # Save the RGB and depth images at a lower resolution
            resized_color_image = cv2.resize(color_image, (320, 240)) #120, 160
            resized_depth_image = cv2.resize(depth_colormap, (320, 240)) #120, 160

            # Save the RGB and depth images
            #print(f"image dir: {image_dir}") #Print direction(s) for images
            cv2.imwrite(os.path.join(image_dir, f"{frame_counts}_color.jpg"), resized_color_image)
            #print(f"depth image dir: {depth_image_dir}") #Print direction(s) for depth images 
            cv2.imwrite(os.path.join(depth_image_dir, f"{frame_counts}_depth.png"), resized_depth_image)
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
