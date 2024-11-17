# ROS2 RealSense, LiDAR, and Joystick Data Collection

import os
import sys
import json
import csv
from datetime import datetime
import pygame
import pyrealsense2 as rs
import numpy as np
import cv2
import serial
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

# Load configs
params_file_path = os.path.join(sys.path[0], 'test_config.json')
with open(params_file_path) as params_file:
    params = json.load(params_file)

# Constants
STEERING_AXIS = params['steering_joy_axis']
THROTTLE_AXIS = params['throttle_joy_axis']
RECORD_BUTTON = params['record_btn']
STOP_BUTTON = params['stop_btn']
MAX_LIDAR_RANGE = 25.0  # Maximum range for LiDAR readings

# LiDAR Node for ROS2
class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_node')
        self.lidar_data = []
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

    def lidar_callback(self, msg):
        # Replace 'inf' with MAX_LIDAR_RANGE and store ranges
        self.lidar_data = [r if np.isfinite(r) else MAX_LIDAR_RANGE for r in msg.ranges]

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
lidar_data_dir = os.path.join(data_dir, 'lidar/')
label_path = os.path.join(data_dir, 'labels.csv')
os.makedirs(combined_image_dir, exist_ok=True)
os.makedirs(lidar_data_dir, exist_ok=True)

# Initialize RealSense camera pipeline for RGB and Depth
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 60)
pipeline.start(config)

# Initialize variables
is_recording = False
frame_counts = 0
ax_val_st = 0.0
ax_val_th = 0.0

# Initialize Pygame for joystick handling
pygame.init()

# Initialize ROS2 LiDAR Node
rclpy.init()  # Initialize ROS2
lidar_node = LidarNode()  # Create the LiDAR node

# Write CSV headers if the file doesn't exist
if not os.path.exists(label_path):
    with open(label_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "steering", "throttle", "lidar_file"])

try:
    while True:
        # Capture RGB frames from RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue  # Skip if frames are not ready

        # Convert RGB to numpy arrays and resize to 120x160
        color_image = np.asanyarray(color_frame.get_data())
        resized_color_image = cv2.resize(color_image, (160, 120))

        # Display the RGB image
        cv2.imshow('RGB Stream', resized_color_image)

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
                    ser_pico.write("END,END\n".encode('utf-8'))
                    raise KeyboardInterrupt

        # Spin the LiDAR Node to process ROS callbacks
        rclpy.spin_once(lidar_node)

        # Get the latest LiDAR data
        lidar_ranges = lidar_node.lidar_data

        # Save data if recording is active
        if is_recording:
            # Save the RGB image
            image_name = f"{frame_counts}_rgb.png"
            cv2.imwrite(os.path.join(combined_image_dir, image_name), resized_color_image)

            # Save LiDAR data as a NumPy file
            lidar_file = f"{frame_counts}_lidar.npy"
            np.save(os.path.join(lidar_data_dir, lidar_file), np.array(lidar_ranges))

            # Log joystick values and LiDAR file with image name
            with open(label_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([image_name, ax_val_st, ax_val_th, lidar_file])

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
    rclpy.shutdown()
    cv2.destroyAllWindows()
