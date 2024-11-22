import time
import os
import glob
import cv2
import csv
import argparse
import subprocess
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message

def sort_files(files):
    return sorted(files, key=lambda f: os.path.getmtime(f)) # Sort on date created

def get_latest_directory(base_dir):
    """Get the latest directory in the given base directory."""
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not dirs:
        return None  # No directories found
    latest_dir = max(dirs, key=lambda d: os.path.getmtime(os.path.join(base_dir, d)))
    return os.path.join(base_dir, latest_dir)   

def get_files_from_directory(directory):
    files = glob.glob(os.path.join(directory, "**"), recursive=True)
    # Filter to include only files (exclude directories)
    files = [f for f in files if os.path.isfile(f)]
    return files if files else None

def extract_rosbag(bag_files, output_dir):
    """
    Extract data from multiple rosbag files sequentially.

    Args:
        bag_files (list): List of paths to rosbag `.db3` files. (Ensure they are sorted!)
        output_dir (str): Directory where extracted data will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Separate directories for color and depth images
    color_images_dir = os.path.join(output_dir, "color_images")
    os.makedirs(color_images_dir, exist_ok=True)

    depth_images_dir = os.path.join(output_dir, "depth_images")
    os.makedirs(depth_images_dir, exist_ok=True)

    # Directory for scans
    scan_dir = os.path.join(output_dir, "scans")
    os.makedirs(scan_dir, exist_ok=True)

    # CSV file for commands
    csv_file = open(os.path.join(output_dir, 'commands.csv'), 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['color_image_filename', 'depth_image_filename', 'linear_x', 'angular_z'])

    bridge = CvBridge()
    reader = SequentialReader()

    for bag_path in bag_files:
        if not bag_path.endswith(".db3"):
            continue
        print(f"Processing bag file: {bag_path}")
        storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = ConverterOptions('', '')

        try:
            reader.open(storage_options, converter_options)
        except RuntimeError as e:
            print(f"Failed to open bag file: {bag_path}, error: {e}")
            continue

        topic_types = reader.get_all_topics_and_types()
        type_map = {t.name: t.type for t in topic_types}

        while reader.has_next():
            topic, msg_data, timestamp = reader.read_next()

            # Process color images
            if topic == '/camera/color/image_raw':
                image_msg = deserialize_message(msg_data, get_message(type_map[topic]))
                cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
                timestamp_str = f"{image_msg.header.stamp.sec}_{image_msg.header.stamp.nanosec}"
                color_image_filename = os.path.join(color_images_dir, f"color_{timestamp_str}.jpg")
                cv2.imwrite(color_image_filename, cv_image)

            # Process depth images
            elif topic == '/camera/depth/image_rect_raw':
                depth_msg = deserialize_message(msg_data, get_message(type_map[topic]))
                depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
                timestamp_str = f"{depth_msg.header.stamp.sec}_{depth_msg.header.stamp.nanosec}"
                depth_image_filename = os.path.join(depth_images_dir, f"depth_{timestamp_str}.png")
                cv2.imwrite(depth_image_filename, depth_image)

            # Process command velocities
            elif topic == '/cmd_vel_stamped':
                # Deserialize the TwistStamped message
                command_msg = deserialize_message(msg_data, get_message(type_map[topic]))
                linear_x = command_msg.twist.linear.x
                angular_z = command_msg.twist.angular.z

                # Use the header's timestamp
                sec = command_msg.header.stamp.sec
                nanosec = command_msg.header.stamp.nanosec
                timestamp_str = f"{sec}_{nanosec}"

                # Match image filenames for commands
                color_image_filename = os.path.join(color_images_dir, f"color_{timestamp_str}.jpg")
                depth_image_filename = os.path.join(depth_images_dir, f"depth_{timestamp_str}.png")

                # Save the command to the CSV
                csv_writer.writerow([color_image_filename, depth_image_filename, linear_x, angular_z])

            # Process laser scan data
            elif topic == '/scan':
                scan_msg = deserialize_message(msg_data, get_message(type_map[topic]))
                scan_filename = os.path.join(scan_dir, f"scan_{scan_msg.header.stamp.sec}_{scan_msg.header.stamp.nanosec}.yaml")
                with open(scan_filename, 'w') as scan_file:
                    yaml.dump({
                        'ranges': list(scan_msg.ranges),
                        'intensities': list(scan_msg.intensities)
                    }, scan_file)

            # # Process camera info (color and depth)
            # elif topic == '/camera/color/camera_info':
            #     camera_info_msg = deserialize_message(msg_data, get_message(type_map[topic]))
            #     camera_info_filename = os.path.join(output_dir, f"color_camera_info.yaml")
            #     with open(camera_info_filename, 'w') as camera_info_file:
            #         yaml.dump({
            #             'width': camera_info_msg.width,
            #             'height': camera_info_msg.height,
            #             'K': list(camera_info_msg.K),
            #             'D': list(camera_info_msg.D),
            #             'distortion_model': camera_info_msg.distortion_model,
            #         }, camera_info_file)

            # elif topic == '/camera/depth/camera_info':
            #     camera_info_msg = deserialize_message(msg_data, get_message(type_map[topic]))
            #     camera_info_filename = os.path.join(output_dir, f"depth_camera_info.yaml")
            #     with open(camera_info_filename, 'w') as camera_info_file:
            #         yaml.dump({
            #             'width': camera_info_msg.width,
            #             'height': camera_info_msg.height,
            #             'K': list(camera_info_msg.K),
            #             'D': list(camera_info_msg.D),
            #             'distortion_model': camera_info_msg.distortion_model,
            #         }, camera_info_file)

            # # Process metadata topics (if needed)
            # elif topic in ['/camera/color/metadata', '/camera/depth/metadata']:
            #     metadata_msg = deserialize_message(msg_data, get_message(type_map[topic]))
            #     metadata_filename = os.path.join(output_dir, f"{topic.split('/')[-1]}_{metadata_msg.header.stamp.sec}.yaml")
            #     with open(metadata_filename, 'w') as metadata_file:
            #         yaml.dump({'data': str(metadata_msg)}, metadata_file)

            # # Extrinsics (if needed)
            # elif topic == '/camera/extrinsics/depth_to_color':
            #     extrinsics_msg = deserialize_message(msg_data, get_message(type_map[topic]))
            #     extrinsics_filename = os.path.join(output_dir, 'extrinsics.yaml')
            #     with open(extrinsics_filename, 'w') as extrinsics_file:
            #         yaml.dump({
            #             'rotation': extrinsics_msg.rotation,
            #             'translation': extrinsics_msg.translation
            #         }, extrinsics_file)

    csv_file.close()


def main():

    # Path to the topics configuration file
    config_path = os.path.join(
        get_package_share_directory('data_processing'),
        'config',
        'topics.yaml'
    )

    # Get the workspace directory from an environment variable or default to the install space
    workspace_dir = os.getenv('ROS_WORKSPACE', os.path.abspath(os.path.join(
        get_package_share_directory('data_processing'), '..', '..', '..', '..', # TODO is this a good practice? 
    )))

    rosbag_dir = os.path.join(workspace_dir, 'data', 'rosbags')

    latest_dir = get_latest_directory(rosbag_dir)

    extracted_rosbag_dir = os.path.join(workspace_dir, 'data', 'rosbags_extracted', os.path.basename(latest_dir))

    if latest_dir is None:
        raise FileNotFoundError # TODO improve this to log message

    db_file = get_files_from_directory(latest_dir) # NOTE in the case the user paused/resumed during data collection

    db_file = sort_files(db_file) # Maintain correct replay order
    
    # Load topics from the configuration file
    with open(config_path, 'r') as file:
        topics_config = yaml.safe_load(file)
    topics = topics_config['topics']  # List of topics from YAML

    extract_rosbag(db_file, extracted_rosbag_dir)

    # parser = argparse.ArgumentParser(description="Extract data from a rosbag.")
    # parser.add_argument('bag_path', type=str, help="Path to the input rosbag file")
    # parser.add_argument('output_dir', type=str, help="Directory where extracted data will be saved")
    # args = parser.parse_args()
    # extract_rosbag(args.bag_path, args.output_dir) # this needs to be modified
