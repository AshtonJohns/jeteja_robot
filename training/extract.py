import path
import os
import cv2
import csv
import yaml
import jeteja_launch.config.master_config as  master_config
import jeteja_launch.scripts.image_processing as image_processing
from file_utilities import get_files_from_directory, get_latest_directory, sort_files, get_files_from_subdirectory
from ament_index_python.packages import get_package_share_directory
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message

COLOR_ENCODING = master_config.COLOR_ENCODING
DEPTH_ENCODING = master_config.DEPTH_ENCODING


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
    csv_writer.writerow(['color_image_filename', 'depth_image_filename', 'motor_pwm', 'steering_pwm'])

    bridge = image_processing.get_cvbridge()
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
            topic, msg_data, timestamp = reader.read_next() # NOTE this timestamp is not used

            # Process color images
            if topic == '/camera/camera/color/image_raw':
                topic_type = type_map[topic]
                image_msg = image_processing.deserialize_ros_message(msg_data,topic_type)
                cv_image = image_processing.ros_to_cv(msg_data, topic_type, color=True)
                timestamp_str = f"{image_msg.header.stamp.sec}_{image_msg.header.stamp.nanosec}"
                color_image_filename = os.path.join(color_images_dir, f"color_{timestamp_str}.jpg")
                cv2.imwrite(color_image_filename, cv_image)

            # Process depth images
            elif topic == '/camera/camera/depth/image_rect_raw':
                topic_type = type_map[topic]
                depth_msg = image_processing.deserialize_ros_message(msg_data,topic_type)
                depth_image = image_processing.ros_to_cv(msg_data, topic_type, depth=True)
                timestamp_str = f"{depth_msg.header.stamp.sec}_{depth_msg.header.stamp.nanosec}"
                depth_image_filename = os.path.join(depth_images_dir, f"depth_{timestamp_str}.png")
                cv2.imwrite(depth_image_filename, depth_image)

            # Process command velocities
            elif topic == '/pwm_signals':
                # Deserialize the TwistStamped message
                command_msg = deserialize_message(msg_data, get_message(type_map[topic]))
                motor_pwm = command_msg.motor_pwm
                steering_pwm = command_msg.steering_pwm

                # Use the header's timestamp
                sec = command_msg.stamp.sec
                nanosec = command_msg.stamp.nanosec
                timestamp_str = f"{sec}_{nanosec}"

                # Match image filenames for commands
                color_image_filename = f"color_{timestamp_str}.jpg"
                depth_image_filename = f"depth_{timestamp_str}.png"

                # Save the command to the CSV
                csv_writer.writerow([color_image_filename, depth_image_filename, motor_pwm, steering_pwm])

            # Process laser scan data
            elif topic == '/scan':
                scan_msg = deserialize_message(msg_data, get_message(type_map[topic]))
                scan_filename = os.path.join(scan_dir, f"scan_{scan_msg.header.stamp.sec}_{scan_msg.header.stamp.nanosec}.yaml")
                with open(scan_filename, 'w') as scan_file:
                    yaml.dump({
                        'ranges': list(scan_msg.ranges),
                        'intensities': list(scan_msg.intensities)
                    }, scan_file)

    csv_file.close()


def main():

    rosbag_dir = os.path.join('.', 'data', 'rosbags')

    latest_dir = get_latest_directory(rosbag_dir)

    extracted_rosbag_dir = os.path.join('.', 'data', 'rosbags_extracted', os.path.basename(latest_dir))

    if latest_dir is None:
        raise FileNotFoundError # TODO improve this to log message

    db_file = get_files_from_subdirectory(latest_dir) # NOTE in the case the user paused/resumed during data collection

    db_file = sort_files(db_file) # Maintain correct replay order

    extract_rosbag(db_file, extracted_rosbag_dir)


if __name__ == '__main__':
    main()

