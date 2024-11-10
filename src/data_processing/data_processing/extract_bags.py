import os
import cv2
import csv
import argparse
import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message

def extract_rosbag(bag_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    csv_file = open(os.path.join(output_dir, 'commands.csv'), 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['image_filename', 'linear_x', 'angular_z'])

    # Initialize components
    bridge = CvBridge()
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    while reader.has_next():
        topic, msg_data, _ = reader.read_next()

        # Process images
        if topic == '/camera/camera/color/image_raw':
            # Ensure msg_data is interpreted as a sensor_msgs/Image message
            image_msg = deserialize_message(msg_data, get_message(type_map[topic]))
            cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            image_filename = os.path.join(images_dir, f"{image_msg.header.stamp.sec}_{image_msg.header.stamp.nanosec}.jpg")
            cv2.imwrite(image_filename, cv_image)

        # Process commands
        elif topic == '/cmd_vel':
            command_msg = deserialize_message(msg_data, get_message(type_map[topic]))
            linear_x = command_msg.linear.x
            angular_z = command_msg.angular.z
            # Save the command to CSV with image filename (if paired)
            csv_writer.writerow([image_filename, linear_x, angular_z])

    csv_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from a rosbag.")
    parser.add_argument('bag_path', type=str, help="Path to the input rosbag file")
    parser.add_argument('output_dir', type=str, help="Directory where extracted data will be saved")

    args = parser.parse_args()
    extract_rosbag(args.bag_path, args.output_dir)
    # extract_rosbag('~/my_tmp_data', '~/my_output_data')
