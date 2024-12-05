import os
import rclpy
import subprocess
from rclpy.node import Node
from std_msgs.msg import String

class RosbagManager(Node):
    def __init__(self):
        super().__init__('rosbag_manager')

        # Get parameters
        self.topics = self.declare_parameter('topics', ['']).get_parameter_value().string_array_value
        self.get_logger().info(f"Topics: {self.topics}")
        self.split_size = self.declare_parameter('split_size', '10000').get_parameter_value().integer_value
        self.get_logger().info(f"Will split rosbags at {self.split_size} MB ({self.split_size/1000} GB)")
        self.output_dir = self.declare_parameter('output_dir', 'data/rosbags').get_parameter_value().string_value
        self.get_logger().info(f"Output directory: {self.output_dir}")

        # Multiple database file support
        self.db_file_num = 0

        self.subscription = self.create_subscription(
            String,
            '/recording_status',
            self.status_callback,
            10
        )
        self.rosbag_process = None

    def status_callback(self, msg):
        if msg.data == 'pause':
            self.stop_recording()
        elif msg.data == 'resume':
            self.start_recording()

    def get_output_filename(self):
        output_fn = os.path.join(self.output_dir,f"{self.db_file_num}_data")
        self.db_file_num += 1
        return output_fn

    def start_recording(self):
        if self.rosbag_process is None:
            db_fn = self.get_output_filename()
            self.get_logger().info(f'Starting rosbag recording in {self.output_dir}...')
            try:
                # Start the ros2 bag record process
                self.rosbag_process = subprocess.Popen(
                    ['ros2', 'bag', 'record', '-o', db_fn, *self.topics, '--split', '--max-bag-size', self.split_size],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except Exception as e:
                self.get_logger().error(f"Failed to start rosbag recording: {e}")

    def stop_recording(self):
        if self.rosbag_process is not None:
            self.get_logger().info('Stopping rosbag recording...')
            self.rosbag_process.terminate()  # Send termination signal
            self.rosbag_process.wait()  # Wait for the process to exit
            self.rosbag_process = None

def main(args=None):
    rclpy.init(args=args)
    node = RosbagManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()