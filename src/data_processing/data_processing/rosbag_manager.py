import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from launch.actions import ExecuteProcess

class RosbagManager(Node):
    def __init__(self):
        super().__init__('rosbag_manager')

        # Get parameters
        self.topics = self.declare_parameter('topics', []).get_parameter_value().string_array_value
        self.get_logger().info(f"Topics: {self.topics}")
        self.output_dir = self.declare_parameter('output_dir', 'data/rosbags').get_parameter_value().string_value

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

    def start_recording(self):
        if self.rosbag_process is None:
            self.get_logger().info(f'Starting rosbag recording in {self.output_dir}...')
            self.rosbag_process = ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'record',
                    '-o', self.output_dir,
                    *self.topics  # Pass the topics as individual arguments
                ],
                output='screen'
            )
            self.rosbag_process.execute()

    def stop_recording(self):
        if self.rosbag_process is not None:
            self.get_logger().info('Stopping rosbag recording...')
            self.rosbag_process.process.terminate()
            self.rosbag_process = None

def main(args=None):
    rclpy.init(args=args)
    node = RosbagManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()