import os
import numpy as np
import cv2
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import scripts.lower_control as lower_control
from scripts.pico_handler import PicoConnection
from ament_index_python.packages import get_package_share_directory

class AutoPilotControlHandler(Node):
    def __init__(self):
        super().__init__('autopilot_control_handler')

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.setup_realsense()

        # Instantiate a RemoteControlHandler and PicoHandler
        pico_port_script_path = os.path.join(
            get_package_share_directory('jeteja_launch'),
            'scripts',
            'main.py',
        )
        self.pico_execute = PicoConnection(pico_port_script_path)

        # Create subscribers and publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel_fixed_rate', 10)
        self.recording_status_pub = self.create_publisher(String, '/recording_status', 10)

        # State variables
        self.microcontroller_state = False
        self.recording_state = False
        self.enable_recording_state = False

        # Start RealSense processing at 60Hz
        self.timer = self.create_timer(0.016, self.process_frame_callback)

    def setup_realsense(self):
        """Configures the RealSense pipeline for color and depth streams."""
        self.config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 60)
        self.config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 60)
        self.pipeline.start(self.config)
        self.get_logger().info("RealSense pipeline started.")

    def process_frame_callback(self):
        """Callback to capture RealSense frames, process data, and control motors."""
        try:
            # Capture synchronized color and depth frames
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return

            # Preprocess the frames (resize and normalize)
            color_input = self.preprocess_frame(color_frame, (640, 360), is_color=True)
            depth_input = self.preprocess_frame(depth_frame, (640, 360), is_color=False)

            # Infer or process the data here (mocked for now)
            speed_duty_cycle, steering_duty_cycle = self.generate_mock_output(color_input, depth_input)

            # Send to Pico
            self.send_duty_cycle_to_pico(speed_duty_cycle, steering_duty_cycle)

        except Exception as e:
            self.get_logger().error(f"Error in process_frame_callback: {e}")

    def preprocess_frame(self, frame, target_shape, is_color):
        """Preprocess color or depth frame for further processing."""
        frame_data = np.asanyarray(frame.get_data())
        resized_frame = cv2.resize(frame_data, target_shape)

        if is_color:
            # Normalize color frame
            return resized_frame / 255.0
        else:
            # Normalize depth frame
            return resized_frame / np.max(resized_frame)

    def generate_mock_output(self, color_input, depth_input):
        """Placeholder for model inference. Replace with actual TensorRT code."""
        # Mock values for motor_pwm and steering_pwm
        return 0.5, -0.2

    def send_duty_cycle_to_pico(self, speed_duty_cycle, steering_duty_cycle):
        """Send calculated duty cycle to Pico via RemoteControlHandler."""
        if self.microcontroller_state:
            command = lower_control.create_command_message(speed_duty_cycle,steering_duty_cycle)
            self.pico_execute.write(command)
            self.get_logger().info(f"Sent to Pico: {command}")
        # else:
        #     self.get_logger().info("Pico is not alive!")

    def destroy_node(self):
        """Clean up resources on shutdown."""
        self.pipeline.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = AutoPilotControlHandler()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down AutoPilotControlHandler.')
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
