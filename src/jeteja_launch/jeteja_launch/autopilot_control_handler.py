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
from jeteja_launch_msgs.msg import PwmSignals

class AutoPilotControlHandler(Node):
    def __init__(self):
        super().__init__('autopilot_control_handler')

        # Instantiate a RemoteControlHandler and PicoHandler
        pico_port_script_path = os.path.join(
            get_package_share_directory('jeteja_launch'),
            'scripts',
            'main.py',
        )
        self.pico_execute = PicoConnection(pico_port_script_path)

        # Create subscribers and publishers
        self.pwm_signals_subscription = self.create_subscription(PwmSignals, '/pwm_signals', self.send_duty_cycle_to_pico, 15)
        self.recording_status_pub = self.create_publisher(String, '/recording_status', 10)

        # State variables
        self.microcontroller_state = False
        self.recording_state = False
        self.enable_recording_state = False
        self.pico_enable_state = False


    def send_duty_cycle_to_pico(self, msg):
        if self.pico_execute.get_state(): # Check if Pico is running and serial connection is established
            motor_pwm = msg.motor_pwm
            steering_pwm = msg.steering_pwm
            command = lower_control.create_command_message(motor_pwm, steering_pwm)
            if self.pico_enable_state: # Has surpassed the lockdown timer and is enabled
                self.pico_execute.write(command)
                # self.get_logger().info(f"Sent: {command}")
            # else:
            #     self.get_logger().info(f"Pico is not alive!")

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
