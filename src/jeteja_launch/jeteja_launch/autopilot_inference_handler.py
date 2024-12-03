import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import os
import numpy as np
import cv2
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import scripts.lower_control as lower_control
import scripts.postprocessing as postprocessing
from scripts.pico_handler import PicoConnection
from ament_index_python.packages import get_package_share_directory
from jeteja_launch_msgs.msg import PwmSignals
from scripts.preprocessing import image_msg_to_numpy
from scripts.model_inference_handler import TensorRTInference

class AutopilotInferenceHandler(Node):
    def __init__(self):
        super().__init__('autopilot_inference_handler')

        self.color_sub = self.create_subscription(Image, '/autopilot/preprocessed_images/color_image', self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/autopilot/preprocessed_images/depth_image', self.depth_callback, 10)

        self.pwm_pub = self.create_publisher(PwmSignals, '/pwm_signals', 10)

        self.trt_infer = TensorRTInference('/path/to/model.trt')

        self.color_image = None
        self.depth_image = None

    def color_callback(self, msg):
        self.color_image = image_msg_to_numpy(msg)
        self.run_inference()

    def depth_callback(self, msg):
        self.depth_image = image_msg_to_numpy(msg)
        self.run_inference()

    def run_inference(self):
        if self.color_image is not None and self.depth_image is not None:
            outputs = self.trt_infer.infer(self.color_image, self.depth_image)

            # Denormalize outputs
            motor_pwm, steering_pwm = postprocessing.denormalize_pwm(outputs)

            # Publish PWM values
            pwm_msg = PwmSignals()
            pwm_msg.stamp = self.get_clock().now().to_msg()
            pwm_msg.motor_pwm = motor_pwm
            pwm_msg.steering_pwm = steering_pwm

            self.pwm_pub.publish(pwm_msg)


