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
from scripts.pico_handler import PicoConnection
from ament_index_python.packages import get_package_share_directory
from jeteja_launch_msgs.msg import PwmSignals

class TensorRTInference:
    def __init__(self, trt_model_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(trt_model_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.input_shape = (1, 360, 640, 3)  # Adjust as needed
        self.output_shape = (1, 2)  # For motor and steering
        self.input_size = np.prod(self.input_shape) * np.float32(1).nbytes
        self.output_size = np.prod(self.output_shape) * np.float32(1).nbytes
        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_output = cuda.mem_alloc(self.output_size)
        self.h_output = np.empty(self.output_shape, dtype=np.float32)

    def infer(self, color_image, depth_image):
        # Concatenate or preprocess images as needed
        input_data = np.concatenate([color_image, depth_image], axis=-1).astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

        # Transfer input to GPU
        cuda.memcpy_htod(self.d_input, input_data)

        # Execute inference
        self.context.execute_v2([int(self.d_input), int(self.d_output)])

        # Transfer output back to CPU
        cuda.memcpy_dtoh(self.h_output, self.d_output)

        return self.h_output



class AutopilotInferenceHandler(Node):
    def __init__(self):
        super().__init__('autopilot_inference_handler')

        self.color_sub = self.create_subscription(Image, '/autopilot/preprocessed_images/color_image', self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/autopilot/preprocessed_images/depth_image', self.depth_callback, 10)

        self.motor_pub = self.create_publisher(Float32, '/autopilot/motor_pwm', 10)
        self.steering_pub = self.create_publisher(Float32, '/autopilot/steering_pwm', 10)

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
            motor_pwm = outputs[0, 0] * (MOTOR_PWM_MAX - MOTOR_PWM_MIN) + MOTOR_PWM_MIN
            steering_pwm = outputs[0, 1] * (STEERING_PWM_MAX - STEERING_PWM_MIN) + STEERING_PWM_MIN

            self.motor_pub.publish(Float32(data=motor_pwm))
            self.steering_pub.publish(Float32(data=steering_pwm))

import numpy as np
from sensor_msgs.msg import Image

def image_msg_to_numpy(image_msg):
    """
    Converts a sensor_msgs/Image to a NumPy array.

    Args:
        image_msg (sensor_msgs.msg.Image): The ROS image message.

    Returns:
        numpy.ndarray: The image as a NumPy array.
    """
    dtype = np.uint8 if image_msg.encoding in ['rgb8', 'bgr8', 'mono8'] else np.float32
    image = np.frombuffer(image_msg.data, dtype=dtype)
    return image.reshape((image_msg.height, image_msg.width, -1))

