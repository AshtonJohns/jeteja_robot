import traceback
import rclpy
import scripts.image_processing as image_processing
from rclpy.node import Node
from sensor_msgs.msg import Image
from jeteja_launch_msgs.msg import PreprocessedImage, PwmSignals
from scripts.model_inference_handler import TensorRTInference


class AutopilotInferenceHandler(Node):
    def __init__(self):
        super().__init__('autopilot_inference_handler')

        # Subscriptions to raw camera topics
        self.rgb_image_topic = '/camera/camera/color/image_raw'
        self.color_sub = self.create_subscription(Image, self.rgb_image_topic, self.color_callback, 10) # TODO QoS reliability
        self.depth_image_topic = '/camera/camera/depth/image_rect_raw'
        self.depth_sub = self.create_subscription(Image, self.depth_image_topic, self.depth_callback, 10)

        # # Publisher for combined preprocessed images
        # self.preprocessed_pub = self.create_publisher(
        #     PreprocessedImage, '/autopilot/preprocessed_images', 10)
        
        # Model for inference
        self.trt_infer = TensorRTInference()

        # PWM publisher
        self.pwm_pub = self.create_publisher(PwmSignals, '/pwm_signals', 10)

        # Internal storage for synchronized processing
        self.color_image = None
        self.depth_image = None

    def color_callback(self, msg):
        # Convert ROS Image message to NumPy array
        self.color_image = image_processing.deserialized_ros_to_cv(msg, color=True)
        self.process_and_publish()

    def depth_callback(self, msg):
        # Convert ROS Image message to NumPy array
        self.depth_image = image_processing.deserialized_ros_to_cv(msg, depth=True)
        self.process_and_publish()

    def process_and_publish(self):
        if self.color_image is not None and self.depth_image is not None:
            # Preprocess color and depth images
            color_image = image_processing.normalize_image(self.color_image,color=True)
            depth_image = image_processing.normalize_image(self.depth_image,depth=True)

            # Infer images
            outputs = self.trt_infer.infer(color_image, depth_image)
            motor_pwm, steering_pwm = image_processing.denormalize_pwm(outputs)

            # DEBUG
            # self.get_logger().info(f"{outputs}")
            # self.get_logger().info(f"{motor_pwm}")
            # self.get_logger().info(f"{steering_pwm}")

            # Publish PWM values
            pwm_msg = PwmSignals()
            pwm_msg.stamp = self.get_clock().now().to_msg()
            pwm_msg.motor_pwm = motor_pwm
            pwm_msg.steering_pwm = steering_pwm
            self.pwm_pub.publish(pwm_msg)

            self.color_image = None
            self.depth_image = None

            # TODO ROS topic to publish image data
            # # Convert back to ROS Image messages
            # color_msg = self.image_exec.convert_image_array_to_ros_image_msg(color_image,color=True)
            # depth_msg = self.image_exec.convert_image_array_to_ros_image_msg(depth_image,depth=True)

            # # Create and publish PreprocessedImage message
            # preprocessed_msg = PreprocessedImage()
            # preprocessed_msg.color_image = color_msg
            # preprocessed_msg.depth_image = depth_msg
            # self.preprocessed_pub.publish(preprocessed_msg)

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = AutopilotInferenceHandler()
        rclpy.spin(node)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()