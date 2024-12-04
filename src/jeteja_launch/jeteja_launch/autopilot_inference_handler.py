import traceback
import rclpy
import scripts.postprocessing as postprocessing
from rclpy.node import Node
from jeteja_launch_msgs.msg import PwmSignals, PreprocessedImage
from scripts.preprocessing import image_msg_to_numpy
from scripts.model_inference_handler import TensorRTInference

class AutopilotInferenceHandler(Node):
    def __init__(self):
        super().__init__('autopilot_inference_handler')

        self.image_sub = self.create_subscription(PreprocessedImage, '/autopilot/preprocessed_images', self.image_callback, 10)

        self.pwm_pub = self.create_publisher(PwmSignals, '/pwm_signals', 10)

        self.trt_infer = TensorRTInference()

        self.color_image = None
        self.depth_image = None

    def image_callback(self, msg):
        color_image = msg.color_image
        depth_image = msg.depth_image
        # self.get_logger().info(color_image)
        # self.get_logger().info(depth_image)
        # self.get_logger().info(color_image.encoding)
        # self.get_logger().info(depth_image.encoding)
        self.color_image = image_msg_to_numpy(color_image)
        self.depth_image = image_msg_to_numpy(depth_image)
        self.run_inference()

    def run_inference(self):
        if self.color_image is not None and self.depth_image is not None:
            # self.get_logger().info(self.color_image)
            # self.get_logger().info(self.depth_image)

            outputs = self.trt_infer.infer(self.color_image, self.depth_image)

            self.get_logger().info(str(outputs[0][0][0]))
            self.get_logger().info(str(outputs[1][0][0]))

            # motor_pwm, steering_pwm = postprocessing.denormalize_pwm(outputs)


            # # Publish PWM values
            # pwm_msg = PwmSignals()
            # pwm_msg.stamp = self.get_clock().now().to_msg()
            # pwm_msg.motor_pwm = motor_pwm
            # pwm_msg.steering_pwm = steering_pwm

            # self.pwm_pub.publish(pwm_msg)


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
