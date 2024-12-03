import traceback
import rclpy
from scripts.preprocessing import ImageToRosMsg
from rclpy.node import Node
from sensor_msgs.msg import Image
from jeteja_launch_msgs.msg import PreprocessedImage 

class ImageToProcessedImage(Node):
    def __init__(self):
        super().__init__('image_to_processed_image')

        self.image_exec = ImageToRosMsg()

        # Subscriptions to raw camera topics
        self.color_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)

        # Publisher for combined preprocessed images
        self.preprocessed_pub = self.create_publisher(
            PreprocessedImage, '/autopilot/preprocessed_images', 10)

        # Internal storage for synchronized processing
        self.color_image = None
        self.depth_image = None

    def color_callback(self, msg):
        # Convert ROS Image message to NumPy array
        self.color_image = self.image_exec.bridge_imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.process_and_publish()

    def depth_callback(self, msg):
        # Convert ROS Image message to NumPy array
        self.depth_image = self.image_exec.bridge_imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.process_and_publish()

    def process_and_publish(self):
        if self.color_image is not None and self.depth_image is not None:
            # Preprocess color and depth images
            color_image = self.image_exec.preprocess(self.color_image,color=True)
            depth_image = self.image_exec.preprocess(self.depth_image,depth=True)

            # Convert back to ROS Image messages
            color_msg = self.image_exec.convert_image_array_to_ros_image_msg(color_image,color=True)
            depth_msg = self.image_exec.convert_image_array_to_ros_image_msg(depth_image,depth=True)

            # Create and publish PreprocessedImage message
            preprocessed_msg = PreprocessedImage()
            preprocessed_msg.color_image = color_msg
            preprocessed_msg.depth_image = depth_msg
            self.preprocessed_pub.publish(preprocessed_msg)

            self.color_image = None
            self.depth_image = None


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ImageToProcessedImage()
        rclpy.spin(node)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()