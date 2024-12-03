import os
import yaml
import numpy as np
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge

realsense2_camera_config = os.path.join(
    get_package_share_directory('jeteja_launch'),
    'config',
    'realsense2_camera.yaml'
)

autopilot_config = os.path.join(
    get_package_share_directory('jeteja_launch'),
    'config',
    'autopilot.yaml'
)

# Parse the realsense camera YAML file
with open(realsense2_camera_config, 'r') as file:
    config = yaml.safe_load(file)

# Color camera settings
COLOR_HEIGHT = config['rgb_camera.color_profile'].split("x")[0]
COLOR_WIDTH = config['rgb_camera.color_profile'].split("x")[1]
COLOR_FORMAT = config['rgb_camera.color_format']

DEPTH_HEIGHT = config['depth_module.depth_profile'].split("x")[0]
DEPTH_WIDTH = config['depth_module.depth_profile'].split("x")[1]
COLOR_FORMAT = config['depth_module.depth_format']

# Parse the autopilot YAML file
with open(autopilot_config, 'r') as file:
    config = yaml.safe_load(file)

# Extract parameters from the YAML configuration
COLOR_NORMALIZATION_FACTOR = config.get('COLOR_NORMALIZATION_FACTOR')
COLOR_DATA_TYPE = config.get('COLOR_DATA_TYPE')
COLOR_ENCODING = config.get('COLOR_ENCODING')
COLOR_INPUT_IDX = config.get('COLOR_INPUT_IDX')

DEPTH_NORMALIZATION_FACTOR = config.get('DEPTH_NORMALIZATION_FACTOR')
DEPTH_DATA_TYPE = config.get('DEPTH_DATA_TYPE')
DEPTH_ENCODING = config.get('DEPTH_ENCODING')
DEPTH_INPUT_IDX = config.get('DEPTH_INPUT_IDX')

BATCH_SIZE = config.get('BATCH_SIZE')
OUTPUT_IDX = config.get('OUTPUT_IDX')
COLOR_CHANNELS = config['COLOR_CHANNELS']
DEPTH_CHANNELS = config['DEPTH_CHANNELS']
OUTPUT_SHAPE = config['OUTPUT_SHAPE']


class ImageToRosMsg(object):
    def __init__(self) -> None:
        
        # Initialize the CvBridge
        self.bridge = CvBridge()
    
    def preprocess(self, image, **kwargs):
        """"""
        color_img = kwargs.get('color',False)
        depth_img = kwargs.get('depth',False)
        if color_img:
            return image.astype(np.float32) / COLOR_NORMALIZATION_FACTOR
        elif depth_img:
            return image.astype(np.float32) / DEPTH_NORMALIZATION_FACTOR

    def bridge_imgmsg_to_cv2(self, msg, desired_encoding):
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding)
    
    def convert_image_array_to_ros_image_msg(self, image, **kwargs):
        """"""
        color_img = kwargs.get('color',False)
        depth_img = kwargs.get('depth', False)
        encoding = kwargs.get('encoding', False)
        if depth_img:
            return self.bridge.cv2_to_imgmsg(
                    (image * DEPTH_NORMALIZATION_FACTOR).astype(DEPTH_DATA_TYPE), 
                    encoding=encoding)
        elif color_img:
            return self.bridge.cv2_to_imgmsg(
                (image * COLOR_NORMALIZATION_FACTOR).astype(COLOR_DATA_TYPE), 
                encoding=encoding)

def image_msg_to_numpy(image_msg):
    """
    Converts a sensor_msgs/Image to a NumPy array.

    Args:
        image_msg (sensor_msgs.msg.Image): The ROS image message.

    Returns:
        numpy.ndarray: The image as a NumPy array.
    """
    image_encoding = image_msg.encoding
    if image_encoding in COLOR_ENCODING:
        dtype = np.uint16
    elif image_encoding in DEPTH_ENCODING:
        dtype = np.int16
    image = np.frombuffer(image_msg.data, dtype=dtype)
    return image.reshape((image_msg.height, image_msg.width, -1))

