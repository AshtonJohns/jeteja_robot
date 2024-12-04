import numpy as np
import config.master_config as master_config
from cv_bridge import CvBridge

COLOR_NORMALIZATION_FACTOR = master_config.COLOR_NORMALIZATION_FACTOR
COLOR_DATA_TYPE = master_config.COLOR_DATA_TYPE
COLOR_ENCODING = master_config.COLOR_ENCODING
COLOR_CHANNELS = master_config.COLOR_CHANNELS
COLOR_PREPROCESS_DATA_TYPE = master_config.COLOR_PREPROCESS_DATA_TYPE
DEPTH_NORMALIZATION_FACTOR = master_config.DEPTH_NORMALIZATION_FACTOR
DEPTH_DATA_TYPE = master_config.DEPTH_DATA_TYPE
DEPTH_ENCODING = master_config.DEPTH_ENCODING
DEPTH_CHANNELS = master_config.DEPTH_CHANNELS
DEPTH_PREPROCESS_DATA_TYPE = master_config.DEPTH_PREPROCESS_DATA_TYPE

class ImageToRosMsg(object):
    def __init__(self) -> None:
        
        # Initialize the CvBridge
        self.bridge = CvBridge()
    
    def preprocess(self, image, **kwargs):
        """"""
        color_img = kwargs.get('color',False)
        depth_img = kwargs.get('depth',False)
        if color_img:
            return image.astype(COLOR_PREPROCESS_DATA_TYPE) / COLOR_NORMALIZATION_FACTOR
        elif depth_img:
            return image.astype(DEPTH_PREPROCESS_DATA_TYPE) / DEPTH_NORMALIZATION_FACTOR

    def bridge_imgmsg_to_cv2(self, msg, desired_encoding):
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding)
    
    def convert_image_array_to_ros_image_msg(self, image, **kwargs):
        """"""
        color_img = kwargs.get('color',False)
        depth_img = kwargs.get('depth', False)
        if depth_img:
            image = self.bridge.cv2_to_imgmsg(
                    (image * DEPTH_NORMALIZATION_FACTOR).astype(DEPTH_DATA_TYPE), 
                    encoding=DEPTH_ENCODING)
        elif color_img:
            image = self.bridge.cv2_to_imgmsg(
                (image * COLOR_NORMALIZATION_FACTOR).astype(COLOR_DATA_TYPE), 
                encoding=COLOR_ENCODING)
        
        print(f"TO ROS: After Conversion - Min: {image.min()}, Max: {image.max()}, Shape: {image.shape}")
        return image

def image_msg_to_numpy(image_msg): # TODO use config parameters
    """
    Converts a sensor_msgs/Image to a NumPy array.

    Args:
        image_msg (sensor_msgs.msg.Image): The ROS image message.

    Returns:
        numpy.ndarray: The image as a NumPy array.
    """
    image_encoding = image_msg.encoding
    print(f"Image Encoding: {image_encoding}")
    print(f"Image Data Size: {len(image_msg.data)}")
    print(f"Image Height: {image_msg.height}, Image Width: {image_msg.width}")

    if image_encoding in COLOR_ENCODING:
        dtype = COLOR_DATA_TYPE
        channels = COLOR_CHANNELS
    elif image_encoding in DEPTH_ENCODING:
        dtype = DEPTH_DATA_TYPE
        channels = DEPTH_CHANNELS

    # Convert buffer to NumPy array
    image = np.frombuffer(image_msg.data, dtype=dtype)
    print(f"Image NumPy Array Size: {image.size}")

    # Validate array size before reshaping
    expected_size = image_msg.height * image_msg.width * channels
    if image.size != expected_size:
        raise ValueError(f"Cannot reshape array of size {image.size} into shape "
                         f"({image_msg.height}, {image_msg.width}, {channels})")

    # Reshape into correct dimensions
    return image.reshape((image_msg.height, image_msg.width, channels))


