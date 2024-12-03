import os
import yaml
import numpy as np
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge

class ImageToRosMsg(object):
    def __init__(self) -> None:
        # Load the YAML configuration file
        autopilot_config = os.path.join(
            get_package_share_directory('jeteja_launch'),
            'config',
            'autopilot.yaml'
        )
        
        # Parse the YAML file
        with open(autopilot_config, 'r') as file:
            config = yaml.safe_load(file)
        
        # Extract parameters from the YAML configuration
        ros_parameters = config.get('image_to_processed_image', {}).get('ros__parameters', {})
        self.COLOR_NORMALIZATION_FACTOR = ros_parameters.get('COLOR_NORMALIZATION_FACTOR', 255.0)
        self.DEPTH_NORMALIZATION_FACTOR = ros_parameters.get('DEPTH_NORMALIZATION_FACTOR', 65535.0)
        self.COLOR_DATA_TYPE = ros_parameters.get('COLOR_DATA_TYPE', 'uint8')
        self.DEPTH_DATA_TYPE = ros_parameters.get('DEPTH_DATA_TYPE', 'uint16')
        self.COLOR_ENCODING = ros_parameters.get('COLOR_ENCODING', 'rgb8')
        self.DEPTH_ENCODING = ros_parameters.get('DEPTH_ENCODING', 'mono16')

        # Map the data types
        if self.COLOR_DATA_TYPE == 'uint8':
            self.COLOR_DATA_TYPE = np.uint8
        if self.DEPTH_DATA_TYPE == 'uint16':
            self.DEPTH_DATA_TYPE = np.uint16
        
        # Initialize the CvBridge
        self.bridge = CvBridge()
    
    def preprocess(self, image, **kwargs):
        """"""
        color_img = kwargs.get('color',False)
        depth_img = kwargs.get('depth',False)
        if color_img:
            return image.astype(np.float32) / self.COLOR_NORMALIZATION_FACTOR
        elif depth_img:
            return image.astype(np.float32) / self.DEPTH_NORMALIZATION_FACTOR

    def bridge_imgmsg_to_cv2(self, msg, desired_encoding):
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding)
    
    def convert_image_array_to_ros_image_msg(self, image, **kwargs):
        """"""
        color_img = kwargs.get('color',False)
        depth_img = kwargs.get('depth', False)
        encoding = kwargs.get('encoding', False)
        if not encoding: # TODO
            encoding = 'passthrough'
        if depth_img:
            return self.bridge.cv2_to_imgmsg(
                    (image * self.DEPTH_NORMALIZATION_FACTOR).astype(self.DEPTH_DATA_TYPE), 
                    encoding=encoding)
        elif color_img:
            return self.bridge.cv2_to_imgmsg(
                (image * self.COLOR_NORMALIZATION_FACTOR).astype(self.COLOR_DATA_TYPE), 
                encoding=encoding)

