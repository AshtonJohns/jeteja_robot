import config.master_config as master_config
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
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

PWM_DATA_TYPE = master_config.PWM_DATA_TYPE
MOTOR_MAX_DUTY_CYCLE = master_config.MOTOR_MAX_DUTY_CYCLE
MOTOR_MIN_DUTY_CYCLE = master_config.MOTOR_MIN_DUTY_CYCLE
STEERING_MAX_DUTY_CYCLE = master_config.STEERING_MAX_DUTY_CYCLE
STEERING_MIN_DUTY_CYCLE = master_config.STEERING_MIN_DUTY_CYCLE


bridge = CvBridge()


def denormalize_pwm(outputs):
    """"""
    motor_pwm = outputs[0][0][0] * (MOTOR_MAX_DUTY_CYCLE - MOTOR_MIN_DUTY_CYCLE) + MOTOR_MIN_DUTY_CYCLE
    steering_pwm = outputs[1][0][0] * (STEERING_MAX_DUTY_CYCLE - STEERING_MIN_DUTY_CYCLE) + STEERING_MIN_DUTY_CYCLE

    motor_pwm = PWM_DATA_TYPE(motor_pwm)
    steering_pwm = PWM_DATA_TYPE(steering_pwm)
    return motor_pwm, steering_pwm


def normalize_image(image, **kwargs):
    """"""
    color_img = kwargs.get('color',False)
    depth_img = kwargs.get('depth',False)
    if color_img:
        image =  (image / COLOR_NORMALIZATION_FACTOR).astype(COLOR_PREPROCESS_DATA_TYPE)
    elif depth_img:
        image = (image / DEPTH_NORMALIZATION_FACTOR).astype(DEPTH_PREPROCESS_DATA_TYPE)
    return image


def deserialize_ros_message(msg, topic_type):
    """"""
    return deserialize_message(msg, get_message(topic_type))


def ros_to_cv(msg, topic_type='sensor_msgs/msg/Image', **kwargs):
    """"""
    color = kwargs.get('color',False)
    depth = kwargs.get('depth',False)
    if color:
        encoding = COLOR_ENCODING
    elif depth:
        encoding = DEPTH_ENCODING

    image_msg = deserialize_ros_message(msg, topic_type)
                
    cv_image = bridge.imgmsg_to_cv2(image_msg,
                                    desired_encoding=encoding)
    
    return cv_image

def deserialized_ros_to_cv(msg, **kwargs):
    """
    Converts a ROS Image message to an OpenCV image.
    """
    color = kwargs.get('color', False)
    depth = kwargs.get('depth', False)
    
    # Set encoding based on the image type
    if color:
        encoding = COLOR_ENCODING  # Define this appropriately
    elif depth:
        encoding = DEPTH_ENCODING  # Define this appropriately

    # Directly use the msg (already deserialized) with imgmsg_to_cv2
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)
    return cv_image

def get_cvbridge():
    return bridge
