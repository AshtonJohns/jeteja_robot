import yaml
import os
import numpy as np
import tensorflow as tf
from ament_index_python.packages import get_package_share_directory


################# CONFIG files #################
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

remote_control_handler_config = os.path.join(
    get_package_share_directory('jeteja_launch'),
    'config',
    'remote_control_handler.yaml'
)


################# MOTOR CONFIG ################# # TODO

with open(remote_control_handler_config, 'r') as file:
    lower_control_config = yaml.safe_load(file)

MOTOR_NEUTRAL_DUTY_CYCLE = lower_control_config["motor_neutral_duty_cycle"]
MOTOR_MAX_DUTY_CYCLE = lower_control_config["motor_max_duty_cycle"]
MOTOR_MIN_DUTY_CYCLE = lower_control_config["motor_min_duty_cycle"]

STEERING_NEUTRAL_DUTY_CYCLE = lower_control_config["steering_neutral_duty_cycle"]
STEERING_MAX_DUTY_CYCLE = lower_control_config["steering_max_duty_cycle"]
STEERING_MIN_DUTY_CYCLE = lower_control_config["steering_min_duty_cycle"]

MOTOR_PWM_NORMALIZATION_FACTOR = MOTOR_MAX_DUTY_CYCLE - MOTOR_MIN_DUTY_CYCLE
STEERING_PWM_NORMALIZATION_FACTOR = STEERING_MAX_DUTY_CYCLE - STEERING_MIN_DUTY_CYCLE


################# REALSENSE #################
with open(realsense2_camera_config, 'r') as file:
    config = yaml.safe_load(file)

# Color camera settings
COLOR_HEIGHT = int(config['rgb_camera.color_profile'].split("x")[0]) # TODO should be int!
COLOR_WIDTH = int(config['rgb_camera.color_profile'].split("x")[1])
COLOR_FORMAT = config['rgb_camera.color_format']

DEPTH_HEIGHT = int(config['depth_module.depth_profile'].split("x")[0])
DEPTH_WIDTH = int(config['depth_module.depth_profile'].split("x")[1])
DEPTH_FORMAT = config['depth_module.depth_format']


################# AUTOPILOT #################
with open(autopilot_config, 'r') as file:
    config = yaml.safe_load(file)

# Extract parameters from the YAML configuration
MODEL_PATH = config.get('MODEL_PATH')

COLOR_NORMALIZATION_FACTOR = config.get('COLOR_NORMALIZATION_FACTOR')
COLOR_DATA_TYPE = config.get('COLOR_DATA_TYPE')
if COLOR_DATA_TYPE == 'uint8':
    COLOR_DATA_TYPE = np.uint8
COLOR_PREPROCESS_DATA_TYPE = config.get('COLOR_PREPROCESS_DATA_TYPE')
if COLOR_PREPROCESS_DATA_TYPE == 'float32':
    COLOR_PREPROCESS_DATA_TYPE = np.float32
COLOR_ENCODING = config.get('COLOR_ENCODING')
COLOR_INPUT_IDX = config.get('COLOR_INPUT_IDX')
COLOR_CHANNELS = config['COLOR_CHANNELS']

DEPTH_NORMALIZATION_FACTOR = config.get('DEPTH_NORMALIZATION_FACTOR')
DEPTH_DATA_TYPE = config.get('DEPTH_DATA_TYPE')
if DEPTH_DATA_TYPE == 'uint16':
    DEPTH_DATA_TYPE = np.uint16
DEPTH_PREPROCESS_DATA_TYPE = config.get('DEPTH_PREPROCESS_DATA_TYPE')
if DEPTH_PREPROCESS_DATA_TYPE == 'float32':
    DEPTH_PREPROCESS_DATA_TYPE = np.float32
DEPTH_ENCODING = config.get('DEPTH_ENCODING')
DEPTH_INPUT_IDX = config.get('DEPTH_INPUT_IDX')
DEPTH_CHANNELS = config['DEPTH_CHANNELS']

PWM_DATA_TYPE = config.get('PWM_DATA_TYPE')
if PWM_DATA_TYPE == 'int':
    PWM_DATA_TYPE = int
PWM_PREPROCESS_DATA_TYPE = config.get('PWM_PREPROCESS_DATA_TYPE')
if PWM_PREPROCESS_DATA_TYPE == 'float32':
    PWM_PREPROCESS_DATA_TYPE = tf.float32
PWM_OUTPUT_IDX = config.get('PWM_OUTPUT_IDX')
PWM_OUTPUT_SHAPE = tuple(config.get('PWM_OUTPUT_SHAPE'))  # Convert list to tuple
PWM_OUTPUT_DATA_TYPE = config.get('PWM_OUTPUT_DATA_TYPE')
# if PWM_OUTPUT_DATA_TYPE == 'float16':
#     OUTPUT_DATA_TYPE = tf.float16
OUTPUT_ORIGINAL_DATA_TYPE = config.get('OUTPUT_ORIGINAL_DATA_TYPE')

BATCH_SIZE = config.get('BATCH_SIZE')
