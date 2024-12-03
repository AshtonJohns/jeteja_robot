import yaml
import os
from ament_index_python.packages import get_package_share_directory

remote_control_handler_config = os.path.join(
    get_package_share_directory('jeteja_launch'),
    'config',
    'remote_control_handler.yaml'
    )

with open(remote_control_handler_config, 'r') as file:
    lower_control_config = yaml.safe_load(file)

MOTOR_NEUTRAL_DUTY_CYCLE = lower_control_config["motor_neutral_duty_cycle"]
MOTOR_MAX_DUTY_CYCLE = lower_control_config["motor_max_duty_cycle"]
MOTOR_MIN_DUTY_CYCLE = lower_control_config["motor_min_duty_cycle"]

STEERING_NEUTRAL_DUTY_CYCLE = lower_control_config["steering_neutral_duty_cycle"]
STEERING_MAX_DUTY_CYCLE = lower_control_config["steering_max_duty_cycle"]
STEERING_MIN_DUTY_CYCLE = lower_control_config["steering_min_duty_cycle"]

def denormalize_pwm(outputs):
    motor_pwm = outputs[0, 0] * (MOTOR_MAX_DUTY_CYCLE - MOTOR_MIN_DUTY_CYCLE) + MOTOR_MIN_DUTY_CYCLE
    steering_pwm = outputs[0, 1] * (STEERING_MAX_DUTY_CYCLE - STEERING_MIN_DUTY_CYCLE) + STEERING_MIN_DUTY_CYCLE
    return motor_pwm, steering_pwm

