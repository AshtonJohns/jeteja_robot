import os
import yaml
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

def calculate_motor_duty_cycle(value):
    """cmd velocity to pwm motor duty"""
    if value > 0:
        return int(MOTOR_NEUTRAL_DUTY_CYCLE + (value * (MOTOR_MAX_DUTY_CYCLE - MOTOR_NEUTRAL_DUTY_CYCLE)))
    elif value < 0:
        return int(MOTOR_NEUTRAL_DUTY_CYCLE + (value * (MOTOR_NEUTRAL_DUTY_CYCLE - MOTOR_MIN_DUTY_CYCLE)))
    else:
        return MOTOR_NEUTRAL_DUTY_CYCLE

def calculate_steering_duty_cycle(value):
    """cmd velocity to pwm steering duty"""
    if value > 0:
        return int(STEERING_NEUTRAL_DUTY_CYCLE + (value * (STEERING_MAX_DUTY_CYCLE - STEERING_NEUTRAL_DUTY_CYCLE)))
    elif value < 0:
        return int(STEERING_NEUTRAL_DUTY_CYCLE + (value * (STEERING_NEUTRAL_DUTY_CYCLE - STEERING_MIN_DUTY_CYCLE)))
    else:
        return STEERING_NEUTRAL_DUTY_CYCLE

def create_command_message(speed_duty_cycle, steering_duty_cycle):
    command = f"SPEED:{speed_duty_cycle};STEER:{steering_duty_cycle}\n"
    return command