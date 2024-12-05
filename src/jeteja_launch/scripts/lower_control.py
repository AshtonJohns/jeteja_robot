import config.master_config as master_config

MOTOR_NEUTRAL_DUTY_CYCLE = master_config.MOTOR_NEUTRAL_DUTY_CYCLE
MOTOR_MAX_DUTY_CYCLE = master_config.MOTOR_MAX_DUTY_CYCLE
MOTOR_MIN_DUTY_CYCLE = master_config.MOTOR_MIN_DUTY_CYCLE
STEERING_NEUTRAL_DUTY_CYCLE = master_config.STEERING_NEUTRAL_DUTY_CYCLE
STEERING_MAX_DUTY_CYCLE = master_config.STEERING_MAX_DUTY_CYCLE
STEERING_MIN_DUTY_CYCLE = master_config.STEERING_MIN_DUTY_CYCLE

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
    
def perform_safe_speed_check(speed, steering):
    if speed > MOTOR_MAX_DUTY_CYCLE:
        print(f"Exceeded motor_pwm: {speed}")
        speed = MOTOR_MAX_DUTY_CYCLE
    elif speed < MOTOR_MIN_DUTY_CYCLE:
        print(f"Low motor_pwm: {speed}")
        speed = MOTOR_MIN_DUTY_CYCLE
        
    if steering > STEERING_MAX_DUTY_CYCLE:
        print(f"Exceeded steering_pwm: {speed}")
        steering = STEERING_MAX_DUTY_CYCLE
    elif steering < STEERING_MIN_DUTY_CYCLE:
        print(f"Low steering_pwm: {speed}")
        steering = STEERING_MIN_DUTY_CYCLE
    
    return speed, steering

def create_command_message(speed_duty_cycle, steering_duty_cycle):
    speed_duty_cycle, steering_duty_cycle = perform_safe_speed_check(speed_duty_cycle,
                                                                     steering_duty_cycle)
    command = f"SPEED:{speed_duty_cycle};STEER:{steering_duty_cycle}\n"
    return command

def create_neutral_command_message():
    speed, steer = get_neutral_pwm()
    command = create_command_message(speed,steer)
    return command

def get_neutral_pwm():
    return master_config.MOTOR_NEUTRAL_DUTY_CYCLE, master_config.STEERING_NEUTRAL_DUTY_CYCLE