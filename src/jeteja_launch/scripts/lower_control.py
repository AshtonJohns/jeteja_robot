import config.master_config as master_config

MOTOR_NEUTRAL_DUTY_CYCLE = master_config.MOTOR_NEUTRAL_DUTY_CYCLE
MOTOR_MAX_DUTY_CYCLE = master_config.MOTOR_MAX_DUTY_CYCLE
MOTOR_MIN_DUTY_CYCLE = master_config.MOTOR_MIN_DUTY_CYCLE
STEERING_NEUTRAL_DUTY_CYCLE = master_config.STEERING_NEUTRAL_DUTY_CYCLE
STEERING_MAX_DUTY_CYCLE = master_config.STEERING_MAX_DUTY_CYCLE
STEERING_MIN_DUTY_CYCLE = master_config.STEERING_MIN_DUTY_CYCLE
PWM_DATA_TYPE = master_config.PWM_DATA_TYPE

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
    # if speed > ADJUSTED_MOTOR_MAX_DUTY_CYCLE:
    #     print(f"Exceeded motor_pwm: {speed}")
    #     speed = ADJUSTED_MOTOR_MAX_DUTY_CYCLE
    # elif speed < ADJUSTED_MOTOR_MIN_DUTY_CYCLE:
    #     print(f"Low motor_pwm: {speed}")
    #     speed = ADJUSTED_MOTOR_MIN_DUTY_CYCLE
        
    # if steering > ADJUSTED_STEERING_MAX_DUTY_CYCLE:
    #     print(f"Exceeded steering_pwm: {speed}")
    #     steering = ADJUSTED_STEERING_MAX_DUTY_CYCLE
    # elif steering < ADJUSTED_STEERING_MIN_DUTY_CYCLE:
    #     print(f"Low steering_pwm: {speed}")
    #     steering = ADJUSTED_STEERING_MIN_DUTY_CYCLE
    
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

def calculate_adjusted_pwm_range(scale_linear, scale_angular):
    # Motor PWM range
    motor_max_pwm = MOTOR_NEUTRAL_DUTY_CYCLE + (scale_linear * (MOTOR_MAX_DUTY_CYCLE - MOTOR_NEUTRAL_DUTY_CYCLE))
    motor_min_pwm = MOTOR_NEUTRAL_DUTY_CYCLE - (scale_linear * (MOTOR_NEUTRAL_DUTY_CYCLE - MOTOR_MIN_DUTY_CYCLE))

    # Steering PWM range
    steering_max_pwm = STEERING_NEUTRAL_DUTY_CYCLE + (scale_angular * (STEERING_MAX_DUTY_CYCLE - STEERING_NEUTRAL_DUTY_CYCLE))
    steering_min_pwm = STEERING_NEUTRAL_DUTY_CYCLE - (scale_angular * (STEERING_NEUTRAL_DUTY_CYCLE - STEERING_MIN_DUTY_CYCLE))

    return {
        "motor_max_pwm": PWM_DATA_TYPE(motor_max_pwm),
        "motor_min_pwm": PWM_DATA_TYPE(motor_min_pwm),
        "steering_max_pwm": PWM_DATA_TYPE(steering_max_pwm),
        "steering_min_pwm": PWM_DATA_TYPE(steering_min_pwm)
    }

SCALE_LINEAR = master_config.SCALE_LINEAR
SCALE_ANGULAR = master_config.SCALE_ANGULAR
ADJUSTED_PWMS = calculate_adjusted_pwm_range(SCALE_LINEAR, SCALE_ANGULAR)
ADJUSTED_MOTOR_MAX_DUTY_CYCLE = ADJUSTED_PWMS['motor_max_pwm']
master_config.ADJUSTED_MOTOR_MAX_DUTY_CYCLE = ADJUSTED_MOTOR_MAX_DUTY_CYCLE
ADJUSTED_MOTOR_MIN_DUTY_CYCLE = ADJUSTED_PWMS['motor_min_pwm']
master_config.ADJUSTED_MOTOR_MIN_DUTY_CYCLE = ADJUSTED_MOTOR_MIN_DUTY_CYCLE
ADJUSTED_STEERING_MAX_DUTY_CYCLE = ADJUSTED_PWMS['steering_max_pwm']
master_config.ADJUSTED_STEERING_MAX_DUTY_CYCLE = ADJUSTED_STEERING_MAX_DUTY_CYCLE
ADJUSTED_STEERING_MIN_DUTY_CYCLE = ADJUSTED_PWMS['steering_min_pwm']
master_config.ADJUSTED_STEERING_MIN_DUTY_CYCLE = ADJUSTED_STEERING_MIN_DUTY_CYCLE
MOTOR_PWM_NORMALIZATION_FACTOR = ADJUSTED_MOTOR_MAX_DUTY_CYCLE - ADJUSTED_MOTOR_MIN_DUTY_CYCLE
master_config.MOTOR_PWM_NORMALIZATION_FACTOR = MOTOR_PWM_NORMALIZATION_FACTOR
STEERING_PWM_NORMALIZATION_FACTOR = ADJUSTED_STEERING_MAX_DUTY_CYCLE - ADJUSTED_STEERING_MIN_DUTY_CYCLE
master_config.STEERING_PWM_NORMALIZATION_FACTOR = STEERING_PWM_NORMALIZATION_FACTOR

def main():
    print(calculate_adjusted_pwm_range(0.17, 0.6))
    print(master_config.ADJUSTED_MOTOR_MAX_DUTY_CYCLE)
    print(master_config.ADJUSTED_STEERING_MAX_DUTY_CYCLE)

if __name__ == '__main__':
    main()