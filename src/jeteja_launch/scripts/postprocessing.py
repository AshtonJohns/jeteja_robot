import config.master_config as master_config

PWM_DATA_TYPE = master_config.PWM_DATA_TYPE
MOTOR_MAX_DUTY_CYCLE = master_config.MOTOR_MAX_DUTY_CYCLE
MOTOR_MIN_DUTY_CYCLE = master_config.MOTOR_MIN_DUTY_CYCLE
STEERING_MAX_DUTY_CYCLE = master_config.STEERING_MAX_DUTY_CYCLE
STEERING_MIN_DUTY_CYCLE = master_config.STEERING_MIN_DUTY_CYCLE

def denormalize_pwm(outputs):
    motor_pwm = outputs[0][0][0] * (MOTOR_MAX_DUTY_CYCLE - MOTOR_MIN_DUTY_CYCLE) + MOTOR_MIN_DUTY_CYCLE
    steering_pwm = outputs[1][0][0] * (STEERING_MAX_DUTY_CYCLE - STEERING_MIN_DUTY_CYCLE) + STEERING_MIN_DUTY_CYCLE

    motor_pwm = PWM_DATA_TYPE(motor_pwm)
    steering_pwm = PWM_DATA_TYPE(steering_pwm)
    return motor_pwm, steering_pwm

