"""
Integrated test with controller, pico usb communication, throttle motor.
"""
import sys
import os
import serial
import pygame
import json
from time import sleep


# SETUP
# Load configs
params_file_path = os.path.join(os.path.dirname(sys.path[0]), 'test_config.json')
params_file = open(params_file_path)
params = json.load(params_file)
# Constants
STEERING_AXIS = params['steering_joy_axis']
STEERING_CENTER = params['steering_center']
STEERING_RANGE = params['steering_range']
THROTTLE_AXIS = params['throttle_joy_axis']
THROTTLE_STALL = params['throttle_stall']
THROTTLE_FWD_RANGE = params['throttle_fwd_range']
THROTTLE_REV_RANGE = params['throttle_rev_range']
THROTTLE_LIMIT = params['throttle_limit']
STOP_BUTTON = params['stop_btn'] # 'X' button emergency stop
# Init serial port
ser_pico = serial.Serial(port='/dev/ttyACM0', baudrate=115200)
print(f"Pico is connected to port: {ser_pico.name}")
# Init controller
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)
# Init joystick axes values
ax_val_st = 0.
ax_val_th = 0.

# MAIN LOOP
try:
    while True:
        for e in pygame.event.get():  # read controller input
            if e.type == pygame.JOYAXISMOTION:
                ax_val_st = round((js.get_axis(STEERING_AXIS)), 2)  # keep 2 decimals
                ax_val_th = round((js.get_axis(THROTTLE_AXIS)), 2)  # keep 2 decimals
            elif e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(STOP_BUTTON):  # emergency stop 
                    print("E-STOP PRESSED. TERMINATE")
                    pygame.quit()
                    ser_pico.close()
                    sys.exit()
        # Calaculate steering and throttle value
        act_st = -ax_val_st
        act_th = -ax_val_th  # throttle action: -1: max forward, 1: max backward
        # Encode steering value to dutycycle in nanosecond
        duty_st = STEERING_CENTER - STEERING_RANGE + int(STEERING_RANGE * (act_st + 1))
        # Encode throttle value to dutycycle in nanosecond
        if act_th > 0:
            duty_th = THROTTLE_STALL + int(THROTTLE_FWD_RANGE * min(act_th, THROTTLE_LIMIT))
        elif act_th < 0:
            duty_th = THROTTLE_STALL + int(THROTTLE_REV_RANGE * max(act_th, -THROTTLE_LIMIT))
        else:
            duty_th = THROTTLE_STALL 
        print(f"Duty Throttle: {duty_th}")
        print(f"Duty Steering: {duty_st}")
        msg = (str(duty_st) + "," + str(duty_th) + "\n").encode('utf-8')
        # f"{act_st,act_th}\n".encode('utf-8')
        ser_pico.write(msg)
        # Log action
        print(f"action: {act_st, act_th}")
        # 20Hz
        sleep(0.05)

# Take care terminal signal (Ctrl-c)
except KeyboardInterrupt:
    pygame.quit()
    ser_pico.close()
    sys.exit()
