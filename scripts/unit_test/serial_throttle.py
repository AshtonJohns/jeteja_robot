"""
Transmit control signal (a float number range from -1 to 1) to Pico
Pico will decode this signal to PWM dutycycle to regulate ESC
"""
import sys
import os
import serial
from time import sleep
import json

# SETUP
# Load configs
params_file_path = os.path.join(os.path.dirname(sys.path[0]), 'configs.json')
params_file = open(params_file_path)
params = json.load(params_file)
STEERING_CENTER = params['steering_center']
THROTTLE_STALL = params['throttle_stall']
THROTTLE_FWD_RANGE = params['throttle_fwd_range']
THROTTLE_REV_RANGE = params['throttle_rev_range']
THROTTLE_LIMIT = params['throttle_limit']
# Init serial port
ser = serial.Serial(port='/dev/ttyACM0', baudrate=115200)
print(f"Pico is connected to port: {ser.name}")
# Init drivetrain
duty_st = STEERING_CENTER
duty_th = THROTTLE_STALL


# LOOP
# Throttle: stall->fwd->stall->rev->stall
for i in range(100): # forward ramp up
    duty_th = THROTTLE_STALL + int(THROTTLE_FWD_RANGE * i / 100)
    msg = (str(duty_st) + "," + str(duty_th) + "\n").encode('utf-8')
    # msg = f"{i/100}\n".encode('utf-8')
    ser.write(msg)
    sleep(0.1)
    if ser.inWaiting() > 0:
        pico_data = ser.readline()
        pico_data = pico_data.decode('utf-8', 'ignore')
        print(pico_data)
for i in reversed(range(100)): # forward ramp down
    duty_th = THROTTLE_STALL + int(THROTTLE_FWD_RANGE * i / 100)
    msg = (str(duty_st) + "," + str(duty_th) + "\n").encode('utf-8')
    # msg = f"{i/100}\n".encode('utf-8')
    ser.write(msg)
    sleep(0.1)
    if ser.inWaiting() > 0:
        pico_data = ser.readline()
        pico_data = pico_data.decode('utf-8', 'ignore')
        print(pico_data)
for i in range(100): # reverse ramp up
    duty_th = THROTTLE_STALL - int(THROTTLE_REV_RANGE * i / 100)
    msg = (str(duty_st) + "," + str(duty_th) + "\n").encode('utf-8')
    # msg = f"{-i/100}\n".encode('utf-8')
    ser.write(msg)
    sleep(0.1)
    if ser.inWaiting() > 0:
        pico_data = ser.readline()
        pico_data = pico_data.decode('utf-8', 'ignore')
        print(pico_data)
for i in reversed(range(100)): # reverse ramp down
    duty_th = THROTTLE_STALL - int(THROTTLE_REV_RANGE * i / 100)
    msg = (str(duty_st) + "," + str(duty_th) + "\n").encode('utf-8')
    # msg = f"{-i/100}\n".encode('utf-8')
    ser.write(msg)
    sleep(0.1)
    if ser.inWaiting() > 0:
        pico_data = ser.readline()
        pico_data = pico_data.decode('utf-8', 'ignore')
        print(pico_data)
ser.close()
