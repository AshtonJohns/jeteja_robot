import serial
import pygame
import cv2 as cv
import pyrealsense2 as rs
import numpy as np

def setup_realsense_camera():
    # Configure both color and depth streams
    pipeline = rs.pipeline()  # type: ignore
    config = rs.config()  # type: ignore
    config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)  # RGB stream
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)  # Depth stream
    pipeline.start(config)
    return pipeline

def get_realsense_frame(pipeline):
    # Wait for coherent color and depth frames
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        return False, None, None

    # Convert color and depth frames to NumPy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Resize images to 120x160
    color_image_resized = cv.resize(color_image, (160, 120))
    depth_image_resized = cv.resize(depth_image, (160, 120))

    return True, color_image_resized, depth_image_resized

def setup_serial(port, baudrate=115200):
    ser = serial.Serial(port=port, baudrate=baudrate)
    print(f"Serial connected on {ser.name}")
    return ser

def setup_joystick():
    pygame.joystick.init()
    js = pygame.joystick.Joystick(0)
    js.init()
    return js

def encode_dutycylce(ax_val_st, ax_val_th, params):
    # Constants
    STEERING_AXIS = params['steering_joy_axis']
    STEERING_CENTER = params['steering_center']
    STEERING_RANGE = params['steering_range']
    THROTTLE_AXIS = params['throttle_joy_axis']
    THROTTLE_STALL = params['throttle_stall']
    THROTTLE_FWD_RANGE = params['throttle_fwd_range']
    THROTTLE_REV_RANGE = params['throttle_rev_range']
    THROTTLE_LIMIT = params['throttle_limit']
    RECORD_BUTTON = params['record_btn']
    STOP_BUTTON = params['stop_btn']
    
    # Calculate steering and throttle value
    act_st = -ax_val_st
    act_th = -ax_val_th  # throttle action: -1: max forward, 1: max backward

    # Encode steering value to duty cycle in nanoseconds
    duty_st = STEERING_CENTER - STEERING_RANGE + int(STEERING_RANGE * (act_st + 1))
    # Encode throttle value to duty cycle in nanoseconds
    if act_th > 0:
        duty_th = THROTTLE_STALL + int(THROTTLE_FWD_RANGE * min(act_th, THROTTLE_LIMIT))
    elif act_th < 0:
        duty_th = THROTTLE_STALL + int(THROTTLE_REV_RANGE * max(act_th, -THROTTLE_LIMIT))
    else:
        duty_th = THROTTLE_STALL
    duty_st = round(duty_st, 2)
    duty_th = round(duty_th, 2)
    msg = encode(duty_st, duty_th)
    return msg

def encode(duty_st, duty_th):
    msg = (str(duty_st) + "," + str(duty_th) + "\n").encode('utf-8')
    return msg

def encode_throttle(throttle_value, params):
    throttle_limit = min(max(throttle_value, -1), 1)
    if throttle_limit > 0:
        return params['throttle_stall'] + int(params['throttle_fwd_range'] * throttle_limit)
    elif throttle_limit < 0:
        return params['throttle_stall'] + int(params['throttle_rev_range'] * throttle_limit)
    else:
        return params['throttle_stall']
