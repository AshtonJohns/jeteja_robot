import os
import sys
import json
import torch
from hardware import get_realsense_frame, setup_realsense_camera, setup_serial, setup_joystick, encode_dutycylce, encode_throttle
from torchvision import transforms
import pygame
import cv2 as cv
from convnets import DonkeyNet
from time import time  # Import time module for frame rate calculation

# Adds dummy to run Pygame without a display
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize only the required Pygame modules
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)

# SETUP
# Load model
model_path = os.path.join('models', 'DonkeyNet-15epochs-0.001lr-JetsonTest3.pth')  # Adjust to your .pth file path
model = DonkeyNet()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Load configs
params_file_path = os.path.join(sys.path[0], 'config_new.json')
with open(params_file_path) as params_file:
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
RECORD_BUTTON = params['record_btn']
STOP_BUTTON = params['stop_btn']
PAUSE_BUTTON = params['pause_btn']

# Initialize hardware
try:
    ser_pico = setup_serial(port='/dev/ttyACM0', baudrate=115200)
except:
    ser_pico = setup_serial(port='/dev/ttyACM1', baudrate=115200)

cam = setup_realsense_camera()
js = setup_joystick()

to_tensor = transforms.ToTensor()
is_paused = True
frame_counts = 0

# Frame rate calculation variables
prev_time = time()
frame_count = 0
fps = 0

# Initialize Pygame for joystick handling
pygame.init()

# MAIN LOOP
try:
    while True:
        # Get both RGB and LiDAR frames
        ret, frame = get_realsense_frame(cam)
        if not ret or frame is None:
            print("No frame received. TERMINATE!")
            break

        for e in pygame.event.get():
            if e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(params['pause_btn']):
                    is_paused = not is_paused
                elif js.get_button(params['stop_btn']):
                    print("E-STOP PRESSED. TERMINATE!")
                    break

        # Process the frame for prediction
        img_tensor = to_tensor(cv.resize(frame, (160, 120))).unsqueeze(0)  # Resize to 120x160 and add batch dimension
        pred_st, pred_th = model(img_tensor).squeeze()

        # Clip predictions to valid range
        st_trim = max(min(float(pred_st), 0.999), -0.999)
        th_trim = max(min(float(pred_th), 0.999), -0.999)

        # Encode and send commands
        if not is_paused:
            msg = encode_dutycylce(st_trim, th_trim, params)
        else:
            duty_st, duty_th = params['steering_center'], params['steering_center']  # Set to center if paused
            msg = encode_dutycylce(duty_st, duty_th, params)

        # Send message to hardware
        if ser_pico is not None:
            ser_pico.write(msg)

        # Frame rate calculation and display
        frame_count += 1
        if time() - prev_time >= 1:
            fps = frame_count
            frame_count = 0
            prev_time = time()
            print(f"FPS: {fps}")

except KeyboardInterrupt:
    print("Autopilot terminated by user.")

finally:
    if ser_pico:
        ser_pico.close()
    pygame.quit()
    print("Autopilot cleanup complete.")
