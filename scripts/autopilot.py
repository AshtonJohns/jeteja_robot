import os
import sys
import json
import torch
#from models import MultiModalNet
from hardware import get_realsense_frame, setup_realsense_camera, setup_serial, setup_joystick, encode_dutycylce, encode_throttle, encode
from torchvision import transforms
import pygame
import cv2 as cv
from convnets import DonkeyNet
from time import time  # Import time module for frame rate calculation

#Adds dummy to run Pygame without a display
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize only the required Pygame modules
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)


# SETUP
# Load model
model_path = os.path.join('models', 'DonkeyNet-15epochs-0.001lr.pth') #Change to name of pth file you want to use
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
is_paused = True #True to enable pause mode, False to not enable pause mode ***Only put True if you have pause button***
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
        ret, frame = get_realsense_frame(cam)
        if not ret or frame is None:
            print("No frame received. TERMINATE!")
            break

        for e in pygame.event.get():
            if e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(params['pause_btn']):
                    is_paused = not is_paused
                    #headlight.toggle()
                elif js.get_button(params['stop_btn']):
                    print("E-STOP PRESSED. TERMINATE!")
                    break

        # Predict steering and throttle
        img_tensor = to_tensor(cv.resize(frame, (320,240))).unsqueeze(0)  # Add batch dimension
        pred_st, pred_th = model(img_tensor).squeeze()
        st_trim = float(pred_st)
        if st_trim >= 1:  # trim steering signal
            st_trim = .999
        elif st_trim <= -1:
            st_trim = -.999
        th_trim = (float(pred_th))
        if th_trim >= 1:  # trim throttle signal
            th_trim = .999
        elif th_trim <= -1:
            th_trim = -.999
        # Encode steering value to dutycycle in nanosecond
        if is_paused:
            duty_st = STEERING_CENTER
        else:
            duty_st = STEERING_CENTER - STEERING_RANGE + int(STEERING_RANGE * (st_trim + 1))
        # Encode throttle value to dutycycle in nanosecond
        if is_paused:
            duty_th = THROTTLE_STALL
        else:
            if th_trim > 0:
                duty_th = THROTTLE_STALL + int(THROTTLE_FWD_RANGE * min(th_trim, THROTTLE_LIMIT))
            elif th_trim < 0:
                duty_th = THROTTLE_STALL + int(THROTTLE_REV_RANGE * max(th_trim, -THROTTLE_LIMIT))
            else:
                duty_th = THROTTLE_STALL

        #print(f"{pred_st},{pred_th}")

        # Encode and send commands
        if not is_paused:
            msg = encode_dutycylce(pred_st,pred_th,params)
        else:
            duty_st, duty_th = params['steering_center'], params['throttle_stall']
            msg = encode(duty_st,duty_th)
            
        
        ser_pico.write(msg)

        # Calculate and print frame rate
        frame_count += 1
        current_time = time()
        if current_time - prev_time >= 1.0:
            fps = frame_count / (current_time - prev_time)
            print(f"Autopilot Frame Rate: {fps:.2f} FPS")
            prev_time = current_time
            frame_count = 0

except KeyboardInterrupt:
    print("Terminated by user.")
finally:
    # cam.stop()
    pygame.joystick.quit()
    ser_pico.close()
    cv.destroyAllWindows()
