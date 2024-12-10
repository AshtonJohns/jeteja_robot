"""
If ssh from other machine, please enable X11 forwarding, either `ssh -X` or `ssh -Y`
"""
import sys
import cv2
from picamera2 import Picamera2
from time import sleep

# SETUP
print("Please adjust lens focus if blurry")
for i in reversed(range(1, 4)):
    print(i)
    sleep(1)
cv2.startWindowThread()
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
picam2.start()

# LOOP
while True:
    im = picam2.capture_array()
    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Camera", im)
    # Press "q" to quit
    if cv2.waitKey(1)==ord('q'):
        cv2.destroyAllWindows()
        sys.exit()
