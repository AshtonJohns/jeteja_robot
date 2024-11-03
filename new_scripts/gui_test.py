import cv2
import numpy as np

# Create a black image
img = np.zeros((512, 512, 3), np.uint8)
cv2.putText(img, 'Test Image', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Display it
cv2.imshow("Test Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
