import os
import numpy as np
import tensorflow as tf
from utils.file_utilities import get_latest_directory

# Input preparation (same as above)
color_input = np.random.random((1, 360, 640, 3)).astype(np.float32)
depth_input = np.random.random((1, 360, 640, 1)).astype(np.float32)

inputs = [color_input, depth_input]

# Load the TFLite model
models_dir = os.path.join('.','data','models')
latest_models_dir = get_latest_directory(models_dir)
print(latest_models_dir)
if not latest_models_dir.endswith('_Ready'):
    raise FileNotFoundError("You must have a tflite directory")
model_path = os.path.join(latest_models_dir,'best_model_optimized.tflite')
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set input data
interpreter.set_tensor(input_details[0]['index'], color_input)
interpreter.set_tensor(input_details[1]['index'], depth_input)

# Run inference
interpreter.invoke()

# Get the output
output1 = interpreter.get_tensor(output_details[0]['index'])
output2 = interpreter.get_tensor(output_details[1]['index'])

# Print the outputs
print("Output 1:", output1)
print("Output 2:", output2)
