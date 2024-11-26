import tensorflow as tf
import os
from utils.file_utilities import get_latest_directory

# Load the model
models_dir = os.path.join('.','data','models')
latest_opt_model_dir = get_latest_directory((models_dir))
model = tf.saved_model.load(latest_opt_model_dir)

# Inspect available signatures
print(list(model.signatures.keys()))

# Prepare sample inputs
color_input = tf.random.uniform([1, 360, 640, 3], dtype=tf.float32)
depth_input = tf.random.uniform([1, 360, 640, 1], dtype=tf.float32)

# Run inference
result = model.signatures['serving_default'](color_input=color_input, depth_input=depth_input)
print(result)
