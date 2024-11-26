import tensorflow as tf
import shutil
import os
from tensorflow.keras.layers import Layer
from utils.file_utilities import get_latest_directory

# Define the Cast layer
class Cast(Layer):
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

# Load the pre-trained model
models_dir = os.path.join('.','data','models')
latest_models_dir = get_latest_directory(models_dir)
if 'opt' in latest_models_dir:
    shutil.rmtree(latest_models_dir,ignore_errors=True)
    latest_models_dir = get_latest_directory(models_dir)
final_model_keras_path = os.path.join(latest_models_dir, 'final_model.keras')
model = tf.keras.models.load_model(final_model_keras_path, custom_objects={"Cast": Cast})

# Export the model in SavedModel format
export_file_path = latest_models_dir + ' opt'
model.export(export_file_path)
