import tensorflow as tf
import numpy as np
import shutil
import os
from tensorflow.keras.layers import Layer
from utils.file_utilities import get_latest_directory

def convert_to_tflite(saved_model_dir, tflite_file):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # TensorFlow Lite operations
        tf.lite.OpsSet.SELECT_TF_OPS     # TensorFlow operations
    ]
    tflite_model = converter.convert()

    # Save the converted model
    with open(tflite_file, 'wb') as f:
        f.write(tflite_model)


# Define the Cast layer
class Cast(Layer):
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

# Load the pre-trained model
models_dir = os.path.join('.','data','models')
latest_models_dir = get_latest_directory(models_dir).strip("_Ready _SavedModel")

final_model_keras_path = os.path.join(latest_models_dir, 'final_model.keras')
model = tf.keras.models.load_model(final_model_keras_path, custom_objects={"Cast": Cast})

# Export the model in SavedModel format
export_file_path = latest_models_dir + '_SavedModel'
model.export(export_file_path)

SAVED_MODEL_DIR = export_file_path
tflite_dir = latest_models_dir + '_Ready'
os.makedirs(tflite_dir,exist_ok=True)
TFLITE_FILE = os.path.join(tflite_dir,"best_model_optimized.tflite") # Output TFLite file
convert_to_tflite(SAVED_MODEL_DIR, TFLITE_FILE)
