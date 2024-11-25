import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np
import os
from utils.file_utilities import get_latest_directory

# Paths
models_dir = os.path.join(".","data","models")
MODEL_DIR = get_latest_directory(models_dir)
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
SAVED_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_savedmodel.keras")
TENSORRT_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_tensorrt")

# Convert Keras model to TensorFlow SavedModel format
def convert_to_saved_model(keras_model_path, saved_model_path):
    print("Converting Keras model to TensorFlow SavedModel...")
    model = tf.keras.models.load_model(keras_model_path)
    model.save(saved_model_path)
    print(f"SavedModel saved at: {saved_model_path}")

# Optimize the model using TensorRT
def optimize_with_tensorrt(saved_model_path, tensorrt_model_path):
    print("Optimizing model with TensorRT...")
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode=trt.TrtPrecisionMode.FP16,  # Use FP16 for Jetson
        max_workspace_size_bytes=1 << 30          # 1GB workspace
    )

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=saved_model_path,
        conversion_params=conversion_params
    )
    converter.convert()
    converter.save(output_saved_model_dir=tensorrt_model_path)
    print(f"TensorRT-optimized model saved at: {tensorrt_model_path}")

# Test the optimized TensorRT model
def test_model(tensorrt_model_path):
    print("Loading and testing TensorRT model...")
    model = tf.saved_model.load(tensorrt_model_path)

    # Example input (adjust to match your preprocessing pipeline)
    color_input = np.random.rand(1, 360, 640, 3).astype(np.float32)  # Simulated color input
    depth_input = np.random.rand(1, 360, 640, 1).astype(np.float32)  # Simulated depth input

    # Run inference
    input_data = {"color_input": color_input, "depth_input": depth_input}
    output = model(input_data)

    print("Model inference results:")
    print(f"Linear velocity: {output['linear_x'].numpy()}")
    print(f"Angular velocity: {output['angular_z'].numpy()}")

# Main execution flow
if __name__ == "__main__":
    # Ensure model directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Step 1: Convert Keras model to TensorFlow SavedModel
    convert_to_saved_model(KERAS_MODEL_PATH, SAVED_MODEL_PATH)

    # Step 2: Optimize the model with TensorRT
    optimize_with_tensorrt(SAVED_MODEL_PATH, TENSORRT_MODEL_PATH)

    # Step 3: Test the optimized TensorRT model
    test_model(TENSORRT_MODEL_PATH)
