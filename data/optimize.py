import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np
import os
from utils.file_utilities import get_latest_directory

raise Warning("This is not implemented yet.")

# Paths
models_dir = os.path.join(".", "data", "models")
latest_model_dir = get_latest_directory(models_dir)
if "_converted" not in latest_model_dir:
    raise FileNotFoundError("Only model.exports (not keras) are being used.")
EXPORTED_MODEL_PATH = latest_model_dir  # Exported model directory
TENSORRT_MODEL_PATH = latest_model_dir + "_optimized"  # TensorRT-optimized model directory

# Optimize the model using TensorRT
def optimize_with_tensorrt(saved_model_path, tensorrt_model_path):
    print(f"Optimizing model with TensorRT from: {saved_model_path}")
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode=trt.TrtPrecisionMode.FP16,  # Use FP16 for Jetson
        max_workspace_size_bytes=1 << 30          # 1GB workspace
    )

    # Set up the converter
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=saved_model_path,
        conversion_params=conversion_params
    )
    # Perform the conversion
    converter.convert()

    # Save the optimized model
    converter.save(output_saved_model_dir=tensorrt_model_path)
    print(f"TensorRT-optimized model saved at: {tensorrt_model_path}")

# Test the optimized TensorRT model
def test_model(tensorrt_model_path):
    print(f"Loading and testing TensorRT model from: {tensorrt_model_path}")
    model = tf.saved_model.load(tensorrt_model_path)

    # Example input (adjust to match your preprocessing pipeline)
    color_input = np.random.rand(1, 360, 640, 3).astype(np.float32)  # Simulated color input
    depth_input = np.random.rand(1, 360, 640, 1).astype(np.float32)  # Simulated depth input

    # Prepare inputs
    inputs = {"color_input": color_input, "depth_input": depth_input}

    # Run inference
    infer = model.signatures["serving_default"]  # Default signature for inference
    outputs = infer(**inputs)

    # Print results
    print("Model inference results:")
    for key, value in outputs.items():
        print(f"{key}: {value.numpy()}")

# Main execution flow
if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(TENSORRT_MODEL_PATH, exist_ok=True)

    # Step 1: Optimize the model with TensorRT
    optimize_with_tensorrt(EXPORTED_MODEL_PATH, TENSORRT_MODEL_PATH)

    # Step 2: Test the optimized TensorRT model
    test_model(TENSORRT_MODEL_PATH)
