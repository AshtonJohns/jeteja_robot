import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os

# Path to your ONNX model
onnx_model_path = "/path/to/your_model.onnx"  # Replace with your ONNX file path
trt_engine_path = "your_model.trt"  # The output TensorRT engine file

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Function to convert ONNX to TensorRT
def build_engine(onnx_file_path, engine_file_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 30  # 1GB workspace
        builder.max_batch_size = 1

        # Parse the ONNX file
        with open(onnx_file_path, "rb") as model:
            if not parser.parse(model.read()):
                print("Failed to parse the ONNX file")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Build the TensorRT engine
        print("Building TensorRT engine. This may take a few minutes...")
        engine = builder.build_cuda_engine(network)
        if engine:
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            print(f"TensorRT engine saved to {engine_file_path}")
        else:
            print("Failed to build the TensorRT engine.")
        return engine

# Convert the ONNX model to TensorRT
build_engine(onnx_model_path, trt_engine_path)
