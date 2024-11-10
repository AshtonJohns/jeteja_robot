import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os

# Path to your ONNX model

onnx_model_path = "/home/ucajetson/UCAJetson/data/2024-11-09-16-30/DonkeyNet-15epochs-0.001lr-JetsonTest3.onnx"  # Replace with your ONNX file path
trt_engine_path = "/home/ucajetson/UCAJetson/models/TensorRT_JetsonTest3.trt"  # The output TensorRT engine file

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Function to convert ONNX to TensorRT
def build_engine(onnx_file_path, engine_file_path):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        # Create the config object and set max_workspace_size here
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB workspace
        builder.max_batch_size = 1

        # Parse the ONNX file
        with open(onnx_file_path, "rb") as model:
            if not parser.parse(model.read()):
                print("Failed to parse the ONNX file")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Build the TensorRT engine with the config
        print("Building TensorRT engine. This may take a few minutes...")
        engine = builder.build_engine(network, config)
        if engine:
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            print(f"TensorRT engine saved to {engine_file_path}")
        else:
            print("Failed to build the TensorRT engine.")
        return engine

# Convert the ONNX model to TensorRT
build_engine(onnx_model_path, trt_engine_path)