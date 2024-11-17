import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os

# Path to your ONNX model
onnx_model_path = "/home/ucajetson/UCAJetson/data/2024-11-10-12-47/DonkeyNet-15epochs-0.001lr.onnx"  # Replace with your ONNX file path
trt_engine_path = "/home/ucajetson/UCAJetson/models/TensorRT_JetsonTest4.trt"  # The output TensorRT engine file

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Function to convert ONNX to TensorRT
def build_engine(onnx_file_path, engine_file_path):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        # Create the config object and set max_workspace_size here
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB workspace
        builder.max_batch_size = 1  # Explicit batch mode

        # Parse the ONNX file
        with open(onnx_file_path, "rb") as model:
            print(f"Parsing ONNX model: {onnx_file_path}")
            if not parser.parse(model.read()):
                print("Failed to parse the ONNX file")
                for error in range(parser.num_errors):
                    print(f"Error {error + 1}: {parser.get_error(error)}")
                return None

        # Validate input dimensions
        input_tensor = network.get_input(0)
        input_shape = tuple(input_tensor.shape)
        expected_shape = (1, 4, 120, 160)  # Batch size 1, 4 channels, 120x160 resolution

        if input_shape != expected_shape:
            print(f"Error: The ONNX model's input shape {input_shape} does not match the expected shape {expected_shape}.")
            return None
        else:
            print(f"Input shape is correct: {input_shape}")

        # Set optimization profiles if needed (optional)
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name, (1, 4, 120, 160), (1, 4, 120, 160), (1, 4, 120, 160))
        config.add_optimization_profile(profile)

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

# Check if ONNX model file exists
if not os.path.exists(onnx_model_path):
    print(f"Error: ONNX model file not found at {onnx_model_path}")
    sys.exit(1)

# Create directory for TensorRT engine if it doesn't exist
os.makedirs(os.path.dirname(trt_engine_path), exist_ok=True)

# Convert the ONNX model to TensorRT
engine = build_engine(onnx_model_path, trt_engine_path)
if engine:
    print("TensorRT engine successfully created.")
else:
    print("TensorRT engine creation failed.")
