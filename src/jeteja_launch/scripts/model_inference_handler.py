import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import scripts.postprocessing as postprocessing
import config.master_config as master_config

MODEL_PATH = master_config.MODEL_PATH
COLOR_PREPROCESS_DATA_TYPE = master_config.COLOR_PREPROCESS_DATA_TYPE
DEPTH_PREPROCESS_DATA_TYPE = master_config.DEPTH_PREPROCESS_DATA_TYPE
PWM_OUTPUT_DATA_TYPE = master_config.PWM_OUTPUT_DATA_TYPE

class TensorRTInference:
    def __init__(self):
        model_path = MODEL_PATH
        print(f"Loading TensorRT model from: {model_path}")
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load the TensorRT engine
        with open(model_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Inspect and map bindings
        self.bindings = {}
        self.binding_shapes = {}
        self.binding_types = {}
        self._inspect_bindings()

        # Input and output bindings
        self.color_input_idx = self.bindings.get('color_input')
        self.depth_input_idx = self.bindings.get('depth_input')
        self.output_0_idx = self.bindings.get('output_0')
        self.output_1_idx = self.bindings.get('output_1')

        # Allocate memory on GPU
        self.d_color_input = cuda.mem_alloc(
            int(np.prod(self.binding_shapes['color_input']) * COLOR_PREPROCESS_DATA_TYPE(1).nbytes)
        )
        self.d_depth_input = cuda.mem_alloc(
            int(np.prod(self.binding_shapes['depth_input']) * DEPTH_PREPROCESS_DATA_TYPE(1).nbytes)
        )
        self.d_output_0 = cuda.mem_alloc(
            int(np.prod(self.binding_shapes['output_0']) * PWM_OUTPUT_DATA_TYPE(1).nbytes)
        )
        self.d_output_1 = cuda.mem_alloc(
            int(np.prod(self.binding_shapes['output_1']) * PWM_OUTPUT_DATA_TYPE(1).nbytes)
        )

        # Host buffers for outputs
        self.h_output_0 = np.empty(self.binding_shapes['output_0'], dtype=PWM_OUTPUT_DATA_TYPE)
        self.h_output_1 = np.empty(self.binding_shapes['output_1'], dtype=PWM_OUTPUT_DATA_TYPE)

    def _inspect_bindings(self):
        """Inspect engine bindings and populate binding information."""
        print("Inspecting TensorRT Engine Bindings:")
        for idx in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(idx)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
            self.bindings[tensor_name] = idx
            self.binding_shapes[tensor_name] = tensor_shape
            self.binding_types[tensor_name] = tensor_dtype

            print(f"Binding {idx}:")
            print(f"  Name: {tensor_name}")
            print(f"  Shape: {tensor_shape}")
            print(f"  Data Type: {tensor_dtype}")

    def infer(self, color_image, depth_image):
        """Run inference on input images."""

        # print(f"Color Image: shape={color_image.shape}, dtype={color_image.dtype}, range=({color_image.min()}, {color_image.max()})")
        # print(f"Depth Image: shape={depth_image.shape}, dtype={depth_image.dtype}, range=({depth_image.min()}, {depth_image.max()})")

        # Transfer data to device
        cuda.memcpy_htod(self.d_color_input, color_image)
        cuda.memcpy_htod(self.d_depth_input, depth_image)

        # Perform inference
        bindings = [None] * self.engine.num_io_tensors
        bindings[self.color_input_idx] = int(self.d_color_input)
        bindings[self.depth_input_idx] = int(self.d_depth_input)
        bindings[self.output_0_idx] = int(self.d_output_0)
        bindings[self.output_1_idx] = int(self.d_output_1)

        self.context.execute_v2(bindings)

        # Transfer output back to host
        cuda.memcpy_dtoh(self.h_output_0, self.d_output_0)
        cuda.memcpy_dtoh(self.h_output_1, self.d_output_1)

        return self.h_output_0, self.h_output_1

COLOR_WIDTH = master_config.COLOR_WIDTH
COLOR_HEIGHT = master_config.COLOR_HEIGHT
COLOR_CHANNELS = master_config.COLOR_CHANNELS
COLOR_PREPROCESS_DATA_TYPE = master_config.COLOR_PREPROCESS_DATA_TYPE
COLOR_NORMALIZATION_FACTOR = master_config.COLOR_NORMALIZATION_FACTOR
DEPTH_WIDTH = master_config.DEPTH_WIDTH
DEPTH_HEIGHT = master_config.DEPTH_HEIGHT
DEPTH_CHANNELS = master_config.DEPTH_CHANNELS
DEPTH_PREPROCESS_DATA_TYPE = master_config.DEPTH_PREPROCESS_DATA_TYPE
DEPTH_NORMALIZATION_FACTOR = master_config.DEPTH_NORMALIZATION_FACTOR

def main():
    # Generate test data for color_image (shape: (360, 640, 3))
    color_image = np.random.rand(COLOR_WIDTH, COLOR_HEIGHT, COLOR_CHANNELS).astype(COLOR_PREPROCESS_DATA_TYPE)
    color_image /= COLOR_NORMALIZATION_FACTOR  # Normalize to [0, 1], if applicable to your preprocessing

    # Generate test data for depth_image (shape: (360, 640, 1))
    depth_image = np.random.rand(DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_CHANNELS).astype(DEPTH_PREPROCESS_DATA_TYPE)
    depth_image /= DEPTH_NORMALIZATION_FACTOR  # Normalize to [0, 1], if applicable to your preprocessing

    print(f"Color Image: shape={color_image.shape}, dtype={color_image.dtype}, range=({color_image.min()}, {color_image.max()})")
    print(f"Depth Image: shape={depth_image.shape}, dtype={depth_image.dtype}, range=({depth_image.min()}, {depth_image.max()})")

    # Prepare inputs
    color_image = np.expand_dims(color_image, axis=0).astype(COLOR_PREPROCESS_DATA_TYPE)  # Add batch dimension
    depth_image = np.expand_dims(depth_image, axis=0).astype(DEPTH_PREPROCESS_DATA_TYPE)  # Add batch dimension

    # Pass to inference method
    model = TensorRTInference()
    outputs = model.infer(color_image, depth_image)

    speed, steering = postprocessing.denormalize_pwm(outputs)

    print(f"Output 0: {speed}")
    print(f"Output 1: {steering}")

if __name__ == "__main__":
    main()

