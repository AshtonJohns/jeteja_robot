import yaml
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
from ament_index_python.packages import get_package_share_directory

realsense2_camera_config = os.path.join(
    get_package_share_directory('jeteja_launch'),
    'config',
    'realsense2_camera.yaml'
)

autopilot_config = os.path.join(
    get_package_share_directory('jeteja_launch'),
    'config',
    'autopilot.yaml'
)

# Parse the realsense camera YAML file
with open(realsense2_camera_config, 'r') as file:
    config = yaml.safe_load(file)

# Color camera settings
COLOR_HEIGHT = config['rgb_camera.color_profile'].split("x")[0]
COLOR_WIDTH = config['rgb_camera.color_profile'].split("x")[1]
COLOR_FORMAT = config['rgb_camera.color_format']

DEPTH_HEIGHT = config['depth_module.depth_profile'].split("x")[0]
DEPTH_WIDTH = config['depth_module.depth_profile'].split("x")[1]
COLOR_FORMAT = config['depth_module.depth_format']

# Parse the autopilot YAML file
with open(autopilot_config, 'r') as file:
    config = yaml.safe_load(file)

# Extract parameters from the YAML configuration
COLOR_NORMALIZATION_FACTOR = config.get('COLOR_NORMALIZATION_FACTOR')
COLOR_DATA_TYPE = config.get('COLOR_DATA_TYPE')
COLOR_ENCODING = config.get('COLOR_ENCODING')
COLOR_INPUT_IDX = config.get('COLOR_INPUT_IDX')

DEPTH_NORMALIZATION_FACTOR = config.get('DEPTH_NORMALIZATION_FACTOR')
DEPTH_DATA_TYPE = config.get('DEPTH_DATA_TYPE')
DEPTH_ENCODING = config.get('DEPTH_ENCODING')
DEPTH_INPUT_IDX = config.get('DEPTH_INPUT_IDX')

BATCH_SIZE = config.get('BATCH_SIZE')
OUTPUT_IDX = config.get('OUTPUT_IDX')
COLOR_CHANNELS = config['COLOR_CHANNELS']
DEPTH_CHANNELS = config['DEPTH_CHANNELS']
OUTPUT_SHAPE = config['OUTPUT_SHAPE']

class TensorRTInference:
    def __init__(self, trt_model_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(trt_model_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # # Binding indices based on your model's TensorRT engine
        # self.color_input_idx = 0
        # self.depth_input_idx = 1
        # self.output_idx = 2

        # Buffer sizes
        self.color_input_shape = (BATCH_SIZE, COLOR_WIDTH, COLOR_HEIGHT, COLOR_CHANNELS)
        self.depth_input_shape = (BATCH_SIZE, DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_CHANNELS)
        self.output_shape = OUTPUT_SHAPE

        self.color_input_size = np.prod(self.color_input_shape) * np.float32(1).nbytes
        self.depth_input_size = np.prod(self.depth_input_shape) * np.float32(1).nbytes
        self.output_size = np.prod(self.output_shape) * np.float32(1).nbytes

        # Allocate memory on GPU
        self.d_color_input = cuda.mem_alloc(int(self.color_input_size))
        self.d_depth_input = cuda.mem_alloc(int(self.depth_input_size))
        self.d_output = cuda.mem_alloc(int(self.output_size))

        # Host output buffer
        self.h_output = np.empty(self.output_shape, dtype=np.float32)


    def infer(self, color_image, depth_image):
        # Prepare inputs
        color_image = np.expand_dims(color_image, axis=0).astype(np.float32)  # Add batch dimension
        depth_image = np.expand_dims(depth_image, axis=0).astype(np.float32)  # Add batch dimension

        # Transfer data to device
        cuda.memcpy_htod(self.d_color_input, color_image)
        cuda.memcpy_htod(self.d_depth_input, depth_image)

        # Perform inference
        bindings = [int(self.d_color_input), int(self.d_depth_input), int(self.d_output)]
        self.context.execute_v2(bindings)

        # Transfer output back to host
        cuda.memcpy_dtoh(self.h_output, self.d_output)

        return self.h_output


if __name__ == '__main__':
    TensorRTInference('model.trt')