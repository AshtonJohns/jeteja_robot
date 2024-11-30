import os

def write_run_trt_optimizer_script(color_width, color_length, depth_width, depth_length, output_file="run_trt.sh"):
    bash_content = f"""#!/bin/bash

# Set maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks  # Lock GPU and CPU frequencies

# Run TensorRT execution with ONNX model and specified input shapes
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16 --shapes=color_input:1x{color_width}x{color_length}x3,depth_input:1x{depth_width}x{depth_length}x1
"""
    # Write the content to the specified file
    with open(output_file, "w") as bash_file:
        bash_file.write(bash_content)

    import os
    os.chmod(output_file, 0o755)

    print(f"Bash script '{output_file}' written and made executable.")


def write_savedmodel_to_onnx_script(output_file):
    bash_content = f"""#!/bin/bash

# Set maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks  # Lock GPU and CPU frequencies

# Convert savedmodel 
python -m tf2onnx.convert --saved-model SavedModel --output model.onnx
"""
    # Write the content to the specified file
    with open(output_file, "w") as bash_file:
        bash_file.write(bash_content)

    os.chmod(output_file, 0o755)

    print(f"Bash script '{output_file}' written and made executable.")


if __name__ == '__main__':
    write_savedmodel_to_onnx_script('example1.bash')
    write_run_trt_optimizer_script(360,640,360,640,'example2.bash')
