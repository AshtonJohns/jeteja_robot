import os
import tensorflow as tf
from utils.file_utilities import get_latest_directory

def convert_keras_to_hdf5(input_dir, output_dir):
    """
    Convert all `.keras` model files in the input directory to `.h5` format
    and save them in the output directory.

    Parameters:
        input_dir (str): Path to the directory containing `.keras` model files.
        output_dir (str): Path to the directory to save `.h5` model files.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all `.keras` files in the input directory
    keras_files = [f for f in os.listdir(input_dir) if f.endswith(".keras")]

    if not keras_files:
        print("No .keras files found in the specified directory.")
        return

    for keras_file in keras_files:
        input_path = os.path.join(input_dir, keras_file)
        output_file = keras_file.replace(".keras", ".h5")
        output_path = os.path.join(output_dir, output_file)

        print(f"Converting {keras_file} to {output_file}...")

        # Load the Keras model
        model = tf.keras.models.load_model(input_path)

        # Save the model in HDF5 format
        model.save(output_path, save_format="h5")

        print(f"Saved HDF5 model to: {output_path}")

    print("All .keras models have been successfully converted to .h5 format.")

if __name__ == "__main__":
    models_dir = os.path.join('.','data','models')
    latest_models_dir = get_latest_directory(models_dir)
    input_directory = latest_models_dir
    output_directory = os.path.join(latest_models_dir,'converted_h5')

    # Convert .keras to .h5
    convert_keras_to_hdf5(input_directory, output_directory)
