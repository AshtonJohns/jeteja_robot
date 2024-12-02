import tensorflow as tf
import glob
import os
import pandas as pd
import cv2
import numpy as np
from utils.file_utilities import get_latest_directory
from sklearn.model_selection import train_test_split


def process_commands(commands_path, output_dir, **kwargs):
    """
    Processes the commands.csv file to find existing image files or remove rows with missing images.

    :param commands_path: Path to the commands.csv file
    :param output_dir: Output directory for preprocessed data
    :param kwargs:
        -   color_dir: Directory containing color images
        -   depth_dir: Directory containing depth images
    """
    color_dir = kwargs.get('color_dir', False)
    depth_dir = kwargs.get('depth_dir', False)

    # Load the CSV file into a DataFrame
    commands_df = pd.read_csv(commands_path)

    # Process color image filenames
    if color_dir and 'color_image_filename' in commands_df.columns:
        print("Validating color image filenames...")
        commands_df['color_image_filename'] = commands_df['color_image_filename'].apply(
            lambda x: find_image(x, color_dir) if pd.notnull(x) else None
        )

    # Process depth image filenames
    if depth_dir and 'depth_image_filename' in commands_df.columns:
        print("Validating depth image filenames...")
        commands_df['depth_image_filename'] = commands_df['depth_image_filename'].apply(
            lambda x: find_image(x, depth_dir) if pd.notnull(x) else None
        )

    # Remove rows where either color or depth image is missing
    commands_df.dropna(subset=['color_image_filename', 'depth_image_filename'], inplace=True)

    # Save the updated DataFrame back to a CSV
    processed_commands_path = os.path.join(output_dir, "commands.csv")
    commands_df.to_csv(processed_commands_path, index=False)

    print(f"Commands file processed and saved to {processed_commands_path}.")


def find_image(image_name, search_dir):
    """
    Search for an image file in a directory, matching up to the second underscore,
    without including the file extension in the pattern.

    :param image_name: The original file name to find (from the CSV).
    :param search_dir: The directory to search for the file.
    :return: The first matched file name (with its actual extension), or None if no valid match is found.
    """
    if not search_dir or not image_name:
        return None

    base_name, _ = os.path.splitext(image_name)  # Ignore the extension
    parts = base_name.split('_')

    # Ensure the file name has at least two underscore-separated parts
    if len(parts) < 3:
        return None

    # Extract the prefix (up to the second underscore)
    prefix = '_'.join(parts[:2])
    nanoseconds = parts[2]

    # Start with the full nanoseconds and truncate step-by-step
    for i in range(len(nanoseconds), 0, -1):
        pattern = os.path.join(search_dir, f"{prefix}_{nanoseconds[:i]}*")  # Match any extension
        print(f"Searching with pattern: {pattern}")  # Debug output
        matches = glob.glob(pattern)
        if matches:
            print(f"Match found: {matches[0]}")  # Debug output
            return os.path.basename(matches[0])  # Return the full matched file name with its actual extension

    # If the final pattern ends with `_`, it's invalid
    if len(nanoseconds) == 0 or f"{prefix}_" in pattern:
        print(f"No valid match for {image_name}. Removing row.")
        return None

    # return None  # No match found
    raise Exception("Didn't find an image")


def process_images_to_tfrecord(color_dir, depth_dir, commands_df, output_path):
    """
    Processes and saves images and commands to a TFRecord file.

    Args:
        color_dir (str): Directory containing color images.
        depth_dir (str): Directory containing depth images.
        commands_df (pd.DataFrame): DataFrame containing commands and image filenames.
        output_path (str): Path to save the TFRecord file.
    """
    def check_image_type(image):
        # Determine if it's an 8-bit image
        if image.dtype == "uint8" and image.min() >= 0 and image.max() <= 255:
            print("The image is 8-bit.")
        else:
            raise Exception("The image is not 8-bit.")

    with tf.io.TFRecordWriter(output_path) as writer:
        for _, row in commands_df.iterrows():
            color_image_path = os.path.join(color_dir, row['color_image_filename'])
            depth_image_path = os.path.join(depth_dir, row['depth_image_filename'])

            # Load and process images
            color_image = cv2.imread(color_image_path)
            depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

            if color_image is None or depth_image is None:
                print(f"Skipping row due to missing images: {row}")
                continue

            # Determine if it's an 8-bit image
            check_image_type(color_image)
            check_image_type(depth_image)

            # Normalize color image
            color_image = (color_image / 255.0).astype(np.float32)

            # Normalize depth image
            depth_image = depth_image.astype(np.float32)
            if len(depth_image.shape) == 2:
                depth_image = np.expand_dims(depth_image, axis=-1)  # Add channel dimension

            # Choose normalization strategy for depth
            max_depth = np.max(depth_image)
            if max_depth > 0:
                depth_image = depth_image / max_depth
            else:
                print(f"Warning: Depth image max value is zero, skipping normalization.")

            # Debugging
            print(f"Color image range: {color_image.min()} to {color_image.max()}")
            print(f"Depth image range: {depth_image.min()} to {depth_image.max()}")
            print(f"Color image shape before serialization: {color_image.shape}, dtype: {color_image.dtype}")
            print(f"Depth image shape before serialization: {depth_image.shape}, dtype: {depth_image.dtype}")

            # Serialize data
            example = serialize_example(
                color_image=color_image,
                depth_image=depth_image,
                motor_pwm=row['motor_pwm'],
                steering_pwm=row['steering_pwm']
            )
            writer.write(example)
    print(f"TFRecord saved to {output_path}")


def serialize_example(color_image, depth_image, motor_pwm, steering_pwm):
    """
    Creates a tf.train.Example message for serialization.

    Args:
        color_image (numpy.ndarray): The processed color image.
        depth_image (numpy.ndarray): The processed depth image.
        motor_pwm (int): Motor PWM for the frame.
        steering_pwm (int): Steering PWM for the frame.

    Returns:
        Serialized tf.train.Example.
    """
    feature = {
        'color_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[color_image.tobytes()])),
        'depth_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[depth_image.tobytes()])),
        'motor_pwm': tf.train.Feature(int64_list=tf.train.Int64List(value=[motor_pwm])),
        'steering_pwm': tf.train.Feature(int64_list=tf.train.Int64List(value=[steering_pwm])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def main():
    # Set directories for preprocessing
    extracted_rosbag_dir = os.path.join('data', 'rosbags_extracted')
    latest_extracted_rosbag = get_latest_directory(extracted_rosbag_dir)

    if latest_extracted_rosbag is None:
        raise FileNotFoundError("No extracted rosbag directories found.")

    color_image_dir = os.path.join(latest_extracted_rosbag, 'color_images')
    depth_image_dir = os.path.join(latest_extracted_rosbag, 'depth_images')
    commands_path = os.path.join(latest_extracted_rosbag, 'commands.csv')

    output_dir = os.path.join('data', 'processed_data', os.path.basename(latest_extracted_rosbag))
    os.makedirs(output_dir, exist_ok=True)

    # Process commands.csv and update paths
    process_commands(commands_path, output_dir, 
                     color_dir=color_image_dir,
                     depth_dir=depth_image_dir)

    updated_commands_path = os.path.join(output_dir, 'commands.csv')

    # Load the updated commands.csv
    commands_df = pd.read_csv(updated_commands_path)

    # Split the dataset into training and validation
    train_df, val_df = train_test_split(commands_df, test_size=0.2, random_state=42)

    # Output TFRecords for training and validation
    train_tfrecord = os.path.join(output_dir, "train.tfrecord")
    val_tfrecord = os.path.join(output_dir, "val.tfrecord")

    process_images_to_tfrecord(color_image_dir, depth_image_dir, train_df, train_tfrecord)
    process_images_to_tfrecord(color_image_dir, depth_image_dir, val_df, val_tfrecord)

    print(f"TFRecords created: {train_tfrecord}, {val_tfrecord}")


if __name__ == "__main__":
    main()
