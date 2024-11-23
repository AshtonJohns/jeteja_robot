
import glob
import os
import cv2
import numpy as np
import pandas as pd
from utils.file_utilities import get_latest_directory

def validate_and_preprocess(output_dir, **kwargs):
    """
    Validates and preprocesses color and depth images.

    :param kwargs:
        -   color_dir = path

        -   depth_dir = path

        -   scan_dir = path

        -   commands_path = path
    """

    color_image_dir = kwargs.get('color_dir', False)
    depth_image_dir = kwargs.get('depth_dir', False)
    scan_dir = kwargs.get('scan_dir', False)
    commands_path = kwargs.get('commands_path', False)

    os.makedirs(output_dir, exist_ok=True)

    if color_image_dir:
        processed_color_dir = os.path.join(output_dir, "color_images")
        os.makedirs(processed_color_dir, exist_ok=True)
        print(f"Processing color images from {color_image_dir}...")
        process_images(color_image_dir, processed_color_dir, expected_shape=(360, 640, 3), is_depth=False)
    else:
        processed_color_dir = False

    if depth_image_dir:
        processed_depth_dir = os.path.join(output_dir, "depth_images")
        os.makedirs(processed_depth_dir, exist_ok=True)
        print(f"Processing depth images from {depth_image_dir}...")
        process_images(depth_image_dir, processed_depth_dir, expected_shape=(360, 640, 1), is_depth=True)
    else:
        processed_depth_dir = False

    if scan_dir:
        processed_scan_dir = os.path.join(output_dir, 'scans')
        os.makedirs(processed_scan_dir, exist_ok=True) # TODO how will LiDAR data be implemented?
        print(f"Processing scans from {scan_dir}...")
        pass # TODO
    
    if commands_path:
        print(f"Processing commands.csv from {commands_path}")
        process_commands(commands_path, output_dir, 
                         color_dir = processed_color_dir,
                         depth_dir = processed_depth_dir)
        
    print("Preprocessing completed successfully.")

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

    return None  # No match found

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

def process_images(input_dir, output_dir, expected_shape, is_depth):
    """
    Processes images by validating their format and preprocessing them.

    Args:
        input_dir (str): Path to the input images directory.
        output_dir (str): Path to save processed images.
        expected_shape (tuple): Expected shape of the images (H, W, C).
        is_depth (bool): Flag indicating whether the images are depth images.
    """
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Load the image
        if is_depth:
            image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)  # Depth: unchanged for raw values
        else:
            image = cv2.imread(input_path)  # Color: load as BGR

        if image is None:
            print(f"Warning: Could not load image {input_path}")
            continue

        # Validate shape
        if is_depth:
            image = np.expand_dims(image, axis=-1)  # Add channel dimension for depth
        if image.shape != expected_shape:
            print(f"Invalid shape for {input_path}. Expected {expected_shape}, got {image.shape}.")
            continue

        # Normalize
        if is_depth:
            max_depth = np.max(image)
            if max_depth == 0:
                print(f"Invalid depth image at {input_path}, max depth is zero.")
                continue
            image = image / max_depth  # Normalize depth to [0, 1]
        else:
            image = image / 255.0  # Normalize color to [0, 1]

        # Save the processed image
        np.save(output_path.replace('.png', '.npy').replace('.jpg', '.npy'), image)
        print(f"Processed and saved: {output_path.replace('.png', '.npy').replace('.jpg', '.npy')}")

def main():

    # Set directories for preprocessing
    extracted_rosbag_dir = os.path.join('data', 'rosbags_extracted')

    latest_extracted_rosbag = get_latest_directory(extracted_rosbag_dir)

    if latest_extracted_rosbag is None:
        raise FileNotFoundError # TODO improve this to log message

    color_image_dir = os.path.join(latest_extracted_rosbag, 'color_images')
    depth_image_dir = os.path.join(latest_extracted_rosbag, 'depth_images')
    scan_dir = os.path.join(latest_extracted_rosbag, 'scans')
    commands_path = os.path.join(latest_extracted_rosbag, 'commands.csv')

    output_dir = os.path.join('data', 'processed_data', os.path.basename(latest_extracted_rosbag))

    validate_and_preprocess(output_dir, 
                            color_dir=color_image_dir,
                            depth_dir=depth_image_dir,
                            scan_dir = scan_dir,
                            commands_path = commands_path)

if __name__ == '__main__':
    main()
