import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import Model
from utils.file_utilities import get_latest_directory

# Paths
processed_data_dir = os.path.join('.', 'data', 'processed_data')
latest_processed_data = get_latest_directory(processed_data_dir)

DATA_DIR = latest_processed_data
COLOR_DIR = os.path.join(DATA_DIR, "color_images")
DEPTH_DIR = os.path.join(DATA_DIR, "depth_images")
COMMANDS_CSV = os.path.join(DATA_DIR, "commands.csv")
MODEL_DIR = os.path.join('.', 'data', 'models')

# Load commands.csv
commands_df = pd.read_csv(COMMANDS_CSV)

# Split dataset into training and validation
train_df, val_df = train_test_split(commands_df, test_size=0.2, random_state=42)

# Helper to load and preprocess images
def load_image(image_path, is_depth=False):
    if not os.path.exists(image_path):
        print(f"Error: File not found - {image_path}")
        print(f"Current working directory: {os.getcwd()}")
        return None
    else:
        print(f"File {image_path} found.")

    # Load preprocessed .npy file
    image = np.load(image_path)
    return image

# Data generator for training/validation
def data_generator(data_frame, batch_size):
    while True:
        for i in range(0, len(data_frame), batch_size):
            batch_data = data_frame.iloc[i:i + batch_size]

            # Initialize batches
            color_images = []
            depth_images = []
            linear_x = []
            angular_z = []

            for _, row in batch_data.iterrows():
                # Use full paths directly from the CSV
                color_image_path = os.path.join(COLOR_DIR, row['color_image_filename'])
                depth_image_path = os.path.join(DEPTH_DIR, row['depth_image_filename'])

                # Load images
                color_image = load_image(color_image_path)
                depth_image = load_image(depth_image_path, is_depth=True)

                # Skip invalid rows
                if color_image is None or depth_image is None:
                    print(f"Skipping invalid row: {row}")
                    continue

                color_images.append(color_image)
                depth_images.append(depth_image)
                linear_x.append(row['linear_x'])
                angular_z.append(row['angular_z'])

            # Yield only if there are valid samples
            if len(color_images) > 0 and len(depth_images) > 0:
                yield ([np.array(color_images), np.array(depth_images)],
                       [np.array(linear_x), np.array(angular_z)])
            else:
                print("No valid data in batch, skipping...")

# Create the model
def create_model():
    # Input for Color Images
    # color_input = Input(shape=(360, 640, 3), name='color_input') # NOTE possible OOM errors
    color_input = Input(shape=(180, 320, 3), name='color_input')
    color_features = layers.Conv2D(32, (3, 3), activation='relu')(color_input)
    color_features = layers.MaxPooling2D((2, 2))(color_features)
    color_features = layers.GlobalAveragePooling2D()(color_features)
    # color_features = layers.Flatten()(color_features) # Old

    # Input for Depth Images
    # depth_input = Input(shape=(360, 640, 1), name='depth_input') # same as above
    depth_input = Input(shape=(180, 320, 1), name='depth_input')
    depth_features = layers.Conv2D(32, (3, 3), activation='relu')(depth_input)
    depth_features = layers.MaxPooling2D((2, 2))(depth_features)
    depth_features = layers.GlobalAveragePooling2D()(depth_features)
    # depth_features = layers.Flatten()(depth_features) # Old

    # Combine Features
    combined = layers.Concatenate()([color_features, depth_features])
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dense(64, activation='relu')(x)

    # Outputs
    linear_output = layers.Dense(1, name='linear_x')(x)
    angular_output = layers.Dense(1, name='angular_z')(x)

    # Compile the model
    model = Model(inputs=[color_input, depth_input], outputs=[linear_output, angular_output])
    model.compile(optimizer='adam', loss='mse')
    return model

# Training setup
batch_size = 32 # Old
# batch_size = 1
epochs = 10
train_generator = data_generator(train_df, batch_size)

val_generator = data_generator(val_df, batch_size)

# Create and train the model
model = create_model()
model.summary()

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=len(val_df) // batch_size
)

# Save the model
model.save(os.path.join(MODEL_DIR, os.path.basename(DATA_DIR),"robot_model.h5"))
print(f"Model saved to {os.path.join(MODEL_DIR, 'robot_model.h5')}")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


