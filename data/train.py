import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import Model

# Paths
DATA_DIR = "training/data"
COLOR_DIR = os.path.join(DATA_DIR, "color")
DEPTH_DIR = os.path.join(DATA_DIR, "depth")
COMMANDS_CSV = os.path.join(DATA_DIR, "commands.csv")
MODEL_DIR = "training/model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load commands.csv
commands_df = pd.read_csv(COMMANDS_CSV)

# Split dataset into training and validation
train_df, val_df = train_test_split(commands_df, test_size=0.2, random_state=42)

# Helper to load and preprocess images
def load_image(image_path, is_depth=False):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED if is_depth else cv2.IMREAD_COLOR)
    if is_depth:
        image = np.expand_dims(image / 255.0, axis=-1)  # Normalize and add channel dimension
    else:
        image = image / 255.0  # Normalize RGB image
    return image

# Data generator for training/validation
def data_generator(data_frame, color_dir, depth_dir, batch_size):
    while True:
        for i in range(0, len(data_frame), batch_size):
            batch_data = data_frame.iloc[i:i + batch_size]

            # Initialize batches
            color_images = []
            depth_images = []
            linear_x = []
            angular_z = []

            for _, row in batch_data.iterrows():
                # Load images
                color_image_path = os.path.join(color_dir, row['image_filename'])
                depth_image_path = os.path.join(depth_dir, row['image_filename'].replace('.jpg', '.png'))

                color_images.append(load_image(color_image_path))
                depth_images.append(load_image(depth_image_path, is_depth=True))
                linear_x.append(row['linear_x'])
                angular_z.append(row['angular_z'])

            # Yield batches
            yield ([np.array(color_images), np.array(depth_images)],
                   [np.array(linear_x), np.array(angular_z)])

# Create the model
def create_model():
    # Input for Color Images
    color_input = Input(shape=(360, 640, 3), name='color_input')
    color_features = layers.Conv2D(32, (3, 3), activation='relu')(color_input)
    color_features = layers.MaxPooling2D((2, 2))(color_features)
    color_features = layers.Flatten()(color_features)

    # Input for Depth Images
    depth_input = Input(shape=(360, 640, 1), name='depth_input')
    depth_features = layers.Conv2D(32, (3, 3), activation='relu')(depth_input)
    depth_features = layers.MaxPooling2D((2, 2))(depth_features)
    depth_features = layers.Flatten()(depth_features)

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
batch_size = 32
epochs = 10
train_generator = data_generator(train_df, COLOR_DIR, DEPTH_DIR, batch_size)
val_generator = data_generator(val_df, COLOR_DIR, DEPTH_DIR, batch_size)

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
model.save(os.path.join(MODEL_DIR, "robot_model.h5"))
print(f"Model saved to {os.path.join(MODEL_DIR, 'robot_model.h5')}")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


