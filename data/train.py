import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import src.jeteja_launch.config.master_config as master_config
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import Model
from tensorflow.keras import mixed_precision
from utils.file_utilities import get_latest_directory
from utils.training_utilities import write_run_trt_optimizer_script, write_savedmodel_to_onnx_script

COLOR_WIDTH = master_config.COLOR_WIDTH
COLOR_HEIGHT = master_config.COLOR_HEIGHT
COLOR_CHANNELS = master_config.COLOR_CHANNELS
DEPTH_WIDTH = master_config.DEPTH_WIDTH
DEPTH_HEIGHT = master_config.DEPTH_HEIGHT
DEPTH_CHANNELS = master_config.DEPTH_CHANNELS
PWM_PREPROCESS_DATA_TYPE = master_config.PWM_PREPROCESS_DATA_TYPE
PWM_OUTPUT_DATA_TYPE = master_config.PWM_OUTPUT_DATA_TYPE

# Enable memory growth
physical_gpus = tf.config.list_physical_devices('GPU')
if physical_gpus:
    try:
        for gpu in physical_gpus:
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=6144)]
            )
    except RuntimeError as e:
        print(e)
else:
    raise Exception(f"No GPU detected for tensorflow {tf.__version__}.")

# Enable mixed-precision training
if PWM_OUTPUT_DATA_TYPE == np.float16:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

# Paths
processed_data_dir = os.path.join('data', 'processed_data')
latest_processed_data = get_latest_directory(processed_data_dir)

DATA_DIR = latest_processed_data
MODEL_DIR = os.path.join('data', 'models', os.path.basename(DATA_DIR))
os.makedirs(MODEL_DIR, exist_ok=True)

# TFRecord file paths
train_tfrecord = os.path.join(DATA_DIR, "train.tfrecord")
val_tfrecord = os.path.join(DATA_DIR, "val.tfrecord")
assert os.path.exists(train_tfrecord), f"Train TFRecord not found: {train_tfrecord}"
assert os.path.exists(val_tfrecord), f"Validation TFRecord not found: {val_tfrecord}"

# Parse TFRecord
def parse_tfrecord(example_proto):
    feature_description = {
        'color_image': tf.io.FixedLenFeature([], tf.string),
        'depth_image': tf.io.FixedLenFeature([], tf.string),
        'motor_pwm': tf.io.FixedLenFeature([], PWM_PREPROCESS_DATA_TYPE),  # Use float32 if normalized
        'steering_pwm': tf.io.FixedLenFeature([], PWM_PREPROCESS_DATA_TYPE),  # Use float32 if normalized
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    # Decode images from serialized bytes
    color_image = tf.io.decode_raw(parsed_features['color_image'], PWM_PREPROCESS_DATA_TYPE)
    depth_image = tf.io.decode_raw(parsed_features['depth_image'], PWM_PREPROCESS_DATA_TYPE)

    # Reshape images to their correct dimensions
    color_image = tf.reshape(color_image, (COLOR_WIDTH, COLOR_HEIGHT, COLOR_CHANNELS))
    depth_image = tf.reshape(depth_image, (DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_CHANNELS))

    # Motor and Steering PWM are already normalized to [0, 1] during serialization
    motor_pwm = parsed_features['motor_pwm']
    steering_pwm = parsed_features['steering_pwm']

    # # Debugging: Print shapes and ranges
    # tf.print("Parsed color image shape:", tf.shape(color_image))
    # tf.print("Parsed depth image shape:", tf.shape(depth_image))
    # tf.print("Parsed motor PWM:", motor_pwm)
    # tf.print("Parsed steering PWM:", steering_pwm)

    return (
        {"color_input": color_image, "depth_input": depth_image},
        {"motor_pwm": motor_pwm, "steering_pwm": steering_pwm},
    )



# Prepare datasets
def prepare_dataset(tfrecord_path, batch_size, shuffle=True):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(parse_tfrecord)

    if shuffle:
        parsed_dataset = parsed_dataset.shuffle(buffer_size=1000)

    return parsed_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Create the model
def create_model():
    # Color input and features
    color_input = Input(shape=(COLOR_WIDTH, COLOR_HEIGHT, COLOR_CHANNELS), name='color_input')
    color_features = layers.Conv2D(64, (3, 3), activation='relu')(color_input)
    color_features = layers.MaxPooling2D((2, 2))(color_features)
    color_features = layers.Conv2D(128, (3, 3), activation='relu')(color_features)
    color_features = layers.MaxPooling2D((2, 2))(color_features)
    color_features = layers.GlobalAveragePooling2D()(color_features)
    color_features = layers.Dense(512, activation='relu')(color_features)

    # Depth input and features
    depth_input = Input(shape=(DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_CHANNELS), name='depth_input')
    depth_features = layers.Conv2D(64, (3, 3), activation='relu')(depth_input)
    depth_features = layers.MaxPooling2D((2, 2))(depth_features)
    depth_features = layers.Conv2D(128, (3, 3), activation='relu')(depth_features)
    depth_features = layers.MaxPooling2D((2, 2))(depth_features)
    depth_features = layers.GlobalAveragePooling2D()(depth_features)
    depth_features = layers.Dense(512, activation='relu')(depth_features)

    # Combine features
    combined = layers.Concatenate()([color_features, depth_features])
    x = layers.Dense(256, activation='relu')(combined)
    x = layers.Dense(128, activation='relu')(x)

    # Outputs
    linear_output = layers.Dense(1, name='motor_pwm')(x)
    angular_output = layers.Dense(1, name='steering_pwm')(x)

    # Compile model
    model = Model(inputs=[color_input, depth_input], outputs=[linear_output, angular_output])
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss={'motor_pwm': 'mse', 'steering_pwm': 'mse'})
    return model


if __name__ == '__main__':
    # Training setup
    batch_size = 16 # TODO
    epochs = 15 # TODO

    train_dataset = prepare_dataset(train_tfrecord, batch_size=batch_size, shuffle=True)
    val_dataset = prepare_dataset(val_tfrecord, batch_size=batch_size, shuffle=False)

    model = create_model()
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(MODEL_DIR, "training_log.csv"),
            append=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            verbose=1,
            restore_best_weights=True
        ),
    ]

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )

    # Save the final model and plot
    model.save(os.path.join(MODEL_DIR, "final_model.keras"))

    # Export SaveModel for trt
    model.export(os.path.join(MODEL_DIR, "SavedModel"))

    # Write bash scripts
    write_savedmodel_to_onnx_script(os.path.join(MODEL_DIR, 'convert_onnx.bash'))

    write_run_trt_optimizer_script(COLOR_WIDTH, COLOR_HEIGHT, DEPTH_WIDTH, DEPTH_HEIGHT,
                    os.path.join(MODEL_DIR, 'run_trt.bash'))

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(MODEL_DIR, "training_history.csv"), index=False)

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.savefig(os.path.join(MODEL_DIR, "training_vs_validation.png"))
