import path
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import src.jeteja_launch.config.master_config as master_config
from tensorflow.keras import mixed_precision, layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Multiply, Dropout, GlobalAveragePooling2D
from utils.file_utilities import get_latest_directory
from utils.training_utilities import write_run_trt_optimizer_script_color_depth, write_run_trt_optimizer_script_color, write_savedmodel_to_onnx_script

TRAIN_COLOR = master_config.TRAIN_COLOR
COLOR_WIDTH = master_config.COLOR_WIDTH
COLOR_HEIGHT = master_config.COLOR_HEIGHT
COLOR_CHANNELS = master_config.COLOR_CHANNELS
TRAIN_DEPTH = master_config.TRAIN_DEPTH
DEPTH_WIDTH = master_config.DEPTH_WIDTH
DEPTH_HEIGHT = master_config.DEPTH_HEIGHT
DEPTH_CHANNELS = master_config.DEPTH_CHANNELS
PWM_PREPROCESS_DATA_TYPE = master_config.PWM_PREPROCESS_DATA_TYPE
PWM_OUTPUT_DATA_TYPE = master_config.PWM_OUTPUT_DATA_TYPE

# # debug
# print(COLOR_WIDTH)
# exit()

# Enable memory growth for each GPU
physical_gpus = tf.config.list_physical_devices('GPU')

if physical_gpus:
    try:
        # Enable memory growth dynamically for each GPU
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error enabling memory growth: {e}")
else:
    raise Exception(f"No GPU detected for TensorFlow {tf.__version__}.")

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

# Get feature description
def get_feature_description():
    if TRAIN_COLOR and TRAIN_DEPTH:
        feature_description = {
            'color_image': tf.io.FixedLenFeature([], tf.string),
            'depth_image': tf.io.FixedLenFeature([], tf.string),
            'motor_pwm': tf.io.FixedLenFeature([], PWM_PREPROCESS_DATA_TYPE),  
            'steering_pwm': tf.io.FixedLenFeature([], PWM_PREPROCESS_DATA_TYPE),
        }
    elif TRAIN_COLOR:
        feature_description = {
            'color_image': tf.io.FixedLenFeature([], tf.string),
            'motor_pwm': tf.io.FixedLenFeature([], PWM_PREPROCESS_DATA_TYPE),  
            'steering_pwm': tf.io.FixedLenFeature([], PWM_PREPROCESS_DATA_TYPE),
        }
    return feature_description


def get_decoded_reshaped_image(parsed_features, image_height, image_width, image_channels):
    # Decode images from serialized bytes
    decoded_image = tf.io.decode_raw(parsed_features['color_image'], PWM_PREPROCESS_DATA_TYPE)
    reshaped_image = tf.reshape(decoded_image, (image_height, image_width, image_channels))  # Reshape images to their correct dimensions
    return reshaped_image

# Parse TFRecord
def parse_tfrecord(example_proto):

    feature_description = get_feature_description()
    
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    # Motor and Steering PWM are already normalized to [0, 1] during serialization
    motor_pwm = parsed_features['motor_pwm']
    steering_pwm = parsed_features['steering_pwm']


    if TRAIN_COLOR and TRAIN_DEPTH:
        color_image = get_decoded_reshaped_image(parsed_features, COLOR_HEIGHT, COLOR_WIDTH, COLOR_CHANNELS)
        depth_image = get_decoded_reshaped_image(parsed_features, DEPTH_HEIGHT, DEPTH_WIDTH, DEPTH_CHANNELS)
        parsed_tfrecord = (
            {"color_input": color_image, "depth_input": depth_image},
            {"motor_pwm": motor_pwm, "steering_pwm": steering_pwm},
        )
    elif TRAIN_COLOR:
        color_image = get_decoded_reshaped_image(parsed_features, COLOR_HEIGHT, COLOR_WIDTH, COLOR_CHANNELS)
        parsed_tfrecord = (
            {"color_input": color_image},
            {"motor_pwm": motor_pwm, "steering_pwm": steering_pwm},
        )

    # # Debugging: Print shapes and ranges
    # tf.print("Parsed color image shape:", tf.shape(color_image))
    # tf.print("Parsed depth image shape:", tf.shape(depth_image))
    # tf.print("Parsed motor PWM:", motor_pwm)
    # tf.print("Parsed steering PWM:", steering_pwm)

    return parsed_tfrecord


# Prepare datasets
def prepare_dataset(tfrecord_path, batch_size, shuffle=True):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(parse_tfrecord)

    if shuffle:
        parsed_dataset = parsed_dataset.shuffle(buffer_size=1000)

    return parsed_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def se_block(input_tensor, reduction_ratio=16):
    channels = input_tensor.shape[-1]  # Number of channels in the input tensor

    # Squeeze operation: Global average pooling
    se = layers.GlobalAveragePooling2D()(input_tensor)

    # Fully connected layers for excitation
    se = layers.Dense(channels // reduction_ratio, activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)

    # Reshape to match input dimensions
    se = layers.Reshape((1, 1, channels))(se)

    # Scale input tensor by the excitation weights
    return layers.Multiply()([input_tensor, se])


def spatial_attention(input_tensor):
    # Compute average and max pooling along the channel axis
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_tensor)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_tensor)

    # Concatenate the pooling outputs
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])

    # Apply a convolutional layer to compute attention map
    attention = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)

    # Multiply the attention map with the input tensor
    return layers.Multiply()([input_tensor, attention])


def cbam_attention(color_features, depth_features):
    combined_features = Concatenate()([color_features, depth_features])
    # Apply SE block for channel-wise attention
    combined_features = se_block(combined_features)
    # Apply spatial attention
    combined_features = spatial_attention(combined_features)
    return combined_features


def dynamic_weights(color_features, depth_features):
    combined_features = Concatenate()([color_features, depth_features])
    gate = Dense(2, activation='softmax')(combined_features)  # Two weights

    # Use Lambda layer to perform tf.split operation
    def split_gate(gate_tensor):
        return tf.split(gate_tensor, 2, axis=-1)
    
    color_weight, depth_weight = layers.Lambda(split_gate)(gate)
    return layers.Multiply()([color_features, color_weight]), layers.Multiply()([depth_features, depth_weight])

# def build_conv2d_backbone(input_tensor, neurons, use_flatten, use_se_block, use_spatial_attention):
#     # Custom Conv2D backbone
#     x = layers.Conv2D(64, (3, 3), activation='relu')(input_tensor)
#     x = layers.MaxPooling2D((2, 2))(x)
#     x = layers.Conv2D(128, (3, 3), activation='relu')(x)
#     x = layers.MaxPooling2D((2, 2))(x)

#     # Optional attention mechanisms
#     if use_se_block:
#         x = se_block(x)
#     if use_spatial_attention:
#         x = spatial_attention(x)

#     # Reduce spatial dimensions
#     x = Flatten()(x) if use_flatten else GlobalAveragePooling2D()(x)

#     # Add dense layers
#     x = Dense(neurons, activation='relu')(x)
#     x = Dropout(0.3)(x)
#     return x

def build_conv2d_backbone(input_tensor, neurons, use_flatten, use_se_block, use_spatial_attention):
    # First convolutional block
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), activation='relu')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Second convolutional block
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Optional attention mechanisms
    if use_se_block:
        x = se_block(x)
    if use_spatial_attention:
        x = spatial_attention(x)

    # Reduce spatial dimensions
    x = Flatten()(x) if use_flatten else GlobalAveragePooling2D()(x)

    # Dense layers for feature representation
    x = Dense(neurons, activation='relu')(x)
    x = Dropout(0.3)(x)
    return x


def build_efficientnet_backbone(input_tensor, input_shape, neurons, use_se_block, use_spatial_attention):
    # EfficientNet backbone
    efficientnet = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    x = efficientnet(input_tensor)

    # Optional attention mechanisms
    if use_se_block:
        x = se_block(x)
    if use_spatial_attention:
        x = spatial_attention(x)

    # Reduce spatial dimensions
    x = GlobalAveragePooling2D()(x)
    x = Dense(neurons, activation='relu')(x)
    x = Dropout(0.3)(x)
    return x

# Create the model
def create_model(config):
    """
    Model Factory to build a customizable model based on the configuration.
    """
    inputs = []
    features = []

    # Build color input and backbone
    if config['TRAIN_COLOR']:
        color_input = Input(shape=(config['COLOR_HEIGHT'], config['COLOR_WIDTH'], config['COLOR_CHANNELS']), name='color_input')
        inputs.append(color_input)
        
        if config['use_efficientnet']:
            color_features = build_efficientnet_backbone(
                color_input, 
                input_shape=(config['COLOR_HEIGHT'], config['COLOR_WIDTH'], config['COLOR_CHANNELS']),
                neurons=config['neurons']['color_dense'],
                use_se_block=config['use_se_block'],
                use_spatial_attention=config['use_spatial_attention']
            )
        else:
            color_features = build_conv2d_backbone(
                color_input,
                neurons=config['neurons']['color_dense'],
                use_flatten=config['use_flatten'],
                use_se_block=config['use_se_block'],
                use_spatial_attention=config['use_spatial_attention']
            )
        features.append(color_features)

    # Build depth input and backbone
    if config['TRAIN_DEPTH']:
        depth_input = Input(shape=(config['DEPTH_HEIGHT'], config['DEPTH_WIDTH'], config['DEPTH_CHANNELS']), name='depth_input')
        inputs.append(depth_input)
        
        if config['use_efficientnet']:
            depth_features = build_efficientnet_backbone(
                depth_input,
                input_shape=(config['DEPTH_HEIGHT'], config['DEPTH_WIDTH'], config['DEPTH_CHANNELS']),
                neurons=config['neurons']['depth_dense'],
                use_se_block=config['use_se_block'],
                use_spatial_attention=config['use_spatial_attention']
            )
        else:
            depth_features = build_conv2d_backbone(
                depth_input,
                neurons=config['neurons']['depth_dense'],
                use_flatten=config['use_flatten'],
                use_se_block=config['use_se_block'],
                use_spatial_attention=config['use_spatial_attention']
            )
        features.append(depth_features)

    # Combine features if both color and depth are used
    if len(features) > 1:
        if config['use_combined_attention']:
            combined_features = cbam_attention(features[0], features[1])
        elif config['use_dynamic_weights']:
            combined_features = Concatenate()(dynamic_weights(features[0], features[1]))
        else:
            combined_features = Concatenate()(features)
    else:
        combined_features = features[0]

    # Fully connected layers
    x = Dense(config['neurons']['combined_dense1'], activation='relu')(combined_features)
    x = Dropout(0.4)(x)
    x = Dense(config['neurons']['combined_dense2'], activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(config['neurons']['combined_dense3'], activation='relu')(x)

    # Output layers
    motor_pwm = Dense(1, name='motor_pwm')(x)
    steering_pwm = Dense(1, name='steering_pwm')(x)

    # Compile the model
    model = Model(inputs=inputs, outputs=[motor_pwm, steering_pwm])
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss={'motor_pwm': 'mse', 'steering_pwm': 'mse'})
    return model


def main():
    config = {
        'TRAIN_COLOR': TRAIN_COLOR,
        'COLOR_HEIGHT': COLOR_HEIGHT,
        'COLOR_WIDTH': COLOR_WIDTH,
        'COLOR_CHANNELS': COLOR_CHANNELS,
        'TRAIN_DEPTH': TRAIN_DEPTH,
        'DEPTH_HEIGHT': DEPTH_HEIGHT,
        'DEPTH_WIDTH': DEPTH_WIDTH,
        'DEPTH_CHANNELS': DEPTH_CHANNELS,
        'use_efficientnet': False,
        'use_flatten': False,
        'use_se_block': False,
        'use_spatial_attention': False,
        'use_combined_attention': False,
        'use_dynamic_weights': False,
        'neurons': {
            'color_dense': 128,
            'depth_dense': 64,
            'combined_dense1': 256,
            'combined_dense2': 128,
            'combined_dense3': 64,
        },
        'learning_rate': 0.001,
    }


    # Training setup
    batch_size = 24 # TODO
    epochs = 100 # TODO

    train_dataset = prepare_dataset(train_tfrecord, batch_size=batch_size, shuffle=True)
    val_dataset = prepare_dataset(val_tfrecord, batch_size=batch_size, shuffle=False)

    # Debug
    for data, labels in train_dataset.take(1):
        print("Data keys:", data.keys())  # Expect: dict_keys(['color_input'])
        print("Color input shape:", data['color_input'].shape)  # Expect: (batch_size, 360, 640, 3)
    for batch_data, batch_labels in train_dataset.take(1):
        print("Batch data:", batch_data)  # Should print: {'color_input': <Tensor>}
        print("Batch labels:", batch_labels)  # Should print: {'motor_pwm': <Tensor>, 'steering_pwm': <Tensor>}
    for data, labels in train_dataset.take(1):
        print("Final Batch Data Keys:", data.keys())  # Expect: dict_keys(['color_input'])
        print("Final Color Input Shape:", data['color_input'].shape)  # Expect: (24, 360, 640, 3)

    model = create_model(config)

    # Debug
    for data, labels in train_dataset.take(1):
        output = model.predict(data)  # Test single batch
        print("Prediction shape:", output)

    print(model.inputs)

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

    if TRAIN_COLOR and TRAIN_DEPTH:
        write_run_trt_optimizer_script_color_depth(COLOR_WIDTH, COLOR_HEIGHT, DEPTH_WIDTH, DEPTH_HEIGHT, 
                                                   os.path.join(MODEL_DIR, 'run_trt.bash'))
    elif TRAIN_COLOR:
        write_run_trt_optimizer_script_color(COLOR_WIDTH, COLOR_HEIGHT,
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


if __name__ == '__main__':
    main()



####### TRIED MODELS #######
# def create_model():
#     # Color input and features
#     color_input = Input(shape=(COLOR_WIDTH, COLOR_HEIGHT, COLOR_CHANNELS), name='color_input')
#     efficientnet_color = EfficientNetB0(include_top=False, weights=None, input_shape=(COLOR_WIDTH, COLOR_HEIGHT, COLOR_CHANNELS), name="efficientnetb0_color")
#     # efficientnet_color = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(COLOR_WIDTH, COLOR_HEIGHT, COLOR_CHANNELS), name="efficientnetb0_color")
#     color_features = efficientnet_color(color_input)
#     color_features = Flatten()(color_features)  # Use Flatten instead of pooling
#     color_features = Dense(128, activation='relu')(color_features)  # Further decreased neurons for color input
#     color_features = Dropout(0.3)(color_features)

#     # Depth input and features
#     depth_input = Input(shape=(DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_CHANNELS), name='depth_input')
#     efficientnet_depth = EfficientNetB0(include_top=False, weights=None, input_shape=(DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_CHANNELS), name="efficientnetb0_depth")
#     depth_features = efficientnet_depth(depth_input)
#     depth_features = Flatten()(depth_features)  # Use Flatten instead of pooling
#     depth_features = Dense(24, activation='relu')(depth_features)  # Further decreased neurons for depth input
#     depth_features = Dropout(0.3)(depth_features)

#     # Trainable weights for dynamic weighting
#     color_weight = Dense(1, activation='sigmoid', name='color_weight')(color_features)
#     depth_weight = Dense(1, activation='sigmoid', name='depth_weight')(depth_features)

#     # Apply weights
#     weighted_color = Multiply()([color_features, color_weight])
#     weighted_depth = Multiply()([depth_features, depth_weight])

#     # Combine weighted features
#     combined = Concatenate()([weighted_color, weighted_depth])
#     x = Dense(128, activation='relu')(combined)  # Further decreased neurons for combined features
#     x = Dropout(0.4)(x)
#     x = Dense(24, activation='relu')(x)  # Further decreased neurons in this layer
#     x = Dropout(0.3)(x)
#     x = Dense(12, activation='relu')(x)  # Further decreased neurons in this layer

#     # Outputs
#     linear_output = Dense(1, name='motor_pwm')(x)
#     angular_output = Dense(1, name='steering_pwm')(x)

#     # Compile model
#     model = Model(inputs=[color_input, depth_input], outputs=[linear_output, angular_output])
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#     model.compile(optimizer=optimizer, loss={'motor_pwm': 'mse', 'steering_pwm': 'mse'})
#     return model


# def create_model():
#     # Color input and features
#     color_input = Input(shape=(COLOR_WIDTH, COLOR_HEIGHT, COLOR_CHANNELS), name='color_input')
#     efficientnet_color = EfficientNetB0(include_top=False, weights=None, input_shape=(COLOR_WIDTH, COLOR_HEIGHT, COLOR_CHANNELS), name="efficientnetb0_color")
#     # efficientnet_color = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(COLOR_WIDTH, COLOR_HEIGHT, COLOR_CHANNELS), name="efficientnetb0_color")
#     color_features = efficientnet_color(color_input)
#     color_features = Flatten()(color_features)  # Use Flatten instead of pooling
#     color_features = Dense(1024, activation='relu')(color_features)  # Increased neurons for high-resolution
#     color_features = Dropout(0.3)(color_features)

#     # Depth input and features
#     depth_input = Input(shape=(DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_CHANNELS), name='depth_input')
#     efficientnet_depth = EfficientNetB0(include_top=False, weights=None, input_shape=(DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_CHANNELS), name="efficientnetb0_depth")
#     depth_features = efficientnet_depth(depth_input)
#     depth_features = Flatten()(depth_features)  # Use Flatten instead of pooling
#     depth_features = Dense(512, activation='relu')(depth_features)  # Fewer neurons for depth
#     depth_features = Dropout(0.3)(depth_features)

#     # Trainable weights for dynamic weighting
#     color_weight = Dense(1, activation='sigmoid', name='color_weight')(color_features)
#     depth_weight = Dense(1, activation='sigmoid', name='depth_weight')(depth_features)

#     # Apply weights
#     weighted_color = Multiply()([color_features, color_weight])
#     weighted_depth = Multiply()([depth_features, depth_weight])

#     # Combine weighted features
#     combined = Concatenate()([weighted_color, weighted_depth])
#     x = Dense(1024, activation='relu')(combined)
#     x = Dropout(0.4)(x)
#     x = Dense(512, activation='relu')(x)
#     x = Dropout(0.3)(x)
#     x = Dense(256, activation='relu')(x)

#     # Outputs
#     linear_output = Dense(1, name='motor_pwm')(x)
#     angular_output = Dense(1, name='steering_pwm')(x)

#     # Compile model
#     model = Model(inputs=[color_input, depth_input], outputs=[linear_output, angular_output])
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#     model.compile(optimizer=optimizer, loss={'motor_pwm': 'mse', 'steering_pwm': 'mse'})
#     return model

# def create_model():
#     # Color input and features
#     color_input = Input(shape=(COLOR_HEIGHT, COLOR_WIDTH, COLOR_CHANNELS), name='color_input')
#     efficientnet_color = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(COLOR_HEIGHT, COLOR_WIDTH, COLOR_CHANNELS))
#     color_features = efficientnet_color(color_input)
#     color_features = Flatten()(color_features)  # Use Flatten instead of pooling
#     color_features = Dense(1024, activation='relu')(color_features)  # Increased neurons for high-resolution
#     color_features = Dropout(0.3)(color_features)

#     # Depth input and features
#     depth_input = Input(shape=(DEPTH_HEIGHT, DEPTH_WIDTH, DEPTH_CHANNELS), name='depth_input')
#     efficientnet_depth = EfficientNetB0(include_top=False, weights=None, input_shape=(DEPTH_HEIGHT, DEPTH_WIDTH, DEPTH_CHANNELS))
#     depth_features = efficientnet_depth(depth_input)
#     depth_features = Flatten()(depth_features)  # Use Flatten instead of pooling
#     depth_features = Dense(512, activation='relu')(depth_features)  # Fewer neurons for depth
#     depth_features = Dropout(0.3)(depth_features)

#     # Trainable weights for dynamic weighting
#     color_weight = Dense(1, activation='sigmoid', name='color_weight')(color_features)
#     depth_weight = Dense(1, activation='sigmoid', name='depth_weight')(depth_features)

#     # Apply weights
#     weighted_color = Multiply()([color_features, color_weight])
#     weighted_depth = Multiply()([depth_features, depth_weight])

#     # Combine weighted features
#     combined = Concatenate()([weighted_color, weighted_depth])
#     x = Dense(1024, activation='relu')(combined)
#     x = Dropout(0.4)(x)
#     x = Dense(512, activation='relu')(x)
#     x = Dropout(0.3)(x)
#     x = Dense(256, activation='relu')(x)

#     # Outputs
#     linear_output = Dense(1, name='motor_pwm')(x)
#     angular_output = Dense(1, name='steering_pwm')(x)

#     # Compile model
#     model = Model(inputs=[color_input, depth_input], outputs=[linear_output, angular_output])
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#     model.compile(optimizer=optimizer, loss={'motor_pwm': 'mse', 'steering_pwm': 'mse'})
#     return model

# def create_model():
#     # Color input and features
#     color_input = Input(shape=(COLOR_WIDTH, COLOR_HEIGHT, COLOR_CHANNELS), name='color_input')
#     color_features = layers.Conv2D(64, (3, 3), activation='relu')(color_input)
#     color_features = layers.MaxPooling2D((2, 2))(color_features)
#     color_features = layers.Conv2D(128, (3, 3), activation='relu')(color_features)
#     color_features = layers.MaxPooling2D((2, 2))(color_features)
#     color_features = layers.GlobalAveragePooling2D()(color_features)
#     color_features = layers.Dense(512, activation='relu')(color_features)

#     # Depth input and features
#     depth_input = Input(shape=(DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_CHANNELS), name='depth_input')
#     depth_features = layers.Conv2D(64, (3, 3), activation='relu')(depth_input)
#     depth_features = layers.MaxPooling2D((2, 2))(depth_features)
#     depth_features = layers.Conv2D(128, (3, 3), activation='relu')(depth_features)
#     depth_features = layers.MaxPooling2D((2, 2))(depth_features)
#     depth_features = layers.GlobalAveragePooling2D()(depth_features)
#     depth_features = layers.Dense(512, activation='relu')(depth_features)

#     # Combine features
#     combined = layers.Concatenate()([color_features, depth_features])
#     x = layers.Dense(256, activation='relu')(combined)
#     x = layers.Dense(128, activation='relu')(x)

#     # Outputs
#     linear_output = layers.Dense(1, name='motor_pwm')(x)
#     angular_output = layers.Dense(1, name='steering_pwm')(x)

#     # Compile model
#     model = Model(inputs=[color_input, depth_input], outputs=[linear_output, angular_output])
#     optimizer = tf.keras.optimizers.Adam()
#     model.compile(optimizer=optimizer, loss={'motor_pwm': 'mse', 'steering_pwm': 'mse'})
#     return model
