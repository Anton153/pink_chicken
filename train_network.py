import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import os
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#----------------------------------------------------------------------------------------#
def data_generator(video_filenames, processed_data_dir, batch_size=32):
    """
    Generator for training/validation data with image-only inputs.

    Args:
        video_filenames (list): List of video pickle filenames.
        processed_data_dir (str): Directory containing video pickle files.
        batch_size (int): Number of samples per batch.

    Yields:
        X_images (np.ndarray): Batch of input frames of shape (batch_size, height, width, channels).
        y (np.ndarray): Batch of target velocities (linear and angular).
    """
    while True:
        X_images, y = [], []  # Inputs and outputs
        
        # Shuffle filenames at the start of each epoch
        np.random.shuffle(video_filenames)

        for filename in video_filenames:
            file_path = os.path.join(processed_data_dir, filename)

            # Lazy load the pickle file
            with open(file_path, 'rb') as f:
                video_data = pickle.load(f)

            frames = video_data["frames"]
            commands = video_data["commands"]

            # Iterate over frames and commands
            for i in range(len(frames)):
                X_images.append(frames[i])  # Add frame to batch
                y.append(commands[i, 1:3])  # Target velocities: linear and angular

                # When batch is full, yield the data
                if len(X_images) == batch_size:
                    yield np.array(X_images), np.array(y)
                    X_images, y = [], []  # Reset for the next batch

        # Yield any remaining data (last incomplete batch)
        if X_images:
            yield np.array(X_images), np.array(y)


def calculate_steps(video_filenames, processed_data_dir, batch_size):
    total_frames = 0
    for filename in video_filenames:
        file_path = os.path.join(processed_data_dir, filename)
        with open(file_path, 'rb') as f:
            video_data = pickle.load(f)
            total_frames += len(video_data["frames"])

    return total_frames // batch_size

def weighted_mse(y_true, y_pred):
    linear_weight = 0.5  # Smaller weight for linear velocity
    angular_weight = 1.5  # Larger weight for angular velocity

    # Compute squared errors
    linear_error = tf.square(y_true[:, 0] - y_pred[:, 0])  # Linear velocity
    angular_error = tf.square(y_true[:, 1] - y_pred[:, 1])  # Angular velocity

    # Weighted sum
    return tf.reduce_mean(linear_weight * linear_error + angular_weight * angular_error)

def custom_huber_loss(y_true, y_pred, delta=1.0):
    linear_weight = 0.5
    angular_weight = 1.5

    # Errors
    linear_error = y_true[:, 0] - y_pred[:, 0]
    angular_error = y_true[:, 1] - y_pred[:, 1]

    # Huber Loss
    linear_huber = tf.where(tf.abs(linear_error) <= delta,
                            0.5 * tf.square(linear_error),
                            delta * tf.abs(linear_error) - 0.5 * tf.square(delta))
    angular_huber = tf.where(tf.abs(angular_error) <= delta,
                             0.5 * tf.square(angular_error),
                             delta * tf.abs(angular_error) - 0.5 * tf.square(delta))

    # Weighted sum
    return tf.reduce_mean(linear_weight * linear_huber + angular_weight * angular_huber)


# Define directories
processed_data_dir = '/home/fizzer/ENPH353_Competition/src/pink_chicken/pipeline_testing'

# Split video filenames into training and validation sets
video_filenames = [f for f in os.listdir(processed_data_dir) if f.endswith('.pkl')]
np.random.shuffle(video_filenames)

split_ratio = 0.8  # 80% for training, 20% for validation
split_index = int(len(video_filenames) * split_ratio)

train_filenames = video_filenames[:split_index]
val_filenames = video_filenames[split_index:]

print(f"Training videos: {len(train_filenames)}, Validation videos: {len(val_filenames)}")

# Calculate steps per epoch
train_steps = calculate_steps(train_filenames, processed_data_dir, batch_size=32)
val_steps = calculate_steps(val_filenames, processed_data_dir, batch_size=32)

# Create generators
train_generator = data_generator(train_filenames, processed_data_dir, batch_size=32)
val_generator = data_generator(val_filenames, processed_data_dir, batch_size=32)

# Define paths
model_save_path = f'/home/fizzer/ENPH353_Competition/src/pink_chicken/driving_models/model_{timestamp}.h5'

inputs = Input(shape=(128, 128, 3), name='image_input')

# Convolutional layers with reduced filter sizes
x = Conv2D(8, (3, 3), activation='relu')(inputs)  # Fewer filters
x = MaxPooling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)  # Fewer filters
x = MaxPooling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu')(x)  # Fewer filters
x = MaxPooling2D((2, 2))(x)

# Replace Flatten with Global Average Pooling
x = GlobalAveragePooling2D()(x)

# Smaller fully connected layers
linear_output = Dense(1, activation='linear', name='linear_velocity')(x)
angular_output = Dense(1, activation='linear', name='angular_velocity')(x)

# Define the model
model = Model(inputs=inputs, outputs=[linear_output, angular_output])

# Model summary
model.summary()

# Compile the model
# For Weighted MSE
model.compile(
    optimizer='adam',
    loss={'linear_velocity': 'mse', 'angular_velocity': 'mse'},
    loss_weights={'linear_velocity': 1.0, 'angular_velocity': 1.5},
    metrics={'linear_velocity': 'mae', 'angular_velocity': 'mae'}
)


checkpoint = ModelCheckpoint(
    filepath=model_save_path,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=25,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    validation_data=val_generator,
    validation_steps=val_steps,
    epochs=90,
    verbose=1,
    callbacks=[checkpoint, early_stopping]
)

# Visualize training progress
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
