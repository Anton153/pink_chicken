import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
import os
import cv2
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Lambda
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


#----------------------------------------------------------------------------------------#
#                               HELPER FUNCTIONS
#----------------------------------------------------------------------------------------#
def one_hot_encode(value):
    """Encodes -1, 0, 1 into one-hot format."""
    mapping = {-1: [1, 0, 0], 0: [0, 1, 0], 1: [0, 0, 1]}
    return mapping[value]

def one_hot_encode_smoothed(value, num_classes=3, smoothing=0.1):
    """Encodes -1, 0, 1 into smoothed one-hot format."""
    if value not in [-1, 0, 1]:
        raise ValueError(f"Unexpected value for one-hot encoding: {value}")
    one_hot = np.zeros(num_classes)
    one_hot[value + 1] = 1  # Adjust index: -1->0, 0->1, 1->2
    return one_hot * (1 - smoothing) + smoothing / num_classes

def data_generator(video_filenames, processed_data_dir, batch_size=32, smoothing=0.1, linear_class_weight=None, angular_class_weight=None):
    while True:
        X_images, y_linear, y_angular, sw_linear, sw_angular = [], [], [], [], []
        for filename in video_filenames:
            file_path = os.path.join(processed_data_dir, filename)
            with open(file_path, 'rb') as f:
                video_data = pickle.load(f)

            frames = video_data["frames"]
            commands = video_data["commands"]

            for i in range(len(frames)):
                X_images.append(frames[i])  # Add frame to batch
                linear_command = int(round(commands[i, 1]))
                angular_command = int(round(commands[i, 2]))

                y_linear.append(one_hot_encode_smoothed(linear_command, smoothing=smoothing))
                y_angular.append(one_hot_encode_smoothed(angular_command, smoothing=smoothing))

                # Add sample weights
                sw_linear.append(linear_class_weight[linear_command + 1])
                sw_angular.append(angular_class_weight[angular_command + 1])

                if len(X_images) == batch_size:
                    yield (
                        np.array(X_images),
                        {'linear_velocity': np.array(y_linear), 'angular_velocity': np.array(y_angular)},
                        {'linear_velocity': np.array(sw_linear), 'angular_velocity': np.array(sw_angular)}
                    )
                    X_images, y_linear, y_angular, sw_linear, sw_angular = [], [], [], [], []

        if X_images:
            yield (
                np.array(X_images),
                {'linear_velocity': np.array(y_linear), 'angular_velocity': np.array(y_angular)},
                {'linear_velocity': np.array(sw_linear), 'angular_velocity': np.array(sw_angular)}
            )


def calculate_class_weights(processed_data_dir, video_filenames):
    """Calculates class weights based on the training data."""
    linear_classes = []
    angular_classes = []

    for filename in video_filenames:
        file_path = os.path.join(processed_data_dir, filename)
        with open(file_path, 'rb') as f:
            video_data = pickle.load(f)

        commands = video_data["commands"]
        if isinstance(commands, pd.DataFrame):
            commands = commands.to_numpy()

        linear_classes += [int(round(c[1])) + 1 for c in commands]
        angular_classes += [int(round(c[2])) + 1 for c in commands]

    linear_weights = compute_class_weight('balanced', classes=np.arange(3), y=linear_classes)
    angular_weights = compute_class_weight('balanced', classes=np.arange(3), y=angular_classes)

    return linear_weights, angular_weights

def calculate_steps(video_filenames, processed_data_dir, batch_size):
    """Calculates steps per epoch based on the number of frames."""
    total_frames = 0
    for filename in video_filenames:
        file_path = os.path.join(processed_data_dir, filename)
        with open(file_path, 'rb') as f:
            video_data = pickle.load(f)
            total_frames += len(video_data["frames"])

    return total_frames // batch_size

def calibrated_softmax(logits, bias=1.5):
    """Applies calibrated softmax to logits."""
    boosted_logits = tf.concat([
        logits[:, 0:1] * bias,  # Boost "turn left"
        logits[:, 1:2],        # Keep "go straight" as is
        logits[:, 2:3] * bias  # Boost "turn right"
    ], axis=1)
    return tf.nn.softmax(boosted_logits)


#----------------------------------------------------------------------------------------#
#                               MODEL SETUP
#----------------------------------------------------------------------------------------#

# Paths and file setup
processed_data_dir = '/home/fizzer/ENPH353_Competition/src/pink_chicken/pipeline_testing'
video_filenames = [f for f in os.listdir(processed_data_dir) if f.endswith('.pkl')]
np.random.shuffle(video_filenames)

split_ratio = 0.90
split_index = int(len(video_filenames) * split_ratio)
train_filenames = video_filenames[:split_index]
val_filenames = video_filenames[split_index:]

train_steps = calculate_steps(train_filenames, processed_data_dir, batch_size=32)
val_steps = calculate_steps(val_filenames, processed_data_dir, batch_size=32)


# Calculate class weights
linear_weights, angular_weights = calculate_class_weights(processed_data_dir, train_filenames)

linear_class_weight = {0: linear_weights[0], 1: linear_weights[1], 2: linear_weights[2]}
angular_class_weight = {0: angular_weights[0], 1: angular_weights[1], 2: angular_weights[2]}

print("Linear Class Weights:", linear_class_weight)
print("Angular Class Weights:", angular_class_weight)


#data generators
train_generator = data_generator(
    train_filenames,
    processed_data_dir,
    batch_size=32,
    linear_class_weight=linear_class_weight,
    angular_class_weight=angular_class_weight
)

val_generator = data_generator(
    val_filenames,
    processed_data_dir,
    batch_size=32,
    linear_class_weight=linear_class_weight,
    angular_class_weight=angular_class_weight
)


# Class weights
linear_weights, angular_weights = calculate_class_weights(processed_data_dir, train_filenames)
linear_class_weight = {0: linear_weights[0], 1: linear_weights[1], 2: linear_weights[2]}
angular_class_weight = {0: angular_weights[0], 1: angular_weights[1], 2: angular_weights[2]}

# Model architecture

inputs = Input(shape=(128, 128, 3), name='image_input')
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
# x = BatchNormalization()(x)
# x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.2)(x)

# x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.3)(x)

# x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.4)(x)

# x = GlobalAveragePooling2D()(x)
# x = Dense(256, activation='relu')(x)
# x = Dropout(0.4)(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.3)(x)

x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)


# Outputs
linear_logits = Dense(3, activation=None, name='linear_velocity_logits')(x)
angular_logits = Dense(3, activation=None, name='angular_velocity_logits')(x)

linear_output = Lambda(lambda logits: calibrated_softmax(logits), name='linear_velocity')(linear_logits)
angular_output = Lambda(lambda logits: calibrated_softmax(logits), name='angular_velocity')(angular_logits)

model = Model(inputs=inputs, outputs=[linear_output, angular_output])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        'linear_velocity': 'categorical_crossentropy',
        'angular_velocity': 'categorical_crossentropy'
    },
    metrics={
        'linear_velocity': 'accuracy',
        'angular_velocity': 'accuracy'
    }
)
model.summary()

# Callbacks
checkpoint = ModelCheckpoint(filepath=f'/home/fizzer/ENPH353_Competition/src/pink_chicken/driving_models/model_{timestamp}.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    validation_data=val_generator,
    validation_steps=val_steps,
    epochs=200,
    verbose=1,
    callbacks=[checkpoint, early_stopping]
)


# Plot accuracies
plt.plot(history.history['linear_velocity_accuracy'], label='Linear Velocity Accuracy')
plt.plot(history.history['val_linear_velocity_accuracy'], label='Validation Linear Velocity Accuracy')
plt.plot(history.history['angular_velocity_accuracy'], label='Angular Velocity Accuracy')
plt.plot(history.history['val_angular_velocity_accuracy'], label='Validation Angular Velocity Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()
