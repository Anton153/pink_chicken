import os
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define character set for one-hot encoding
characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
char_to_onehot = {char: i for i, char in enumerate(characters)}

# One-hot encoding function
def one_hot_encode(char):
    one_hot = np.zeros(len(characters))  # 36-dimensional vector
    one_hot[char_to_onehot[char]] = 1
    return one_hot

# Function to crop license plate into 4 characters
def crop_license_plate(image, char_width=100, char_height=150, left_start=(45, 85), middle_width=97):
    char1 = (left_start, (left_start[0] + char_width, left_start[1] + char_height))
    char2 = ((char1[1][0], left_start[1]), (char1[1][0] + char_width, left_start[1] + char_height))
    char3 = ((char2[1][0] + middle_width, left_start[1]), (char2[1][0] + char_width + middle_width, left_start[1] + char_height))
    char4 = ((char3[1][0], left_start[1]), (char3[1][0] + char_width, left_start[1] + char_height))

    rectangles = [char1, char2, char3, char4]
    cropped_images = []
    for top_left, bottom_right in rectangles:
        cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        cropped_images.append(cv2.resize(cropped_image, (char_width, char_height)))
    return cropped_images

# Custom DataGenerator class for lazy loading
class LicensePlateDataGenerator(Sequence):
    def __init__(self, directory, batch_size, shuffle=True):
        self.directory = directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_list = [f for f in os.listdir(directory) if f.endswith('.png')]
        self.indexes = np.arange(len(self.file_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))
    
    def __getitem__(self, index):
        # Get batch indexes
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Select files for this batch
        batch_files = [self.file_list[k] for k in batch_indexes]

        X_batch = []
        Y_batch = []

        for file_name in batch_files:
            # Load and preprocess image
            image_path = os.path.join(self.directory, file_name)
            plate_image = cv2.imread(image_path)

            # Extract label from filename
            label = file_name.split('_')[1].split('.')[0]  # "EP68"

            # Crop and preprocess image
            cropped_images = crop_license_plate(plate_image)
            kernel = np.ones((3, 3), np.uint8)
            cropped_images = [cv2.erode(image, kernel, iterations=3) for image in cropped_images]

            # Add each character to the batch
            for i, char in enumerate(label):
                X_batch.append(cropped_images[i] / 255.0)  # Normalize to [0, 1]
                Y_batch.append(one_hot_encode(char))  # One-hot encode label

        return np.array(X_batch), np.array(Y_batch)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Directory containing images
directory = '/home/fizzer/ENPH353_Competition/src/pink_chicken/reading_CNN/pictures'

 #Create training and validation generators
split_ratio = 0.8  # 80% training, 20% validation
batch_size = 32
# Define the convolutional neural network model
conv_model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(150, 100, 3)),  # First Convolutional Block
    MaxPooling2D((2, 2)),
    
    Conv2D(32, (3, 3), activation='relu'),  # Second Convolutional Block
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),  # Third Convolutional Block
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dropout(0.3),  # Lower dropout to preserve more information
    Dense(512, activation='relu'),  # Dense layer with 512 neurons
    Dropout(0.5),
    Dense(128, activation='relu'),  # Dense layer with 128 neurons
    Dropout(0.5),
    Dense(36, activation='softmax')  # Output layer for 36 classes
])

conv_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Separate train and validation generators
train_gen = LicensePlateDataGenerator(directory=directory, batch_size=batch_size, shuffle=True)
val_gen = LicensePlateDataGenerator(directory=directory, batch_size=batch_size, shuffle=False)

train_files = train_gen.file_list[:int(len(train_gen.file_list) * split_ratio)]
val_files = val_gen.file_list[int(len(val_gen.file_list) * split_ratio):]

checkpoint = ModelCheckpoint(
    filepath='/home/fizzer/ENPH353_Competition/src/pink_chicken/reading_CNN/best_model.h5',
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)


# Train the model using the training generator
history_conv = conv_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=[checkpoint]
)

# Generate predictions for the validation set
Y_true_labels = []
Y_pred_labels = []

for X_batch, Y_batch in val_gen:
    # Predict batch
    Y_pred_batch = conv_model.predict(X_batch)

    # Convert predictions and labels to class indices
    Y_pred_labels.extend(np.argmax(Y_pred_batch, axis=1))
    Y_true_labels.extend(np.argmax(Y_batch, axis=1))

    # Stop after one full epoch of validation data
    if len(Y_true_labels) >= len(val_files):
        break

# Generate the confusion matrix
cm = confusion_matrix(Y_true_labels, Y_pred_labels)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=characters, yticklabels=characters)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Plot accuracy
plt.plot(history_conv.history['accuracy'], label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the model
conv_model.save('BAD_MODEL_0.h5')