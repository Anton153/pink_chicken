import string
import random
from random import randint
import cv2
import numpy as np
import os
from PIL import Image, ImageFont, ImageDraw
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


# Define paths
# path = "/content/"
pictures_path = os.path.join("/home/fizzer/ENPH353_Competition/src/pink_chicken/reading_CNN/pictures")
os.makedirs(pictures_path, exist_ok=True)

ALPHABET_PLATES = 0
NUMBER_PLATES = 0
# Initialize the data generator with augmentations
datagen = ImageDataGenerator(
    rotation_range=8,            # Random rotation within 8 degrees
    zoom_range=0.05,             # Small random zoom (5%)
    brightness_range=[0.8, 2.0], # Adjust brightness between 30% and 200%
    shear_range=2,               # Shear by 2 degrees
    width_shift_range=0.05,      # Shift width by 5%
    height_shift_range=0.05      # Shift height by 5%
)

def blue_mask(image):
    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for blue in HSV
    lower_blue = np.array([100, 120, 70])  # Lower range for blue
    upper_blue = np.array([140, 255, 255])  # Upper range for blue

    # Create a binary mask where blue pixels are 1 and others are 0
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    return blue_mask
# Function to add Gaussian noise
def add_gaussian_noise(image, mean=0, std=25):
    gauss = np.random.normal(mean, std, image.shape)  # Generate Gaussian noise
    noisy_image = image + gauss  # Add noise
    noisy_image = np.clip(noisy_image, 0, 255).astype('uint8')  # Clip to valid range
    return noisy_image

# Function to add Salt-and-Pepper noise
def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy_image = image.copy()
    total_pixels = image.size

    # Add salt (white pixels)
    num_salt = int(salt_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 255

    # Add pepper (black pixels)
    num_pepper = int(pepper_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image

# Function to add Gaussian blur
def add_gaussian_blur(image, kernel_size=(10, 50)):
    """
    Applies Gaussian blur to the image.
    Args:
        image: Input image (NumPy array).
        kernel_size: Tuple representing the size of the Gaussian kernel.
    Returns:
        Blurred image (NumPy array).
    """
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image

# Generate NUMBER_OF_PLATES with random values
for i in range(0, ALPHABET_PLATES):
    # Pick two random letters
    plate_alpha = ""
    for _ in range(0, 2):
        plate_alpha += random.choice(string.ascii_uppercase)
    
    # Pick the last letter
    plate_filler = random.choice(string.ascii_uppercase)

    # Pick two more random letters
    plate_num = ""
    for _ in range(0, 2):
        plate_num += random.choice(string.ascii_uppercase)

    # Write plate to image
    blank_plate = cv2.imread('/home/fizzer/ENPH353_Competition/src/pink_chicken/reading_CNN/blank_plate.png')

    # Convert into a PIL image (this is so we can use the monospaced fonts)
    blank_plate_pil = Image.fromarray(blank_plate)

    # Get a drawing context
    draw = ImageDraw.Draw(blank_plate_pil)
    monospace = ImageFont.truetype(
        font="/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        size=165
    )
    draw.text(
        xy=(48, 75),
        text=plate_alpha +  plate_num + plate_filler,
        fill=(255, 0, 0),  # Red text color
        font=monospace
    )

    # Convert back to OpenCV image
    blank_plate = np.array(blank_plate_pil)

    # Expand dimensions to match the generator's input format (batch size, height, width, channels)
    blank_plate = np.expand_dims(blank_plate, axis=0)

    # Apply augmentations using the generator
    augmented_iter = datagen.flow(blank_plate, batch_size=1)

    # Get the augmented image
    value = next(augmented_iter)  # Get the first (and only) batch
    augmented_image = value[0].astype('uint8')  # Convert back to uint8

    augmented_image = add_gaussian_noise(augmented_image)

    augmented_image = add_salt_and_pepper_noise(augmented_image)

    augmented_image = add_gaussian_blur(augmented_image, kernel_size=(5, 5))

    augmented_image = blue_mask(augmented_image)


    # Save the augmented plate image with noise or blur
    cv2.imwrite(os.path.join(
        pictures_path,
        f"plate_{plate_alpha}{plate_num[1]}{plate_filler}.png"
    ), augmented_image)


# Generate NUMBER_OF_PLATES with random values
for i in range(0, NUMBER_PLATES):
    # Pick two random letters
    plate_alpha = ""
    for _ in range(0, 2):
        plate_alpha += random.choice(string.ascii_uppercase)


    # Pick two more random numbers
    num = random.randint(0,9)
    plate_num = "{:02d}".format(num)

    # Write plate to image
    blank_plate = cv2.imread('/home/fizzer/ENPH353_Competition/src/pink_chicken/reading_CNN/blank_plate.png')

    # Convert into a PIL image (this is so we can use the monospaced fonts)
    blank_plate_pil = Image.fromarray(blank_plate)

    # Get a drawing context
    draw = ImageDraw.Draw(blank_plate_pil)
    monospace = ImageFont.truetype(
        font="/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        size=165
    )
    draw.text(
        xy=(48, 75),
        text=plate_alpha + " "+ plate_num,
        fill=(255, 0, 0),  # Red text color
        font=monospace
    )

    # Convert back to OpenCV image
    blank_plate = np.array(blank_plate_pil)

    # Expand dimensions to match the generator's input format (batch size, height, width, channels)
    blank_plate = np.expand_dims(blank_plate, axis=0)

    # Apply augmentations using the generator
    augmented_iter = datagen.flow(blank_plate, batch_size=1)

    # Get the augmented image
    value = next(augmented_iter)  # Get the first (and only) batch
    augmented_image = value[0].astype('uint8')  # Convert back to uint8

    augmented_image = add_gaussian_noise(augmented_image)

    augmented_image = add_salt_and_pepper_noise(augmented_image)

    augmented_image = add_gaussian_blur(augmented_image, kernel_size=(5, 5))

    augmented_image = blue_mask(augmented_image)


    # Save the augmented plate image with noise or blur
    cv2.imwrite(os.path.join(
        pictures_path,
        f"plate_{plate_alpha}{plate_num}.png"
    ), augmented_image)

# Specify the directory you want to count the files in
directory = '/home/fizzer/ENPH353_Competition/src/pink_chicken/reading_CNN/pictures'

# List all files and directories in the specified directory
all_files = os.listdir(directory)

# Filter only files (ignoring directories)
file_count = len([f for f in all_files if os.path.isfile(os.path.join(directory, f))])

print(f"Number of files in the directory: {file_count}")


# Define character set for one-hot encoding (0-9 and A-Z)
characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Match the character to its location in string characters
char_to_onehot = {char: i for i, char in enumerate(characters)}


"""
One-hot Encoding
- We create a 36 letter vector
- Match each character to a vector
"""
# Encode a specific character to a vector
def one_hot_encode(char):
    one_hot = np.zeros(36)  # 36-dimensional vector
    one_hot[char_to_onehot[char]] = 1  # Set the correct index to 1
    return one_hot

"""
Crop License Plate
- Use trial-and-error to find the correct bounds for each character
- Crop each rectangle and
"""
# Crop the license plate by each letter into 4 equal images
def crop_license_plate(image, char_width=100, char_height=150, left_start=(45, 85), middle_width=97):

    # Define the corners for each character region
    char1 = (left_start, (left_start[0] + char_width, left_start[1] + char_height))
    char2 = ((char1[1][0], left_start[1]), (char1[1][0] + char_width, left_start[1] + char_height))
    char3 = ((char2[1][0] + middle_width, left_start[1]), (char2[1][0] + char_width + middle_width, left_start[1] + char_height))
    char4 = ((char3[1][0], left_start[1]), (char3[1][0] + char_width, left_start[1] + char_height))

    # Draw 4 rectangles
    rectangles = [char1, char2, char3, char4]

    # Initiate array to store the cropped images
    cropped_images = []

    # Crop the 4 rectangles
    for top_left, bottom_right in rectangles:
        cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        cropped_image_resized = cv2.resize(cropped_image, (char_width, char_height))
        # Save into array
        cropped_images.append(cropped_image_resized)

    return cropped_images

"""
Process the Images
- We iterate through each image in the directory
- Crop each image into 4 rectangles, one character per image
- Match the name of the license plate to each cropped character
- Create training data ourselves
"""
directory = '/home/fizzer/ENPH353_Competition/src/pink_chicken/reading_CNN/pictures'
def process_license_plate_files(directory):
    X_data = []  # Will store all cropped images (input for CNN)
    Y_data = []  # Will store all one-hot encoded labels

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.png'):  # Only process PNG files
            # Get the path to the image
            image_path = os.path.join(directory, filename)

            # Extract the file name by splitting it at '_' and '.'
            label = filename.split('_')[1].split('.')[0]  # "EP68"

            # Read the image
            plate_image = cv2.imread(image_path)
            # grayscale_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            # _, binary_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)
            # inverted_image = cv2.bitwise_not(binary_image)
            original_height, original_width = plate_image.shape[:2]


            # Downscale the image by a factor of 2
            downscaled_image = cv2.resize(plate_image,
                                          (original_width // 4, original_height // 4),
                                          interpolation=cv2.INTER_AREA)

            # Upscale the image back to its original size
            plate_image = cv2.resize(downscaled_image,
                                        (original_width, original_height),
                                        interpolation=cv2.INTER_LINEAR)

            # Crop the license plate into subsections by calling previous function
            cropped_images = crop_license_plate(plate_image)

            kernel = np.ones((3, 3), np.uint8)

            cropped_images = [cv2.erode(image, kernel, iterations=3) for image in cropped_images]

            # One-hot encode the label and associate with each cropped image
            for i, char in enumerate(label):
                one_hot_label = one_hot_encode(char)  # One-hot encode each character
                X_data.append(cropped_images[i])      # Add cropped image to X_data
                Y_data.append(one_hot_label)          # Add one-hot label to Y_data

    # Convert lists to NumPy arrays for easier manipulation
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    return np.array(X_data), np.array(Y_data)

X_data, Y_data = process_license_plate_files('/home/fizzer/ENPH353_Competition/src/pink_chicken/reading_CNN/pictures')

# Print the shapes of X and Y
print(f"X shape: {X_data.shape}")  # Number of cropped images and their dimensions
print(f"Y shape: {Y_data.shape}")  # Number of one-hot encoded labels and their size

# for i in range(5):  # Display the first 5 images to check if it is good
#     cv2.imshow('i', X_data[i])
#     cv2.waitKey(3)
#     print(Y_data[i])

# Source: https://stackoverflow.com/questions/63435679
def reset_weights_tf2(model):

  # This loop iterates through each layer
  for ix, layer in enumerate(model.layers):
      # Check that there is 'kernel_initializer', 'bias_initializer'
      if (hasattr(model.layers[ix], 'kernel_initializer') and
          hasattr(model.layers[ix], 'bias_initializer')):
          # Finds weight & bias initializer
          weight_initializer = model.layers[ix].kernel_initializer
          bias_initializer = model.layers[ix].bias_initializer

          # Finds the current weights and biases
          old_weights, old_biases = model.layers[ix].get_weights()

          # Reinitializes weight and bias
          model.layers[ix].set_weights([
              weight_initializer(shape=old_weights.shape),
              bias_initializer(shape=len(old_biases))])


conv_model = models.Sequential()

early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=5,
                               restore_best_weights=True)
# Second Convolutional Block
conv_model.add(layers.Conv2D(16, (3, 3), activation='relu'))  # Reduce filters
conv_model.add(layers.MaxPooling2D((2, 2)))

# Third Convolutional Block
conv_model.add(layers.Conv2D(32, (3, 3), activation='relu'))  # Reduce filters
conv_model.add(layers.MaxPooling2D((2, 2)))

# Fourth Convolutional Block
conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # Reduce filters
conv_model.add(layers.MaxPooling2D((2, 2)))

# Flatten and Dense Layers
conv_model.add(layers.Flatten())
conv_model.add(layers.Dropout(0.3))  # Lower dropout to preserve more information
conv_model.add(layers.Dense(512, activation='relu'))  # Reduce neurons
conv_model.add(layers.Dropout(0.5))
conv_model.add(layers.Dense(128, activation='relu'))  # Reduce neurons
conv_model.add(layers.Dropout(0.5))
conv_model.add(layers.Dense(36, activation='softmax'))  # Output layer for 36 classes


# Validation split: 10% of data will be used for validation
VALIDATION_SPLIT = 0.05

LEARNING_RATE = 1e-4
conv_model.compile(loss='categorical_crossentropy',
                   optimizer=optimizers.RMSprop(learning_rate=LEARNING_RATE),
                      metrics=['acc'])


# Batch size: How many samples will be processed in one step
# Samples: From your training data set
# Step size: Samples / Batch size

X_data = np.array(X_data).astype('float32') / 255.0
print(X_data.shape)  # Verify shape

Y_data = np.array(Y_data)
history_conv = conv_model.fit(X_data, Y_data,
                              validation_split=VALIDATION_SPLIT,
                              epochs=1,
                              batch_size= 8,
                              callbacks=[early_stopping])




reset_weights_tf2(conv_model)
conv_model.summary()

# Plot accuracy
plt.plot(history_conv.history['acc'], label='Training Accuracy')
plt.plot(history_conv.history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history_conv.history['loss'], label='Training Loss')
plt.plot(history_conv.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

import seaborn as sns
from sklearn.metrics import confusion_matrix

# Generate predictions
Y_pred = conv_model.predict(X_data)

# Convert predictions and true labels to class indices
Y_pred_labels = np.argmax(Y_pred, axis=1) # My predicted data
Y_true_labels = np.argmax(Y_data, axis=1) # Actual data we are matching

# Generate the confusion matrix
cm = confusion_matrix(Y_true_labels, Y_pred_labels)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=characters, yticklabels=characters)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

conv_model.save('GOOD_MODEL.h5')