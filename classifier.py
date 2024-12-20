
import tensorflow as tf
import sys
import argparse
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image


# Load and preprocess the image
def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)  # creates a 3D numpy array
    img_tensor = np.expand_dims(img_tensor, axis=0)  # expands dimensions for batch size 
    img_tensor /= 255.  # model expects values in the range [0, 1]
    
    return img_tensor

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Classify a given image.')
parser.add_argument('image_path', type=str, help='Path to the image file.')
parser.add_argument('-p', '--percentage', action='store_true', 
                    help='Return prediction percentage instead of class.')

args = parser.parse_args()

img_tensor = preprocess_image(args.image_path)


# Define paths and other parameters
train_dir = './train'
validation_dir = './validation'

# Instantiate an ImageDataGenerator for training with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

# Generate batches of tensor image data for training and validation
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# Define a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=1e-4),
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator
)

# Save the model
# model.save('cardordoc.h5')  # creates a HDF5 file 'my_model.h5'


# Predict the class of the image
prediction = model.predict(img_tensor)

if args.percentage:
    print(f"The prediction percentage for class 1 is {prediction[0][0]*100}%")
else:
    # If prediction is > 0.5 it is one class, otherwise it is the other class
    if prediction > 0.5:
        print("The image is an ID "+str(prediction)+"")
    else:
        print("The image is a document "+str(prediction))
