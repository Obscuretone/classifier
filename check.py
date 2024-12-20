import sys
import argparse
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

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

# Load the model
model = load_model('./cardordoc.keras')

# Predict the class of the image
prediction = model.predict(img_tensor)

if args.percentage:
    print(f"The prediction percentage for class 1 is {prediction[0][0]*100}%")
else:
    # If prediction is > 0.5 it is one class, otherwise it is the other class
    if prediction > 0.5:
        print("The image belongs to class 1")
    else:
        print("The image belongs to class 0")
