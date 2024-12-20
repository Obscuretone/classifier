# Image Classifier Tool

A business unit was facing frustration with users uploading files to the wrong locations, so I suggested solving the problem by implementing AI for better accuracy.

This tool is designed to classify images into two categories using a deep learning model built with TensorFlow and Keras. 

The model distinguishes between two types of images, such as documents and IDs. It was originally created as a demo to showcase how AI can differentiate between ID documents and signed contracts. 

The tool consists of two main scripts that handle training the model and classifying images.
	
1.	`train.py` - Trains the model with a dataset of images and saves the trained model.
2.	`classify.py` - Loads the trained model and uses it to predict the class of a given image.

## Features
- **Image Preprocessing:** Images are resized to 150x150 pixels and normalized to the range [0, 1].
- **Data Augmentation:** The training data is augmented with various transformations (rotation, shifting, flipping, etc.) to help the model generalize better.
- **Binary Classification:** The model classifies images into two classes (e.g., ID vs. document) using a simple Convolutional Neural Network (CNN).
- **Prediction Output:** The model’s prediction can be displayed as a percentage or a class label (0 or 1).

## Requirements
- Python 3.11.9 (Matched to Tensorflow)
- Tested with TensorFlow 2.18.0
- Keras
- NumPy
- argparse
- Pillow
- scipy

To install the required libraries, use:

```
pip install tensorflow numpy argparse Pillow scipy
```

How to Use

## 1. Training the Model (`train.py`)

This script is used to train the CNN model using a labeled image dataset. The training process involves augmenting the images and fitting the model on the data for a specified number of epochs. Once trained, the model is saved as a .keras file.

Running the Script:

To train the model, run the following command:

```
python train.py
```

This command will train the model on the images in the train directory and validate it on the images in the validation directory for 30 epochs.

The trained model will be saved to a .keras file (cardordoc.keras by default).

## 2. Using the Trained Model for Prediction (`classify.py`)

Once the model is trained, you can use classify.py to classify new images. The script loads the trained model and predicts whether the image belongs to class 0 or class 1.

### Running the Script:

To predict the class of an image, run the following command:

```
python classify.py <image_path>
```

For example, 
```
python classify.py test1.jpg
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 113ms/step
The image is an ID [[0.9477982]]

python classify.py test2.png
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 102ms/step
The image is a document [[0.05471132]]

```
```
evandentremont@Mac classifier % python classify.py test2.png --percentage
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 94ms/step
The prediction percentage for class 1 is 5.471131801605225%
```




## Model Architecture

The model is a simple CNN architecture that includes:
- 4 convolutional layers with ReLU activations.
- Max-pooling layers after each convolution.
- A fully connected dense layer with 512 units and ReLU activation.
- A final dense layer with 1 unit and a sigmoid activation function to output a binary classification result.

## Notes
- The model expects the images to be resized to 150x150 pixels.
- Ensure the images are organized in subdirectories according to their class (e.g., train/class_0/, train/class_1/).
- The model is saved as a .keras file and can be loaded in check.py for inference.