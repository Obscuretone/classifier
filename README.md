# Image Classifier Tool

This tool is designed to classify images into two categories, using a deep learning model built with TensorFlow and Keras. The two classes that the model can classify are typically used to distinguish between two types of images, such as documents and IDs. The tool includes two main scripts:
	
1.	`classifier.py` - Trains the model with a dataset of images and saves the trained model.
2.	`check.py` - Loads the trained model and uses it to predict the class of a given image.

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

## 1. Training the Model (`classifier.py`)

This script is used to train the CNN model using a labeled image dataset. The training process involves augmenting the images and fitting the model on the data for a specified number of epochs. Once trained, the model is saved as a .keras file.

Running the Script:

To train the model, run the following command:

```
python classifier.py --train_dir <path_to_train_data> --validation_dir <path_to_validation_data> --epochs <num_epochs>
```

- `--train_dir` : Path to the directory containing the training images. Images should be organized in subdirectories for each class.
- `--validation_dir` : Path to the directory containing the validation images, similarly organized by class.
- `--epochs` : Number of training epochs (e.g., 30).

```
python classifier.py --train_dir ./train --validation_dir ./validation --epochs 30
```

This command will train the model on the images in the train directory and validate it on the images in the validation directory for 30 epochs.

The trained model will be saved to a .keras file (cardordoc.keras by default).

## 2. Using the Trained Model for Prediction (`check.py`)

Once the model is trained, you can use check.py to classify new images. The script loads the trained model and predicts whether the image belongs to class 0 or class 1.

### Running the Script:

To predict the class of an image, run the following command:

```
python check.py <image_path> [-p]
```

- `<image_path>` : Path to the image you want to classify.
- `-p` or `--percentage` : Optional argument that outputs the prediction percentage instead of the class label.


```
python check.py ./test_image.jpg --percentage
```

This will output the prediction percentage for class 1 (e.g., “The prediction percentage for class 1 is 85.23%”).

```
python check.py ./test_image.jpg
```
This will output the class label (0 or 1) for the image.


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