# CIFAR-10 CNN Classification

This project implements a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset using TensorFlow and Keras.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

Classes:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Model Architecture

The CNN model includes:
- Convolutional layers with ReLU activation
- Max pooling layers
- Dropout for regularization
- Dense layers for classification

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib

## Installation

Install the required packages using pip:

```
pip install tensorflow numpy matplotlib
```

## Usage

1. Open the `CNN-CIFAR10.ipynb` notebook in Jupyter Notebook or JupyterLab.
2. Run the cells in order to:
   - Import libraries
   - Load and preprocess the CIFAR-10 dataset
   - Build and compile the CNN model
   - Train the model
   - Evaluate the model on test data
   - Make predictions on sample images

## Results

The trained model achieves high accuracy on the CIFAR-10 test set. The saved models are:
- `cifar10_cnn_model.h5`: Standard model

## Files

- `CNN-CIFAR10.ipynb`: Jupyter notebook containing the complete implementation
- `cifar10_cnn_model.h5`: Saved trained model
- `README.md`: This file

## License

This project is for educational purposes.

