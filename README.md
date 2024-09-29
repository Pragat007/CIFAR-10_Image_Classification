# CIFAR-10 Image Classification with ResNet-164 and Gradio

This project focuses on classifying images from the CIFAR-10 dataset into 10 different categories using a ResNet-164 deep learning model. The model is built using TensorFlow/Keras, and the trained model is deployed using Gradio for interactive use.

## Project Structure

- **CIFAR_classification_ResNet164.ipynb**: A model based on the ResNet-164 architecture for CIFAR-10 classification.
- **Deployment**: The trained ResNet-164 model is deployed on Hugging Face using Gradio, providing an interactive interface to classify CIFAR-10 images.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 testing images.

### Classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Model

### ResNet-164 Model
- **File**: `CIFAR_classification_ResNet164.ipynb`
- **Architecture**: This model leverages the ResNet-164 architecture, which is a deep residual network designed for efficient training of very deep networks. The ResNet-164 model includes:
  - 18 residual blocks with two convolutional layers each.
  - Batch normalization and activation functions applied after each convolutional layer.
  - Skip connections (shortcuts) that help the model avoid the vanishing gradient problem and ensure better training performance.
  
The model is trained using TensorFlow/Keras and achieves high accuracy in classifying CIFAR-10 images.

### 1. Training:
- The model is trained for 30 epochs using the Adam optimizer and `sparse_categorical_crossentropy` loss function.
- Both training and validation accuracy are monitored during training.

### 2. Deployment:
The trained ResNet-164 model is deployed using Gradio, allowing users to upload images and receive predictions. Gradio provides an interactive interface to classify CIFAR-10 images into one of the 10 categories.

## Installation and Usage

To run this project locally, you'll need to install Python and the required libraries.

### Clone this repository:
  ```bash
  git clone https://github.com/Pragat007/cifar-10-object-detection.git
  ```

2. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Open the Jupyter notebooks:**

    ```bash
    jupyter notebook
    ```
4. **Run the cells in any of the notebook files** (`cifar_object_identification.ipynb`) **to train and evaluate the models.**

### Access the Deployed App

The model is deployed using Gradio. You can interact with the deployed model on [Hugging Face](https://huggingface.co/spaces/Pragat007/image_classification), which provides a user-friendly interface for testing CIFAR-10 classification

## Results

The ResNet-164 model achieves strong performance on the CIFAR-10 dataset. The use of residual connections helps in training deep networks efficiently, leading to higher accuracy in classification.

## Future Work

Potential improvements for this project include:

  -  Experimenting with different versions of ResNet or other architectures like EfficientNet.
  -  Adding data augmentation techniques to improve generalization.
  -  Further hyperparameter tuning for better performance.
