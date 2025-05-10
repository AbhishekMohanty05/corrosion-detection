# Corrosion Detection Using Xception Model

## Project Overview
This project implements a binary image classification model to detect corrosion in images using the Xception deep learning architecture. The model is trained on a dataset of images categorized into "corrosion" and "no corrosion" classes. The pipeline includes data preprocessing, augmentation, model training with transfer learning, evaluation, and prediction on new images.

## Dataset Structure
The dataset is organized into three main directories:
- `train/` : Training images, further divided into class subdirectories.
- `valid/` : Validation images, used to monitor model performance during training.
- `test/` : Test images, used for final evaluation of the trained model.

Each directory contains subfolders for the two classes:
- `corrosion/`
- `no corrosion/`

## Data Preprocessing and Augmentation
- Training images are rescaled and augmented with horizontal flips, zoom, and rotation to improve model generalization.
- Validation and test images are only rescaled.

## Model Architecture
- The base model is the pre-trained Xception model (ImageNet weights) without the top classification layers.
- The base model layers are frozen during training to leverage transfer learning.
- Custom layers added on top include:
  - Global Average Pooling
  - Dense layer with 512 units and ReLU activation
  - Dropout layer with 0.5 dropout rate for regularization
  - Output Dense layer with 1 unit and sigmoid activation for binary classification

## Training
- The model is compiled with the Adam optimizer and binary cross-entropy loss.
- Metrics tracked include accuracy.
- Early stopping is used to prevent overfitting, monitoring validation loss with patience of 5 epochs.
- Model checkpoints save the best model based on validation loss.
- Training runs for up to 30 epochs with batch size 16.

## Evaluation
- The best saved model is loaded for evaluation on the test dataset.
- Metrics reported include test loss, accuracy, and AUC.
- Confusion matrix and classification report are generated to analyze performance.
- Training and validation accuracy and loss curves are plotted.

## Prediction
- A utility function is provided to preprocess and predict the class of a single image.
- The prediction threshold is 0.5, with outputs "Corrosion" or "No Corrosion".

## How to Run
1. Mount your Google Drive or set up the dataset directories accordingly.
2. Install required dependencies (TensorFlow, Keras, scikit-learn, matplotlib, etc.).
3. Run the notebook cells sequentially to train, evaluate, and predict.
4. Adjust paths to your dataset directories as needed.

## Dependencies
- TensorFlow
- Keras
- scikit-learn
- matplotlib
- numpy

## Notes
- The model input size is fixed at 299x299 pixels, matching Xception's expected input.
- The test data generator is set with `shuffle=False` to maintain order for evaluation.
- The model uses binary classification with sigmoid activation.

---

This README provides an overview and instructions for the corrosion detection project using the Xception model. For any questions or issues, please refer to the notebook or contact the project maintainer.
