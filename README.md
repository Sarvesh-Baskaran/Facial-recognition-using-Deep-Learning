# Facial-recognition-using-Deep-Learning
A simple face recognition software which detects individuals present in the dataset using a pre-trained Neural Networks (MobileNet)

This project is developed as part of a Deep Learning course and demonstrates the application of neural network models for data analysis and prediction. The notebook includes end-to-end implementation, from data preprocessing to model training, evaluation, and visualization of results.

## Overview

The project demonstrates a practical implementation of Deep Learning concepts using Python. It utilizes popular libraries like TensorFlow/Keras and PyTorch (depending on the implementation) for developing and evaluating models. The aim is to gain hands-on experience in building and training neural networks.

## Objective

The objective of this project is to:
- Implement a deep learning model from scratch.
- Apply preprocessing and normalization techniques to input data.
- Train and validate models on selected datasets.
- Compare the performance of models using evaluation metrics.
- Visualize results and understand model behavior.

## Dataset

The dataset used in this project are user obtained datasets.
Preprocessing includes:
- Handling missing values.
- Normalizing and reshaping data.
- Splitting data into training and testing sets.

## Model Architecture

The model architecture may include one or more of the following:
- Input Layer
- Hidden Layers (Dense / Convolutional / LSTM / GRU etc.)
- Output Layer

Activation functions such as ReLU, Sigmoid, or Softmax are used depending on the task.
Optimization is done using Adam, SGD, or similar optimizers, and loss functions are chosen based on whether the task is classification or regression.

## Implementation Details

Key steps implemented in the notebook:
- Data Import and Preprocessing
- Model Definition using Sequential or Functional API
- Model Compilation (optimizer, loss, metrics)
- Model Training with defined epochs and batch size
- Model Evaluation on test data
- Visualization of Accuracy, Loss, and Predictions

## Results

The model’s performance is evaluated using metrics like:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Plots and graphs demonstrate the training and validation performance across epochs.
