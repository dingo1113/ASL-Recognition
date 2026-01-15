[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://colab.research.google.com/github/dingo1113/asl-hand-gesture-recognition
/blob/main/Final_ASL_Project_Code.ipynb)

# Real-Time ASL Hand Gesture Recognition

This project implements a convolutional neural network (CNN) to classify
American Sign Language (ASL) hand gestures from image data using deep learning.
The model is trained and evaluated in Google Colab using TensorFlow/Keras.

## Overview
The goal of this project is to recognize ASL hand gestures from images and
support real-time inference through webcam input. The system includes image
preprocessing, CNN model training, and performance evaluation.

Key components include:
- Image preprocessing and normalization
- CNN-based image classification
- Model training and validation
- Real-time gesture inference support

## Data
- Image-based ASL dataset (not included in this repository due to size)
- Data preprocessing includes resizing, normalization, and batching
- Dataset structured for multi-class classification

## Model
- Convolutional Neural Network (CNN)
- Implemented using TensorFlow/Keras
- Designed for image-based gesture recognition
- Optimized for real-time inference latency

## Evaluation
- Model performance evaluated using classification accuracy and loss
- Training and validation metrics tracked during training
- Architecture structured to allow easy extension to additional gesture classes

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Google Colab

## Repository Contents
- `Final_ASL_Project_Code.ipynb` — End-to-end notebook including preprocessing,
  model training, and evaluation

## Notes
- Dataset files are excluded from this repository due to size constraints
- The notebook was developed and executed in Google Colab
- Project is intended for educational and research purposes

## Future Improvements
- Expand to full ASL alphabet classification
- Improve robustness to lighting and background variation
- Optimize model for deployment on edge devices
