# AI Image Caption Generator

# Project Description

The AI Image Caption Generator is a deep learning-based application that automatically generates descriptive captions for images. The project combines **Computer Vision** and **Natural Language Processing (NLP)** by using a pre-trained **ResNet50** model for image feature extraction and an **LSTM (Long Short-Term Memory)** network for caption generation. A Flask-based web application provides an interactive interface where users can upload an image and receive a predicted caption.

# Problem Statement

Understanding the content of an image and describing it in natural language is a challenging task. This project addresses this problem by developing an AI model capable of generating relevant and meaningful captions for images automatically.

# Features

* Upload images through a web interface.
* Generate captions using a trained deep learning model.
* Extract image features using ResNet50.
* Predict captions using an LSTM network.
* Simple and responsive Flask application.

# Technologies Used

* Python
* TensorFlow & Keras
* Flask
* ResNet50
* LSTM
* NumPy
* Pickle
* HTML & CSS

# Project Workflow

1. Preprocess the image-caption dataset.
2. Extract image features using the pre-trained ResNet50 model.
3. Train the LSTM model using image features and corresponding captions.
4. Save the trained model and tokenizer.
5. Upload an image through the Flask application.
6. Generate and display the predicted caption.

# Folder Structure

```text
Image_Captioning/
│── app.py
│── train.py
│── preprocessing.py
│── feature_extraction.py
│── predict.py
│── requirements.txt
│── model.keras
│── tokenizer.pkl
│── features.pkl
├── dataset/
├── static/
└── templates/
```

