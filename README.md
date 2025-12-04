Sentiment Analyzer App

A Deep Learning text-classification project using PyTorch and Flask, deployable on Google Cloud App Engine.

Overview

This project implements a sentiment analysis system that classifies text reviews as positive or negative using a custom deep-learning model.
It includes:

- A dataset loader and vocabulary builder (utils.py)

A PyTorch SentimentAnalysis model (EmbeddingBag + linear head) (model.py)

A training pipeline with validation and test evaluation (train.py)

A Flask web app to run inference on user-submitted text reviews (main.py)

A Google Cloud App Engine configuration (app.yaml)

The project is designed to be both a learning exercise and a flexible template for deploying NLP models in production environments.

📂 Project Structure
sentiment_analyzer_app/
│
├── train.py               # Training loop, evaluation, saving model artifacts
├── model.py               # PyTorch model definition
├── utils.py               # Dataset loader, vocabulary, batching utilities
├── main.py               # Flask web application for inference
├── requirements.txt       # Project dependencies
├── app.yaml               # Google Cloud App Engine configuration
├── templates/
│   ├── hello.html         # Landing page
│   └── result.html        # Result page
└── state_dict.pt          # Saved model checkpoint (after training)

Dataset

This project uses a binary sentiment dataset similar to the Yelp Review Polarity dataset.

General characteristics of the dataset:

- Two labels:

    -  0 → negative reviews

    - 1 → positive reviews

- Reviews consist of raw text strings.

- During preprocessing:

    - Text is lower-cased and cleaned of non-alphabetic characters.

    - A custom vocabulary is built from the training corpus. (utils.py)

    - Reviews are tokenized and converted to integer sequences.

The dataset loader automatically:

    - Reads training and test splits.

    - Builds a vocabulary with <pad> and <unk> tokens.

    - Wraps text/label pairs into PyTorch Dataset objects.

    - Provides a custom collate_batch() to handle variable-length sequences. (utils.py)

Installation
Using Conda (recommended)
conda create --name sentiment-app python>=3.10
conda activate sentiment-app


Then install project dependencies:

pip install -r requirements.txt


(See the full dependency list in requirements.txt )

Training the Model

To start model training, run:

python train.py


This script:

    - Loads and preprocesses the dataset.(utils.py)

    - Initializes the PyTorch model (SentimentAnalysis). (model.py)

    - Trains for several epochs using SGD + LR scheduler. (train.py)

    - Evaluates on validation and test datasets.

    - Saves the checkpoint as state_dict.pt, including:

        - model weights

        - optimizer state
        
        - vocabulary
        
        - hyperparameters

This file is later used by the web app for inference.

Running the Web Application

After training, launch the Flask app:

python main.py


Then open your browser at:

http://localhost:8080


The web app:

    - Loads the saved checkpoint on first request. (main.py)

    - Tokenizes user input with the same preprocessing as training.
    
    - Performs inference using the trained model.
    
    - Returns the estimated probability of a positive review.

Endpoints include:

 - / → landing page (hello.html)

 - /predict → form submission route (POST)

Deploying to Google Cloud App Engine

The project includes an App Engine config file (app.yaml) allowing deployment on Google Cloud.

General deployment workflow:

1. Install Google Cloud SDK
    gcloud init
    gcloud auth login
    gcloud config set project <your_project_id>

2. Deploy the application

    Inside sentiment_analyzer_app/:

    gcloud app deploy

3. Visit your hosted app
    gcloud app browse


The App Engine environment automatically:

    - Installs dependencies

    - Starts the Flask app based on app.yaml settings

    - Serves HTTP requests via a production web server

Model Architecture (Summary)

The underlying model is a simple but efficient text classifier:

    - EmbeddingBag layer (mean-pooled embeddings)
    
    - Linear classifier
    
    - CrossEntropy loss
    
    - SGD optimizer with step LR scheduler

This design is well-suited for large text corpora due to its speed and memory efficiency.
(See full implementation in model.py)

How to Extend This Project

    - You can generalize this template for many NLP tasks:
     
    - Replace the dataset with your own text corpus.
    
    - Add n-gram support or switch to transformer embeddings.
    
    - Modify the Flask interface to support batch predictions or APIs.
    
    - Deploy to other cloud providers or use Docker containers.

