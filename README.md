<h1 align="center">
  <br>
  Multiclass Text Classification using Convolutional Neural Networks (CNNs)
  <br>
</h1>

<h4 align="center">This repository contains code for a text classification model implemented in Python using Keras with TensorFlow backend. The model is trained to classify Sinhala news articles into different categories using a convolutional neural network architecture.
  
<p align="center">
  <a href="https://"><img src="https://img.shields.io/badge/language-python-2ea42f?logo=python" alt="language - python"></a>
  <a href="https://"><img src="https://img.shields.io/badge/ML Classifier-orange?logo=ML" alt="Machine LearningC"></a>
  <br>
</p>

# Description

## Dataset

The dataset used for training and evaluation is a collection of Sinhala news articles obtained from various sources. It contains articles from the following news websites:
- Dailymirror_SL
- colombotelegrap
- NewsfirstSL
- theisland_lk__
- CeylonToday
- NewsWireLK
- colombogazette
- TheMorningLK

## Preprocessing

- The text data is preprocessed to remove any unwanted characters or symbols.
- Labels are encoded into integer format for multiclass classification.
- The dataset is split into training and testing sets.

## Model Architecture

- The model architecture consists of an embedding layer, a 1D convolutional layer, global max pooling, and fully connected layers.
- Pre-trained GloVe word embeddings are used to initialize the embedding layer.
- The model is trained using categorical cross-entropy loss and Adam optimizer.

## Evaluation Metrics

- The model's performance is evaluated using accuracy, F1 score, precision, and recall.
- Accuracy: The proportion of correctly classified samples.
- F1 Score: The weighted average of precision and recall.
- Precision: The proportion of true positive predictions among all positive predictions.
- Recall: The proportion of true positive predictions among all actual positive samples.

## Sample Prediction

A sample text is provided to demonstrate how the model predicts the category of a news article. The predicted class along with precision and recall scores are printed for evaluation.

## Usage

1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the `main.py` script to train and evaluate the model.
4. Modify the script to customize the model architecture or dataset according to your requirements.

## Requirements

Make requirements.txt file
- Python 3.x
- TensorFlow
- Keras
- Pandas
- NumPy
- tqdm
