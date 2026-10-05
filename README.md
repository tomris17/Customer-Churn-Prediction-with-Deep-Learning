# Customer Churn Prediction using ANN-Deep Learning

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

This repository contains a deep learning project that trains an Artificial Neural Network (ANN) using TensorFlow and Keras to predict customer churn based on demographic and financial features[cite: 17].

---

## Dataset Notice
*Note: The dataset (`Churn_Modelling.csv`) used in this project[cite: 17] is publicly available on Kaggle.*

---

## Dataset Features & Preprocessing
* **Dropping Unnecessary Columns**: Removed `RowNumber`, `CustomerId`, and `Surname`[cite: 17].
* **Label Encoding**: Converted categorical text columns such as `Geography` and `Gender` into numerical representations[cite: 17].
* **Feature Scaling**: Normalized features using `StandardScaler`[cite: 17].
* **Train/Test Split**: Partitioned the data into training and testing sets[cite: 17].

---

## Project Workflow
1. **Data Loading & Cleaning**: Reading the CSV file and dropping redundant identifiers[cite: 17].
2. **Preprocessing**: Applying label encoding and standard scaling[cite: 17].
3. **Model Architecture**: Building a Sequential neural network containing Dense layers with ReLU activations and a sigmoid output layer[cite: 17].
4. **Model Compilation**: Compiling with the Adam optimizer and binary crossentropy loss[cite: 17].
5. **Training**: Fitting the model for 20 epochs with validation data[cite: 17].
6. **Model Persistence**: Saving the trained neural network model into `churn_ann_model.h5` using Keras model saving[cite: 17].
7. **Web Application**: Interactive deployment interface built with Streamlit.

---

## Getting Started & Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/customer-churn-ann.git](https://github.com/YOUR_USERNAME/customer-churn-ann.git)
   cd customer-churn-ann
