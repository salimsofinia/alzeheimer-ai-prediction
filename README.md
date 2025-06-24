# Alzheimer's Prediction Web Application

This web application provides a streamlined interface for predicting the likelihood of Alzheimer's disease using a machine learning pipeline. Built with Python and Streamlit, the app integrates data preprocessing, visualization, model training, evaluation, and reporting in a step-by-step process.

## Features

- **Data Upload and Display**: Load and preview the dataset.
- **Descriptive Statistics**: Summarize dataset statistics and feature distributions.
- **Data Visualization**: View correlation heatmaps and histograms of key attributes.
- **Preprocessing**: Perform PCA, train-test split, and feature normalization.
- **Model Building**: Train a neural network using Keras and TensorFlow.
- **Evaluation**: View accuracy, ROC curve, confusion matrix, and classification report.
- **Conclusion**: Summarized metrics and diagnostic interpretation.

## Requirements

Ensure that the following files are present in the project directory:
- `alzheimers_prediction.py` – The main Streamlit application script.
- `alzheimers_disease_data.csv` – The dataset file.
- `requirements.txt` – Contains the list of Python dependencies.

## Installation

1. Clone or download the repository and navigate into the project folder.
2. Install dependencies using pip:
   pip install -r requirements.txt
3. Launch the Streamlit application:
   streamlit run alzheimers_prediction.py

## Technologies Used

- Python 3  
- Streamlit  
- Pandas  
- NumPy  
- Seaborn & Matplotlib  
- Scikit-learn  
- TensorFlow (Keras)  

## Data and Model

- **Input Data**: Health-related patient data including symptoms and risk factors.  
- **Preprocessing**: PCA for dimensionality reduction and standard scaling.  
- **Model**: Feed-forward neural network with ReLU activations and a sigmoid output.  
- **Evaluation Metrics**: Accuracy, ROC-AUC, Precision, Recall, F1-score.  

## Notes

- The application includes a multi-page interface controlled via session state.  
- PCA retains 99% variance to optimize model performance.  
- Binary classification is conducted with a sigmoid output and 0.5 threshold.
   
