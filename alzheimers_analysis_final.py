#!/usr/bin/env python3
# alzheimers_analysis_fixed.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow import keras

# Suppress warnings and set theme
warnings.filterwarnings("ignore")
sns.set_theme(context='notebook', palette='pastel', style='whitegrid')

def main():
    # Load data
    df = pd.read_csv('alzheimers_disease_data.csv')
    print("Dataset preview:")
    print(df.head(), '\n')
    print(f"Dataset shape: {df.shape}\n")

    # Data statistics
    print("Info:")
    df.info()
    print("\nMissing values per column:")
    print(df.isnull().sum(), '\n')

    # Drop unnecessary columns
    df.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)

    print("Descriptive statistics:")
    print(df.describe(), '\n')

    print(f"Duplicate rows: {df.duplicated().sum()}\n")

    # Correlation heatmap
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.1f', linewidths=.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # Histograms of numerical features
    num_cols = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality']
    for col in num_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.show()

    # Countplots for binary categorical features
    binary_cols = [
        'Forgetfulness', 'DifficultyCompletingTasks', 'Diagnosis', 'PersonalityChanges',
        'Disorientation', 'Confusion', 'BehavioralProblems', 'MemoryComplaints',
        'Hypertension', 'HeadInjury', 'Depression', 'Diabetes',
        'CardiovascularDisease', 'FamilyHistoryAlzheimers', 'Smoking'
    ]
    for col in binary_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=df[col], palette='pastel')
        plt.title(f'Distribution of {col}')
        plt.xticks([0, 1], ['No', 'Yes'])
        plt.tight_layout()
        plt.show()

    # Gender distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df['Gender'], palette='pastel')
    plt.title('Distribution of Gender')
    plt.xticks([0, 1], ['Male', 'Female'])
    plt.tight_layout()
    plt.show()

    # Education level
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df['EducationLevel'], palette='pastel')
    plt.title('Distribution of Education Level')
    plt.xticks([0, 1, 2, 3], ['None', 'High School', 'Bachelor', 'Higher'])
    plt.tight_layout()
    plt.show()

    # Ethnicity distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df['Ethnicity'], palette='pastel')
    plt.title('Distribution of Ethnicity')
    plt.xticks([0, 1, 2, 3], ['Caucasian', 'African American', 'Asian', 'Other'])
    plt.tight_layout()
    plt.show()

    # Preprocessing
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']

    # PCA for dimensionality reduction
    pca = PCA(0.99)
    X_pca = pca.fit_transform(X)
    print(f"PCA reduced features shape: {X_pca.shape}\n")

    # Train-test split and scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}\n")

    # Build and train ANN model
    model = keras.Sequential([
        keras.layers.Dense(X_train_scaled.shape[1], activation='relu', input_shape=(X_train_scaled.shape[1],)),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(2, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train_scaled, y_train, epochs=70, verbose=1)

    print("\nModel Summary:")
    model.summary()

    # Plot training metrics
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['loss'], label='Loss')
    plt.title('Training Metrics')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Evaluation
    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
