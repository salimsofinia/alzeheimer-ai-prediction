#!/usr/bin/env python3

import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Suppress warnings and set seaborn theme
warnings.filterwarnings("ignore")
sns.set_theme(context='notebook', palette='pastel', style='whitegrid')

def main():
    # Argument parser with default CSV path in current directory
    parser = argparse.ArgumentParser(description="Alzheimer's Disease Dataset Analysis")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="alzheimers_disease_data.csv",
        help="Path to the Alzheimer's dataset CSV file (default: alzheimers_disease_data.csv in current directory)"
    )
    args = parser.parse_args()

    # Load the dataset, handling missing file
    try:
        df = pd.read_csv(args.csv_path)
    except FileNotFoundError:
        print(f"Error: File '{args.csv_path}' not found.")
        return

    # Display basic information
    print("First 5 rows:")
    print(df.head(), "\n")
    print("Shape:", df.shape, "\n")
    print("Data info:")
    df.info()
    print("\nMissing values per column:")
    print(df.isnull().sum(), "\n")

    # Drop unnecessary columns if they exist
    for col in ['PatientID', 'DoctorInCharge']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Descriptive statistics and duplicates
    print("Descriptive statistics:")
    print(df.describe(), "\n")
    print("Number of duplicate rows:", df.duplicated().sum(), "\n")

    # Correlation matrix
    corr = df.corr()
    print("Correlation matrix:")
    print(corr, "\n")

    # Heatmap of correlations
    plt.figure(figsize=(18, 18))
    sns.heatmap(corr, annot=True, linewidths=.5, fmt='.1f')
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # Numerical feature distributions
    numerical_cols = [
        'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity',
        'DietQuality', 'SleepQuality'
    ]
    for col in numerical_cols:
        if col in df.columns:
            plt.figure(figsize=(9, 5))
            sns.histplot(data=df, x=col, kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.show()

    # Binary feature distributions
    binary_cols = [
        'Forgetfulness', 'DifficultyCompletingTasks', 'Diagnosis',
        'PersonalityChanges', 'Disorientation', 'Confusion',
        'BehavioralProblems', 'MemoryComplaints', 'Hypertension',
        'HeadInjury', 'Depression', 'Diabetes',
        'CardiovascularDisease', 'FamilyHistoryAlzheimers', 'Smoking'
    ]
    for col in binary_cols:
        if col in df.columns:
            plt.figure(figsize=(8, 5))
            sns.countplot(data=df, x=col)
            plt.title(f'Distribution of {col}')
            plt.xticks([0, 1], ["No", "Yes"])
            plt.tight_layout()
            plt.show()

    # Categorical feature distributions: Gender, EducationLevel, Ethnicity
    if 'Gender' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x='Gender')
        plt.title('Distribution of Gender')
        plt.xticks([0, 1], ["Male", "Female"])
        plt.tight_layout()
        plt.show()

    if 'EducationLevel' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x='EducationLevel')
        plt.title('Distribution of Education Level')
        plt.xticks([0, 1, 2, 3], ["None", "High School", "Bachelor", "Higher"])
        plt.tight_layout()
        plt.show()

    if 'Ethnicity' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x='Ethnicity')
        plt.title('Distribution of Ethnicity')
        plt.xticks([0, 1, 2, 3], ["Caucasian", "African American", "Asian", "Other"])
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
