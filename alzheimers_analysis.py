# 1. Import Libraries ==================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

sns.set_theme(context='notebook', palette='pastel', style='whitegrid')

# 2. Load data =========================================================================================================

df=pd.read_csv('/kaggle/input/alzheimers-disease-dataset/alzheimers_disease_data.csv')
df.head()

df.shape

# 3. Data Statistics ===================================================================================================

df.info()

df.isnull().sum()

# Drop unnecessary column from the DataFrame
df.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)

df.describe()

sum(df.duplicated())

df.corr()

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

# 4. Data Visualization ================================================================================================

# Lista de columnas específicas que deseas graficar
columns_to_plot = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality']  

# Graficar histograma para cada columna en la lista específica
for column in columns_to_plot:
    plt.figure(figsize=(9, 5))
    sns.histplot(data=df, x=column, kde=True, bins=30)
    plt.title(f'Distribution of {column}')
    plt.show()

# Set custom labels
labels = ["No", "Yes"]
ticks = range(len(labels))

# Create a figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the count plot
sns.countplot(data=df, x='Forgetfulness', ax=ax, palette=sns.color_palette("pastel"))
ax.set_title('Distribution of Forgetfulness')
ax.set_xticks(ticks)
ax.set_xticklabels(labels)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Set custom labels
labels = ["No", "Yes"]
ticks = range(len(labels))

# Create a figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the count plot
sns.countplot(data=df, x='DifficultyCompletingTasks', ax=ax, palette=sns.color_palette("pastel"))
ax.set_title('Distribution of Difficulty to Completing Tasks    ')
ax.set_xticks(ticks)
ax.set_xticklabels(labels)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Set custom labels
labels = ["No", "Yes"]
ticks = range(len(labels))

# Create a figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the count plot
sns.countplot(data=df, x='Diagnosis', ax=ax, palette=sns.color_palette("pastel"))
ax.set_title('Distribution of Alzheimers Diagnosis')
ax.set_xticks(ticks)
ax.set_xticklabels(labels)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Set custom labels
labels = ["No", "Yes"]
ticks = range(len(labels))

# Create a figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the count plot
sns.countplot(data=df, x='PersonalityChanges', ax=ax, palette=sns.color_palette("pastel"))
ax.set_title('Distribution of Personality Changes')
ax.set_xticks(ticks)
ax.set_xticklabels(labels)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Set custom labels
labels = ["No", "Yes"]
ticks = range(len(labels))

# Create a figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the count plot
sns.countplot(data=df, x='Disorientation', ax=ax, palette=sns.color_palette("pastel"))
ax.set_title('Distribution of Disorientation')
ax.set_xticks(ticks)
ax.set_xticklabels(labels)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Set custom labels
labels = ["No", "Yes"]
ticks = range(len(labels))

# Create a figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the count plot
sns.countplot(data=df, x='Confusion', ax=ax, palette=sns.color_palette("pastel"))
ax.set_title('Distribution of Confusion')
ax.set_xticks(ticks)
ax.set_xticklabels(labels)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Set custom labels
labels = ["No", "Yes"]
ticks = range(len(labels))

# Create a figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the count plot
sns.countplot(data=df, x='BehavioralProblems', ax=ax, palette=sns.color_palette("pastel"))
ax.set_title('Distribution of Behavioral Problems')
ax.set_xticks(ticks)
ax.set_xticklabels(labels)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Set custom labels
labels = ["No", "Yes"]
ticks = range(len(labels))

# Create a figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the count plot
sns.countplot(data=df, x='MemoryComplaints', ax=ax, palette=sns.color_palette("pastel"))
ax.set_title('Distribution of Memory Complaints')
ax.set_xticks(ticks)
ax.set_xticklabels(labels)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Set custom labels
labels = ["No", "Yes"]
ticks = range(len(labels))

# Create a figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the count plot
sns.countplot(data=df, x='Hypertension', ax=ax, palette=sns.color_palette("pastel"))
ax.set_title('Distribution of Hypertension Disease')
ax.set_xticks(ticks)
ax.set_xticklabels(labels)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Set custom labels
labels = ["No", "Yes"]
ticks = range(len(labels))

# Create a figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the count plot
sns.countplot(data=df, x='HeadInjury', ax=ax, palette=sns.color_palette("pastel"))
ax.set_title('Distribution of Head Injury')
ax.set_xticks(ticks)
ax.set_xticklabels(labels)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Set custom labels
labels = ["No", "Yes"]
ticks = range(len(labels))

# Create a figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the count plot
sns.countplot(data=df, x='Depression', ax=ax, palette=sns.color_palette("pastel"))
ax.set_title('Distribution of Depression Problems')
ax.set_xticks(ticks)
ax.set_xticklabels(labels)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Set custom labels
labels = ["No", "Yes"]
ticks = range(len(labels))

# Create a figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the count plot
sns.countplot(data=df, x='Diabetes', ax=ax, palette=sns.color_palette("pastel"))
ax.set_title('Distribution of Diabetes Disease')
ax.set_xticks(ticks)
ax.set_xticklabels(labels)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Set custom labels
labels = ["No", "Yes"]
ticks = range(len(labels))

# Create a figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the count plot
sns.countplot(data=df, x='CardiovascularDisease', ax=ax, palette=sns.color_palette("pastel"))
ax.set_title('Distribution of Cardiovascular Disease')
ax.set_xticks(ticks)
ax.set_xticklabels(labels)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Set custom labels
labels = ["No", "Yes"]
ticks = range(len(labels))

# Create a figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the count plot
sns.countplot(data=df, x='FamilyHistoryAlzheimers', ax=ax, palette=sns.color_palette("pastel"))
ax.set_title('Distribution of Family History Alzheimer Disease')
ax.set_xticks(ticks)
ax.set_xticklabels(labels)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Set custom labels
labels = ["No", "Yes"]
ticks = range(len(labels))

# Create a figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the count plot
sns.countplot(data=df, x='Smoking', ax=ax, palette=sns.color_palette("pastel"))
ax.set_title('Distribution of Smoking history')
ax.set_xticks(ticks)
ax.set_xticklabels(labels)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Set custom labels
labels = ["Male", "Female"]
ticks = range(len(labels))

# Create a figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the count plot
sns.countplot(data=df, x='Gender', ax=ax, palette=sns.color_palette("pastel"))
ax.set_title('Distribution of Gender')
ax.set_xticks(ticks)
ax.set_xticklabels(labels)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Set custom labels
labels = ["None", "High School", 'Bachelor', "Higher"]
ticks = range(len(labels))

# Create a figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the count plot
sns.countplot(data=df, x='EducationLevel', ax=ax, palette=sns.color_palette("pastel"))
ax.set_title('Distribution of Education Level')
ax.set_xticks(ticks)
ax.set_xticklabels(labels)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Set custom labels
labels = ["Caucasican", "African American", 'Asian', "Other"]
ticks = range(len(labels))

# Create a figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the count plot
sns.countplot(data=df, x='Ethnicity', ax=ax, palette=sns.color_palette("pastel"))
ax.set_title('Distribution of Ethnicity')
ax.set_xticks(ticks)
ax.set_xticklabels(labels)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# 5. Data Pre-processing ===============================================================================================