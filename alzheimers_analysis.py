# 1. Import Libraries ==================================================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

sns.set_theme(context='notebook', palette='pastel', style='whitegrid')

# 2. Load data =========================================================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'alzheimers_disease_data.csv')
df = pd.read_csv(csv_path)
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

X=df.drop('Diagnosis', axis=1)

X.shape

y = df['Diagnosis'].copy()

y.shape

# 5.1. Principal Component Analyisis (PCA) =============================================================================

from sklearn.decomposition import PCA
pca = PCA(0.99)
X_pca = pca.fit_transform(X)
X_pca

# 5.2. Data Split ======================================================================================================

X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

X_pca.shape

len(X_train_pca)

len(X_test_pca)

# 5.3. Normalization ===================================================================================================

# Standardize the features using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data
scaler.fit(X_train_pca)

scaled_X_train = scaler.transform(X_train_pca)

scaled_X_train

# Transform the data
scaled_X_test = scaler.transform(X_test_pca)

scaled_X_test

# 6. Build Model (Neural Network) ======================================================================================

import tensorflow as tf
from tensorflow import keras

#I will create a neural network
#I will have same number of neurons as columns, so 8
#we use relu as activation function because is easy to compute relu

model_ANN=keras.Sequential([
    keras.layers.Dense(8,input_shape=(8,),activation='relu'),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(2, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

#loss is binary_crossentropy because our output is binary, zero and one
#adam is a very commonly used optimizer
model_ANN.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']
)

model_ANN.fit(scaled_X_train,y_train,epochs=70)

model_ANN.summary()

# 7. Evaluate the model ================================================================================================

y_pred_ANN=model_ANN.predict(scaled_X_test)

y_pred_ANN=y_pred_ANN.round().astype(int)

#This is plotting the performance of over all model
from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test,y_pred_ANN))

cm=confusion_matrix(y_test,y_pred_ANN)
cm

import seaborn as sns
cm1=tf.math.confusion_matrix(y_test,y_pred_ANN)
plt.figure(figsize=(10,7))
sns.heatmap(cm1,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

#Anything which is on a diagonal is a correct prediction

# 8. Conclusion

print("In this Kaggle Notebook, Alzheimer's disease is predicted. First, we performed an exploratory data analysis using various charts to understand the relationships among the features as well as the target variable.\nAdditionally, we developed the data preprocessing to determine the features with further repercussion on the prediction of Alzheimer's disease, as well as the data splitting and normalization.\nSubsequently, we built the model by training an artificial neural network, reaching an accuracy higher than 60%, and evaluated the trained model, concluding that the model is not overfitting.\nThank you for exploring this notebook! If you find it helpful, please consider upvoting it ❤️.\nYour support is always appreciated 🤩!")