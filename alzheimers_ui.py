import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

# --- Page Setup ---
st.set_page_config(page_title="Alzheimer's Analysis Wizard", layout="wide")
st.title("Alzheimer's Prediction")

# Table of contents
sections = [
    "Import Libraries",
    "Load Data",
    "Data Statistics",
    "Data Visualization",
    "Data Pre-processing",
    "Build Model (Neural Network)",
    "Evaluate Model",
    "Conclusion"
]

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 0

# Navigation buttons
cols = st.columns([1,2,1])
with cols[0]:
    if st.button("Previous"):
        st.session_state.page = max(0, st.session_state.page - 1)
with cols[2]:
    if st.button("Next"):
        st.session_state.page = min(len(sections)-1, st.session_state.page + 1)

# Display current section
st.header(f"Step {st.session_state.page+1}: {sections[st.session_state.page]}")

# Cached data loader with cleaning
@st.cache_data
def load_data():
    df = pd.read_csv("alzheimers_disease_data.csv")
    # Drop non-informative columns
    df.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)
    return df

# Section logic
if st.session_state.page == 0:
    st.write("**Libraries** imported: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, tensorflow, streamlit, etc.")
    st.code(
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
# and more...
""", language='python')

elif st.session_state.page == 1:
    df = load_data()
    st.success("Data loaded successfully!")
    st.write("**Data Preview:**")
    st.dataframe(df.head())
    st.write(f"**Dataset shape:** {df.shape}")
    if st.checkbox("Show raw data (all rows)"):
        st.dataframe(df)

elif st.session_state.page == 2:
    df = load_data()
    st.subheader("Descriptive Statistics and Missing Values")
    # Descriptive stats
    st.write(df.describe())
    # Missing values
    st.write("**Missing Values per Column:**")
    st.write(df.isnull().sum())
    # Data info
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

elif st.session_state.page == 3:
    df = load_data()
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(corr, annot=True, fmt='.1f', linewidths=.5, ax=ax)
    st.pyplot(fig)

    st.subheader("Histograms of Numerical Features")
    num_cols = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality']
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(data=df, x=col, kde=True, bins=30, ax=ax)
        ax.set_title(f'Distribution of {col}')
        st.pyplot(fig)

    st.subheader("Countplots of Categorical Features")
    # Binary features
    binary_cols = ['Forgetfulness', 'DifficultyCompletingTasks', 'Diagnosis', 'PersonalityChanges',
                   'Disorientation', 'Confusion', 'BehavioralProblems', 'MemoryComplaints',
                   'Hypertension', 'HeadInjury', 'Depression', 'Diabetes',
                   'CardiovascularDisease', 'FamilyHistoryAlzheimers', 'Smoking']
    for col in binary_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df, x=col, ax=ax, palette=sns.color_palette("pastel"))
        ax.set_title(f'Distribution of {col}')
        ax.set_xticks([0,1])
        ax.set_xticklabels(["No","Yes"])
        st.pyplot(fig)
    # Gender
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x='Gender', ax=ax, palette=sns.color_palette("pastel"))
    ax.set_title('Distribution of Gender')
    ax.set_xticks([0,1]); ax.set_xticklabels(["Male","Female"])
    st.pyplot(fig)
    # Education Level
    fig, ax = plt.subplots(figsize=(8, 4))
    labels_ed = ["None","High School","Bachelor","Higher"]
    sns.countplot(data=df, x='EducationLevel', ax=ax, palette=sns.color_palette("pastel"))
    ax.set_title('Distribution of Education Level')
    ax.set_xticks(range(len(labels_ed)))
    ax.set_xticklabels(labels_ed)
    st.pyplot(fig)
    # Ethnicity
    fig, ax = plt.subplots(figsize=(8, 4))
    labels_eth = ["Caucasican","African American","Asian","Other"]
    sns.countplot(data=df, x='Ethnicity', ax=ax, palette=sns.color_palette("pastel"))
    ax.set_title('Distribution of Ethnicity')
    ax.set_xticks(range(len(labels_eth)))
    ax.set_xticklabels(labels_eth)
    st.pyplot(fig)

elif st.session_state.page == 4:
    df = load_data()
    df_features = df.drop('Diagnosis', axis=1)
    df_target = df['Diagnosis']
    st.subheader("Feature and Target Shapes")
    st.write(f"Features shape: {df_features.shape}")
    st.write(f"Target shape: {df_target.shape}")

    st.subheader("Principal Component Analysis (PCA)")
    pca = PCA(0.99)
    X_pca = pca.fit_transform(df_features)
    st.write(f"Reduced features shape after PCA: {X_pca.shape}")

    st.subheader("Train-Test Split and Normalization")
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(
        X_pca, df_target, test_size=0.2, random_state=42)
    st.write(f"Training set: {X_train_pca.shape}, {y_train.shape}")
    st.write(f"Test set: {X_test_pca.shape}, {y_test.shape}")
    scaler = StandardScaler()
    scaler.fit(X_train_pca)
    X_train_scaled = scaler.transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)
    st.write(f"Scaled training set shape: {X_train_scaled.shape}")
    st.write(f"Scaled test set shape: {X_test_scaled.shape}")

elif st.session_state.page == 5:
    st.subheader("Model Building (ANN)")
    # Ensure data from preprocessing
    df = load_data()
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']
    pca = PCA(0.99)
    X_pca = pca.fit_transform(X)
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train_pca)
    X_train_scaled = scaler.transform(X_train_pca)
    X_test_scaled  = scaler.transform(X_test_pca)
    # Build and train model
    model = keras.Sequential([
        keras.layers.Dense(X_train_scaled.shape[1], activation='relu', input_shape=(X_train_scaled.shape[1],)),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(2, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    with st.spinner("Training the model..."):
        history = model.fit(X_train_scaled, y_train, epochs=70, verbose=0)
    st.success("Training completed!")
    st.session_state.model = model
    # Display summary
    st.subheader("Model Summary")
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    st.text("\n".join(summary_lines))
    # Plot training metrics
    st.subheader("Training Metrics")
    hist_df = pd.DataFrame(history.history)
    st.line_chart(hist_df)

elif st.session_state.page == 6:
    st.subheader("Model Evaluation")
    # Recreate test data preprocessing
    if "model" not in st.session_state:
        st.error("You need to train the model first (go to the Model Building page).")
        st.stop()
    model = st.session_state.model
    df = load_data()
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']
    pca = PCA(0.99)
    X_pca = pca.fit_transform(X)
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)

    # Predictions
    preds = model.predict(X_test_scaled)
    preds = (preds > 0.5).astype(int)

    st.subheader("Classification Report")
    report_dict = classification_report(y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    st.subheader("Accuracy Score")
    acc = accuracy_score(y_test, preds)
    st.write(f"Accuracy: {acc:.2f}")

elif st.session_state.page == 7:
    st.subheader("Conclusion")
    st.write(
        "In this analysis, Alzheimer's disease prediction was performed. Exploratory data analysis "
        "provided insights into feature distributions and correlations. PCA reduced feature dimensionality, "
        "followed by training an ANN achieving reasonable accuracy. Further tuning and data expansion "
        "could improve performance."
    )
