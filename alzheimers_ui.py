import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# --- Page Setup ---
st.set_page_config(page_title="Alzheimer's Analysis Wizard", layout="wide")
st.title("Alzheimer's Analysis Wizard")

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

# Cached data loader
def load_data():
    return pd.read_csv("alzheimers_disease_data.csv")

# Section logic
if st.session_state.page == 0:
    st.write("**Libraries** like pandas, numpy, matplotlib, seaborn, scikit-learn, etc., are imported at the top of this script.")

elif st.session_state.page == 1:
    df = load_data()
    st.success("Data loaded successfully!")
    if st.checkbox("Show raw data"): st.dataframe(df)

elif st.session_state.page == 2:
    df = load_data()
    st.subheader("Shape and Descriptive Stats")
    st.write(df.shape)
    st.write(df.describe())
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

elif st.session_state.page == 3:
    df = load_data()
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(corr, annot=True, fmt='.1f', linewidths=.5, ax=ax)
    st.pyplot(fig)

elif st.session_state.page == 4:
    df = load_data()
    st.subheader("Pre-processing Placeholder")
    st.write("Implement PCA, train-test split, normalization, etc., here.")

elif st.session_state.page == 5:
    st.subheader("Model Building Placeholder")
    st.write("Build and train your neural network here.")

elif st.session_state.page == 6:
    st.subheader("Model Evaluation Placeholder")
    st.write("Evaluate your model: accuracy, confusion matrix, classification report.")

elif st.session_state.page == 7:
    st.subheader("Conclusion")
    st.write("Summarize findings and next steps.")
