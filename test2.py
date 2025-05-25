import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

# ----------------------------------------
# Data Loading and Caching
# ----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('alzheimers_disease_data.csv')
    # Drop unused columns as in original analysis
    if {'PatientID', 'DoctorInCharge'}.issubset(df.columns):
        df.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)
    return df

df = load_data()

@st.cache_data
def preprocess(df):
    # Extract features and target
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis'].copy()
    # PCA to retain 99% variance
    pca = PCA(0.99)
    X_pca = pca.fit_transform(X)
    # Train-test split
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)
    return X_train_scaled, X_test_scaled, y_train, y_test, pca

# ----------------------------------------
# Streamlit Application
# ----------------------------------------
def main():
    st.title("Alzheimer's Disease Analysis")
    pages = [
        'Data Statistics',
        'Data Pre-processing',
        'Build Model & Training',
        'Evaluate Model',
        'Conclusion'
    ]
    choice = st.sidebar.radio('Navigation', pages)

    if choice == 'Data Statistics':
        st.header('Data Statistics')
        st.write('Dataset shape:', df.shape)
        # Show info
        buf = io.StringIO()
        df.info(buf=buf)
        st.text(buf.getvalue())
        # Missing values
        st.subheader('Missing Values per Column')
        st.write(df.isnull().sum())
        # Interactive countplot for categorical/binary columns
        cat_cols = [c for c in df.columns if df[c].nunique() <= 10 and c != 'Diagnosis']
        feature = st.selectbox('Select feature to plot', cat_cols)
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=feature, palette='pastel', ax=ax)
        ax.set_title(f'Distribution of {feature}')
        st.pyplot(fig)

    elif choice == 'Data Pre-processing':
        st.header('Data Pre-processing')
        X_train_scaled, X_test_scaled, y_train, y_test, pca = preprocess(df)
        st.write('PCA retained 99% variance reducing to', X_train_scaled.shape[1], 'components')
        st.write('Training set shape:', X_train_scaled.shape)
        st.write('Test set shape:', X_test_scaled.shape)

    elif choice == 'Build Model & Training':
        st.header('Build Model & Training')
        X_train_scaled, X_test_scaled, y_train, y_test, _ = preprocess(df)
        # Define model as in original analysis (8→4→2→1)
        model = keras.Sequential([
            keras.layers.Dense(X_train_scaled.shape[1], activation='relu', input_shape=(X_train_scaled.shape[1],)),
            keras.layers.Dense(4, activation='relu'),
            keras.layers.Dense(2, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        # Train for 70 epochs
        history = model.fit(
            X_train_scaled, y_train,
            epochs=70,
            validation_split=0.2,
            verbose=0
        )
        # Plot accuracy (validation if available)
        epochs = list(range(1, 71))
        fig1, ax1 = plt.subplots()
        if 'val_accuracy' in history.history:
            ax1.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title('Validation Accuracy over Epochs')
        else:
            ax1.plot(epochs, history.history['accuracy'], label='Accuracy')
            ax1.set_title('Accuracy over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        st.pyplot(fig1)
        # Plot loss (validation if available)
        fig2, ax2 = plt.subplots()
        if 'val_loss' in history.history:
            ax2.plot(epochs, history.history['val_loss'], label='Validation Loss')
            ax2.set_title('Validation Loss over Epochs')
        else:
            ax2.plot(epochs, history.history['loss'], label='Loss')
            ax2.set_title('Loss over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        st.pyplot(fig2)
        # Store trained model for evaluation
        st.session_state['model'] = model

    elif choice == 'Evaluate Model':
        st.header('Evaluate Model')
        if 'model' not in st.session_state:
            st.warning('Please train the model in "Build Model & Training" first.')
        else:
            model = st.session_state['model']
            X_train_scaled, X_test_scaled, y_train, y_test, _ = preprocess(df)
            y_pred_prob = model.predict(X_test_scaled)
            y_pred = (y_pred_prob > 0.5).astype(int)
            st.subheader('Classification Report')
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            st.subheader('Confusion Matrix')
            cm = confusion_matrix(y_test, y_pred)
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='pastel', ax=ax3)
            ax3.set_xlabel('Predicted')
            ax3.set_ylabel('Actual')
            st.pyplot(fig3)

    else:
        st.header('Conclusion')
        st.write(
            "In this Kaggle Notebook, Alzheimer's disease is predicted using an ANN with PCA preprocessing. "
            "Model performance metrics and visualizations are provided above."
        )

if __name__ == '__main__':
    main()
