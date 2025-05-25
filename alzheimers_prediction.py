import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

# --- Page Setup ---
st.set_page_config(page_title="Alzheimer's Analysis Wizard", layout="centered")
st.title("Alzheimer's Prediction")

# Define sections
sections = [
    "Load Data",
    "Data Statistics",
    "Data Visualization",
    "Data Pre-processing",
    "Build Model (Neural Network)",
    "Evaluate Model",
    "Conclusion"
]

# Initialize page state
if 'page' not in st.session_state:
    st.session_state.page = 0

# Navigation buttons
cols = st.columns([1, 2, 1])
with cols[0]:
    if st.button("Previous"):
        st.session_state.page = max(0, st.session_state.page - 1)
with cols[2]:
    if st.button("Next"):
        st.session_state.page = min(len(sections) - 1, st.session_state.page + 1)

# Display header
st.header(f"Step {st.session_state.page + 1}: {sections[st.session_state.page]}")

# Cache data loader
def load_data():
    df = pd.read_csv("alzheimers_disease_data.csv")
    df.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)
    return df
load_data = st.cache_data(load_data)

# Feature lists
binary_cols = [
    'Forgetfulness', 'DifficultyCompletingTasks', 'Diagnosis', 'PersonalityChanges',
    'Disorientation', 'Confusion', 'BehavioralProblems', 'MemoryComplaints',
    'Hypertension', 'HeadInjury', 'Depression', 'Diabetes',
    'CardiovascularDisease', 'FamilyHistoryAlzheimers', 'Smoking'
]
num_cols = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality']

# Section logic
page = st.session_state.page
if page == 0:
    df = load_data()
    st.success("Data loaded successfully!")
    st.write("**Data Preview:**")
    st.dataframe(df.head())
    st.write(f"**Dataset shape:** {df.shape}")
    if st.checkbox("Show raw data (all rows)"):
        st.dataframe(df)

elif page == 1:
    df = load_data()
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe())

    st.subheader("Non-Null Counts and Data Types")
    stats_df = pd.DataFrame({
        'Non-Null Count': df.count(),
        'Data Type': df.dtypes.astype(str)
    })
    stats_df.index.name = 'Feature'
    st.dataframe(stats_df)

    st.subheader("Binary Feature Distribution")
    feature = st.selectbox("Select a binary feature to plot", binary_cols)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x=feature, ax=ax, palette="pastel")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_title(f'Distribution of {feature}')
    st.pyplot(fig)

elif page == 2:
    df = load_data()
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".1f",
        linewidths=.5,
        annot_kws={"size": 8},
        cbar_kws={"shrink": .8},
        ax=ax
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig)

    st.subheader("Histograms of Numerical Features")
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(data=df, x=col, kde=True, bins=30, ax=ax)
        ax.set_title(f'Distribution of {col}')
        st.pyplot(fig)

elif page == 3:
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
    st.markdown("""
    - **Features shape** shows number of samples and initial features.
    - **Reduced features shape** indicates components after retaining 99% variance via PCA.
    """)

    st.subheader("Train-Test Split and Normalization")
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(
        X_pca, df_target, test_size=0.2, random_state=42
    )
    st.write(f"Training set shape: {X_train_pca.shape}")
    st.write(f"Test set shape: {X_test_pca.shape}")
    st.markdown(
        "Normalization with `StandardScaler` scales features to zero mean and unit variance, "
        "which helps improve convergence during model training."
    )

elif page == 4:
    df = load_data()
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']
    pca = PCA(0.99)
    X_pca = pca.fit_transform(X)
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)

    st.subheader("Model Building and Training")
    model = keras.Sequential([
        keras.layers.Dense(X_train_scaled.shape[1], activation='relu', input_shape=(X_train_scaled.shape[1],)),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(2, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    with st.spinner("Training model with validation split…"):
        history = model.fit(
            X_train_scaled, y_train,
            epochs=70, validation_split=0.2, verbose=0
        )
    st.success("Training complete!")
    st.session_state.model = model

    st.subheader("Model Summary")
    summary_data = []
    for layer in model.layers:
        # get output shape from tensor
        output_shape = tuple(layer.output.shape)
        summary_data.append([layer.name, output_shape, layer.count_params()])
    summary_df = pd.DataFrame(summary_data, columns=['Layer', 'Output Shape', 'Param #'])
    st.dataframe(summary_df)

    st.subheader("Training History - Accuracy")
    hist_df = pd.DataFrame(history.history)
    acc_cols = ['accuracy'] + (["val_accuracy"] if 'val_accuracy' in hist_df.columns else [])
    st.dataframe(hist_df[acc_cols])

    st.subheader("Training History - Loss")
    loss_cols = ['loss'] + (["val_loss"] if 'val_loss' in hist_df.columns else [])
    st.dataframe(hist_df[loss_cols])

    # Plots
    fig1, ax1 = plt.subplots()
    epochs = range(1, len(history.history['accuracy']) + 1)
    ax1.plot(epochs, history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        ax1.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Accuracy over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    epochs = range(1, len(history.history['loss']) + 1)
    ax2.plot(epochs, history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        ax2.plot(epochs, history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Loss over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    st.pyplot(fig2)

    # Combined validation metrics graph
    fig3, ax3 = plt.subplots()
    val_acc = history.history.get('val_accuracy')
    val_loss = history.history.get('val_loss')

    # Determine epochs based on whichever metric exists
    if val_acc is not None:
        epochs = range(1, len(val_acc) + 1)
    elif val_loss is not None:
        epochs = range(1, len(val_loss) + 1)
    else:
        epochs = []

    # Plot both on the same y-axis
    if val_acc is not None:
        ax3.plot(epochs, val_acc, label='Validation Accuracy')
    if val_loss is not None:
        ax3.plot(epochs, val_loss, label='Validation Loss')

    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Value')
    ax3.set_title('Validation Accuracy & Loss over Epochs')
    ax3.legend()

    st.pyplot(fig3)

elif page == 5:
    # Model Evaluation
    if "model" not in st.session_state:
        st.error("You need to train the model first (go to the Build Model page).")
        st.stop()
    model = st.session_state.model
    df = load_data()
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']
    pca = PCA(0.99)
    X_pca = pca.fit_transform(X)
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    scaler.fit(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)

    # --- Predictions ---
    preds = model.predict(X_test_scaled).ravel()
    preds = (preds > 0.5).astype(int)

    # --- ROC & AUC ---
    probs = model.predict(X_test_scaled).ravel()
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_score = roc_auc_score(y_test, probs)

    st.subheader("ROC Curve")
    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
    ax_roc.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)
    st.write(f"**AUC:** {auc_score:.4f}")

    # --- Classification Report ---
    st.subheader("Classification Report")
    report_dict = classification_report(y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df)

    # --- Confusion Matrix ---
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    # --- Accuracy ---
    st.subheader("Accuracy Score")
    acc = accuracy_score(y_test, preds)
    st.write(f"Accuracy: {acc:.4%}")

    st.session_state["eval_metrics"] = {
        "accuracy": acc,
        "auc":      auc_score,
        "precision": report_dict["1"]["precision"],
        "recall":    report_dict["1"]["recall"],
        "f1":        report_dict["1"]["f1-score"]
    }
else:
    st.subheader("Conclusion")
    metrics = st.session_state.get("eval_metrics")

    if metrics:
        acc   = metrics["accuracy"]
        auc   = metrics["auc"]
        prec  = metrics["precision"]
        rec   = metrics["recall"]
        f1    = metrics["f1"]

        st.write(
            f"In this pipeline, our ANN achieved an overall accuracy of **{acc:.2%}**, "
            f"meaning it correctly classified **{acc:.2%}** of all cases. "
            f"The ROC AUC of **{auc:.2f}** indicates strong discrimination between healthy and diseased subjects across all thresholds. "
            f"For the disease class, a precision of **{prec:.2%}** shows that when the model predicts Alzheimer’s, it is correct **{prec:.2%}** of the time (limiting false positives), "
            f"while a recall of **{rec:.2%}** demonstrates it successfully identifies {rec:.2%} of actual Alzheimer’s cases (minimizing false negatives). "
            f"The F1-score of **{f1:.2%}**, as the harmonic mean of precision and recall, confirms a balanced performance between sensitivity and specificity. "
            "These results underline that our model not only makes accurate predictions, but also maintains reliable detection power for the positive class."
        )
    else:
        st.write("No evaluation metrics found. Please run the Model Evaluation step first.")
