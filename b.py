import json
import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load models and scaler
@st.cache_resource
def load_models():
    models = {}
    model_paths = [
        "C:/Users/golla/OneDrive/Desktop/internships/project/streamlit/Random_Forest_model.pkl",
        "C:/Users/golla/OneDrive/Desktop/internships/project/streamlit/Support_Vector_Machine_model.pkl",
        "C:/Users/golla/OneDrive/Desktop/internships/project/streamlit/XGBoost_model.pkl",
        "C:/Users/golla/OneDrive/Desktop/internships/project/streamlit/K-Nearest_Neighbors_model.pkl",
        "C:/Users/golla/OneDrive/Desktop/internships/project/streamlit/Logistic_Regression_model.pkl"
    ]

    for model_path in model_paths:
        model_name = model_path.split("_model")[0].split("/")[-1].replace("_", " ")
        with open(model_path, 'rb') as model_file:
            models[model_name] = pickle.load(model_file)

    with open('C:/Users/golla/OneDrive/Desktop/internships/project/streamlit/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    return models, scaler

# Load metrics from JSON file
@st.cache_resource
def load_metrics():
    with open('C:/Users/golla/OneDrive/Desktop/internships/project/streamlit/model_metrics.json', 'r') as metrics_file:
        return json.load(metrics_file)

# Load models, scaler, and metrics
models, scaler = load_models()
model_metrics = load_metrics()

# Select the best model based on F1 score
best_model_name = max(model_metrics, key=lambda name: model_metrics[name]["f1_score"])
best_model = models[best_model_name]

# Streamlit App
st.title("🎗️ Breast Cancer Diagnosis Prediction")
st.markdown(
    """
    This application predicts whether a tumor is **benign** or **malignant** 
    using the best performing machine learning model.
    """
)

# Sidebar: Input features for prediction
st.sidebar.header("🔢 Input Features")
radius_mean = st.sidebar.number_input("Radius Mean", min_value=0.0, max_value=100.0, value=0.0)
texture_mean = st.sidebar.number_input("Texture Mean", min_value=0.0, max_value=100.0, value=0.0)
perimeter_mean = st.sidebar.number_input("Perimeter Mean", min_value=0.0, max_value=1000.0, value=0.0)
area_mean = st.sidebar.number_input("Area Mean", min_value=0.0, max_value=10000.0, value=0.0)
smoothness_mean = st.sidebar.number_input("Smoothness Mean", min_value=0.0, max_value=1.0, value=0.0)
compactness_mean = st.sidebar.number_input("Compactness Mean", min_value=0.0, max_value=1.0, value=0.0)
concavity_mean = st.sidebar.number_input("Concavity Mean", min_value=0.0, max_value=1.0, value=0.0)
concave_points_mean = st.sidebar.number_input("Concave Points Mean", min_value=0.0, max_value=1.0, value=0.0)

# Predict button for best model
if st.sidebar.button("Predict"):
    input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean,
                            smoothness_mean, compactness_mean, concavity_mean, concave_points_mean]])
    input_data_scaled = scaler.transform(input_data)
    prediction = best_model.predict(input_data_scaled)

    st.sidebar.subheader("Prediction Result:")
    result_text = "Malignant" if prediction[0] == 1 else "Benign"
    st.sidebar.success(f"Prediction by Best Model: {result_text}")

# Tabs for analysis
tabs = st.tabs(["Model Scores", "Confusion Matrices", "Best Model", "Data Visualizations", "Histograms", "Model Accuracies"])

# Tab: Model Scores
with tabs[0]:
    st.header("📊 Model Scores")
    for name, metrics in model_metrics.items():
        st.write(f"**{name}**")
        st.write(f"Accuracy: {metrics['accuracy']:.2f}")
        st.write(f"F1 Score: {metrics['f1_score']:.2f}")
        st.write(f"Precision: {metrics['precision']:.2f}")
        st.write(f"Recall: {metrics['recall']:.2f}")
        st.write("---")

# Tab: Confusion Matrices
with tabs[1]:
    st.header("📊 Confusion Matrices")
    for name, metrics in model_metrics.items():
        st.subheader(f"{name}")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix - {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

# Tab: Best Model
with tabs[2]:
    st.header("🏆 Best Model")
    st.write(f"**Best Model:** {best_model_name}")
    st.write(f"F1 Score: {model_metrics[best_model_name]['f1_score']:.2f}")
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(model_metrics[best_model_name]["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {best_model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# Tab: Data Visualizations
with tabs[3]:
    st.header("📊 Data Visualizations")

    # Load and preprocess the dataset
    @st.cache_resource
    def load_data():
        data_path = 'C:/Users/golla/OneDrive/Desktop/internships/project/streamlit/data.csv'
        data = pd.read_csv(data_path)
        
        # Drop unnecessary columns if present
        data = data.drop(columns=['Unnamed: 32'], errors='ignore')

        # Define feature columns
        columns_to_use = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
            'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean'
        ]
        features = data[columns_to_use]
        diagnosis = data['diagnosis']

        # Scale the data using the loaded scaler
        scaled_features = scaler.transform(features)
        scaled_features_df = pd.DataFrame(scaled_features, columns=columns_to_use)

        return features, scaled_features_df, diagnosis

    # Load unscaled, scaled features, and diagnosis
    unscaled_features, scaled_features, diagnosis = load_data()

    # Combined Boxplot: Unscaled Data
    st.subheader("Combined Boxplot (Unscaled Data)")
    unscaled_melted = unscaled_features.melt(var_name="Feature", value_name="Value")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x="Feature", y="Value", data=unscaled_melted, ax=ax, palette="Set2")
    ax.set_title("Combined Boxplot of Features (Unscaled Data)", fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    st.pyplot(fig)

    # Combined Boxplot: Scaled Data
    st.subheader("Combined Boxplot (Scaled Data)")
    scaled_melted = scaled_features.melt(var_name="Feature", value_name="Value")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x="Feature", y="Value", data=scaled_melted, ax=ax, palette="Set3")
    ax.set_title("Combined Boxplot of Features (Scaled Data)", fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    st.pyplot(fig)

# Tab: Histograms
with tabs[4]:
    st.header("📊 Histograms")

    # Unscaled Histograms
    st.subheader("Histograms (Unscaled Data)")
    for column in unscaled_features.columns:
        st.write(f"**{column}**")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(unscaled_features[column], bins=30, kde=True, ax=ax)
        ax.set_title(f"Histogram of {column} (Unscaled)")
        st.pyplot(fig)

    # Scaled Histograms
    st.subheader("Histograms (Scaled Data)")
    for column in scaled_features.columns:
        st.write(f"**{column}**")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(scaled_features[column], bins=30, kde=True, ax=ax)
        ax.set_title(f"Histogram of {column} (Scaled)")
        st.pyplot(fig)

with tabs[5]:
    st.header("📈 Model Accuracies")
    model_names = list(model_metrics.keys())
    accuracies = [model_metrics[name]['accuracy'] for name in model_names]

    # Create a bar plot for model accuracies
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=model_names, y=accuracies, palette="coolwarm", ax=ax)
    ax.set_title("Comparison of Model Accuracies", fontsize=16)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylim(0, 1)  # Assuming accuracy is between 0 and 1
    ax.bar_label(ax.containers[0], fmt='%.2f')  # Add accuracy values on top of bars
    plt.xticks(rotation=45, horizontalalignment='right')
    st.pyplot(fig)

    # Display accuracy values explicitly below the chart
    st.write("### Model Accuracy Values:")
    for model_name, accuracy in zip(model_names, accuracies):
        st.write(f"**{model_name}:** {accuracy:.2f}")

