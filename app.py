import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="WDBC ML Classifier Pipeline", layout="wide")
st.title("🩺 Breast Cancer Diagnostic (WDBC) Machine Learning Pipeline")

# Helper to locate local dataset
DATA_FILE = "wdbc.data"

if not os.path.exists(DATA_FILE):
    st.error(f"⚠️ Source file '{DATA_FILE}' was not detected in root directory. Please place it in your workspace repo.")
else:
    # --- 1. Dataset Profiler Module ---
    column_names = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
    df = pd.read_csv(DATA_FILE, header=None, names=column_names)
    
    st.header("📊 Dataset Structural Profile")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Instances (Rows)", df.shape[0])
    col2.metric("Total Features (Columns)", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    
    # Class Distribution Breakdowns
    class_counts = df['diagnosis'].value_counts()
    class_pct = df['diagnosis'].value_counts(normalize=True) * 100
    
    st.subheader("🎯 Target Variable Distribution")
    st.write(f"🟢 **Benign (B)**: {class_counts['B']} samples ({class_pct['B']:.2f}%) | "
             f"🔴 **Malignant (M)**: {class_counts['M']} samples ({class_pct['M']:.2f}%)")

    # --- 2. Data Engineering & Pipeline Preprocessing ---
    df_ml = df.drop(columns=['id'])
    le = LabelEncoder()
    df_ml['diagnosis'] = le.fit_transform(df_ml['diagnosis']) # M -> 1, B -> 0
    
    X = df_ml.drop(columns=['diagnosis'])
    y = df_ml['diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 3. Model Evaluation Initialization ---
    models = {
        "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbor": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results_dict = {}
    
    # Process Metrics
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        results_dict[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC Score": roc_auc_score(y_test, y_prob),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "MCC Score": matthews_corrcoef(y_test, y_pred)
        }
        
    # Present Final Analytics
    st.header("⚡ Core Classification Model Benchmarks")
    performance_df = pd.DataFrame(results_dict).T
    st.dataframe(performance_df.style.highlight_max(axis=0, color="#2E7D32"), use_container_width=True)

    # --- 4. Interactive Live Sample Predictor Playground ---
    st.sidebar.header("🔬 Live Diagnostic Inference Sandbox")
    st.sidebar.write("Adjust mean attributes of cell nuclei to observe structural prediction alterations:")
    
    # Render sliders matching real statistical data bounds
    s_radius = st.sidebar.slider("Mean Radius", 5.0, 30.0, 14.0)
    s_texture = st.sidebar.slider("Mean Texture", 5.0, 40.0, 19.0)
    s_perimeter = st.sidebar.slider("Mean Perimeter", 40.0, 190.0, 92.0)
    s_area = st.sidebar.slider("Mean Area", 140.0, 2500.0, 650.0)
    s_smoothness = st.sidebar.slider("Mean Smoothness", 0.05, 0.20, 0.10)
    
    # Synthesize synthetic 30 feature entry matching dimensions
    synthetic_features = [s_radius, s_texture, s_perimeter, s_area, s_smoothness] + [0.1]*25
    synthetic_df = pd.DataFrame([synthetic_features], columns=X.columns)
    synthetic_scaled = scaler.transform(synthetic_df)
    
    # Select architecture targeting evaluation inference 
    chosen_model_name = st.sidebar.selectbox("Inference Model Target", list(models.keys()))
    active_model = models[chosen_model_name]
    
    # Generate interactive predictions
    live_pred = active_model.predict(synthetic_scaled)[0]
    live_prob = active_model.predict_proba(synthetic_scaled)[0][1]
    
    st.sidebar.subheader("🔮 Predictive Output")
    if live_pred == 1:
        st.sidebar.error(f"**Malignant Diagnosis (Cancerous)**\nConfidence Probability: {live_prob*100:.2f}%")
    else:
        st.sidebar.success(f"**Benign Diagnosis (Healthy)**\nConfidence Probability: {(1-live_prob)*100:.2f}%")
