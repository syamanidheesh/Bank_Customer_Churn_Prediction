# 05_dashboard.py
# Run with: streamlit run 05_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import os

# Page config
st.set_page_config(
    page_title="Bank Churn Prediction Dashboard",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Customer Churn Prediction & Retention Dashboard")
st.markdown("---")

# Check current directory
st.write(f"Current directory: {os.getcwd()}")
st.write(f"Files in current directory: {os.listdir('.')}")

@st.cache_resource
def load_model():
    """Load the trained ANN model"""
    try:
        # Use 'models/' not '../models/' since we're in the project root
        model_path = "models/best_ann_model.keras"
        st.write(f"Looking for model at: {os.path.abspath(model_path)}")
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.error(f"Files in models directory: {os.listdir('models') if os.path.exists('models') else 'models folder not found'}")
            return None
        model = keras.models.load_model(model_path)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_data():
    """Load all necessary data files"""
    try:
        # Use 'models/' not '../models/'
        X_test_path = "models/X_test_processed.pkl"
        y_test_path = "models/y_test_processed.pkl"
        threshold_path = "models/optimal_threshold.pkl"
        
        if not os.path.exists(X_test_path):
            st.error(f"File not found: {X_test_path}")
            return None, None, None, None, None, None
            
        X_test = joblib.load(X_test_path)
        y_test = joblib.load(y_test_path)
        threshold = joblib.load(threshold_path)
        
        st.success("✅ Data loaded successfully!")
        
        # Try to load optional files
        try:
            strategies = pd.read_csv("models/retention_strategies.csv")
        except:
            strategies = None
            
        try:
            feature_imp = pd.read_csv("models/feature_importance.csv")
        except:
            feature_imp = None
            
        # Load original data for reference
        try:
            df_original = pd.read_csv("data/processed/processed_churn_data.csv")
        except:
            df_original = None
        
        return X_test, y_test, threshold, strategies, feature_imp, df_original
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None

# Load everything
with st.spinner("Loading model and data..."):
    model = load_model()
    X_test, y_test, threshold, strategies, feature_imp, df_original = load_data()

if model is not None and X_test is not None and y_test is not None:
    # Convert to numpy for processing
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    
    # Get predictions
    with st.spinner("Generating predictions..."):
        y_pred_prob = model.predict(X_test).flatten()
        y_pred = (y_pred_prob > threshold).astype(int)
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    risk_filter = st.sidebar.multiselect(
        "Risk Level",
        options=['Low Risk', 'Medium Risk', 'High Risk'],
        default=['High Risk', 'Medium Risk', 'Low Risk']
    )
    
    prob_range = st.sidebar.slider(
        "Churn Probability Range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0)
    )
    
    # Main KPI metrics
    st.subheader("📊 Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Customers", len(X_test))
    with col2:
        st.metric("Predicted Churn", int(sum(y_pred)))
    with col3:
        st.metric("Avg Churn Prob", f"{y_pred_prob.mean():.2%}")
    with col4:
        st.metric("Model Accuracy", f"{accuracy:.2%}")
    with col5:
        st.metric("ROC-AUC", f"{roc_auc:.3f}")
    
    st.markdown("---")
    
    # Model Performance Section
    st.subheader("📈 Model Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Churned', 'Churned'],
                   yticklabels=['Not Churned', 'Churned'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)
    
    with col2:
        # Metrics bar chart
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Score': [accuracy, precision, recall, f1, roc_auc]
        })
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=metrics_df, x='Metric', y='Score', palette='viridis')
        plt.ylim(0, 1)
        plt.title('Model Performance Metrics')
        for i, v in enumerate(metrics_df['Score']):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Feature Importance
    if feature_imp is not None:
        st.subheader("🔑 Top 10 Feature Importance")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = feature_imp.head(10)
            sns.barplot(data=top_features, x='importance', y='feature', palette='rocket')
            plt.xlabel('Mean |SHAP Value|')
            plt.title('Feature Importance Ranking')
            st.pyplot(fig)
        
        with col2:
            st.markdown("### Key Insights:")
            st.markdown("""
            - **Age** is the strongest predictor of churn
            - **Number of Products** significantly impacts retention
            - **Geography** and **Engagement Score** are key factors
            - **Active Member** status strongly influences churn
            """)
    else:
        st.info("Feature importance data not available. Run SHAP analysis first.")
    
    st.markdown("---")
    
    # Risk Distribution
    st.subheader("📊 Customer Risk Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk levels
        risk_levels = []
        for prob in y_pred_prob:
            if prob < 0.3:
                risk_levels.append('Low Risk')
            elif prob < 0.7:
                risk_levels.append('Medium Risk')
            else:
                risk_levels.append('High Risk')
        
        risk_df = pd.DataFrame({'Risk Level': risk_levels})
        risk_counts = risk_df['Risk Level'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = {'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
        color_list = [colors[x] for x in risk_counts.index]
        risk_counts.plot(kind='pie', autopct='%1.1f%%', 
                        colors=color_list, ax=ax)
        plt.title('Customer Risk Segmentation')
        st.pyplot(fig)
    
    with col2:
        # Probability distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(y_pred_prob, bins=30, kde=True, color='blue')
        plt.axvline(0.3, color='green', linestyle='--', label='Low Risk Threshold')
        plt.axvline(0.7, color='red', linestyle='--', label='High Risk Threshold')
        plt.xlabel('Churn Probability')
        plt.ylabel('Count')
        plt.title('Churn Probability Distribution')
        plt.legend()
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Retention Strategies
    if strategies is not None:
        st.subheader("🎯 Personalized Retention Strategies")
        
        # Filter strategies
        filtered_strategies = strategies[
            (strategies['risk_level'].isin(risk_filter))
        ]
        
        # Additional probability filter if column exists
        if 'churn_probability' in filtered_strategies.columns:
            filtered_strategies = filtered_strategies[
                (filtered_strategies['churn_probability'].str.rstrip('%').astype(float) / 100 >= prob_range[0]) &
                (filtered_strategies['churn_probability'].str.rstrip('%').astype(float) / 100 <= prob_range[1])
            ]
        
        st.write(f"Showing {len(filtered_strategies)} of {len(strategies)} strategies")
        
        for idx, row in filtered_strategies.head(10).iterrows():
            with st.expander(f"Customer #{row['customer_id']} - {row['risk_level']} ({row.get('churn_probability', 'N/A')})"):
                strategy_text = row['strategy'].replace('\n', '  \n')
                st.markdown(strategy_text)
        
        # Download button
        st.download_button(
            label="📥 Download All Strategies (CSV)",
            data=strategies.to_csv(index=False),
            file_name="retention_strategies.csv",
            mime="text/csv"
        )
    else:
        st.info("Retention strategies not available. Run the retention strategy generator first.")
    
    st.markdown("---")
    st.success("✅ Dashboard loaded successfully!")

else:
    st.error("❌ Failed to load model or data. Please run the preprocessing and training notebooks first.")
    st.info("Required files should be in the 'models' directory:")
    st.code("""
    models/
    ├── best_ann_model.keras
    ├── X_test_processed.pkl
    ├── y_test_processed.pkl
    ├── optimal_threshold.pkl
    ├── feature_importance.csv (optional)
    └── retention_strategies.csv (optional)
    """)
    
    # Show what's actually in the models directory
    if os.path.exists("models"):
        st.write(f"Files found in models directory: {os.listdir('models')}")
    else:
        st.write("Models directory not found!")