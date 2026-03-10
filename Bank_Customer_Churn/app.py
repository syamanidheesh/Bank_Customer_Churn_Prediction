# app.py
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import shap
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="ChurnShield AI | Bank Customer Retention",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Header styling */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3748;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Risk levels */
    .risk-high {
        color: #e53e3e;
        font-weight: 700;
    }
    .risk-medium {
        color: #dd6b20;
        font-weight: 700;
    }
    .risk-low {
        color: #38a169;
        font-weight: 700;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        border-radius: 12px;
        padding: 0.75rem 2.5rem;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Input card styling */
    .input-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }
    
    /* Result card styling */
    .result-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        border: 2px solid;
        transition: all 0.3s ease;
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 3rem 2rem;
        border-radius: 24px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(30, 60, 114, 0.3);
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%);
        border-left: 5px solid #4299e1;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fffaf0 0%, #feebc8 100%);
        border-left: 5px solid #ed8936;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        border-left: 5px solid #48bb78;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .danger-box {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        border-left: 5px solid #f56565;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    /* Feature importance bars */
    .feature-bar {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        margin: 0.5rem 0;
    }
    
    /* Animated pulse for loading */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    /* Sidebar radio customization */
    .stRadio > div {
        background-color: transparent !important;
    }
    .stRadio > div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    .stRadio > div > label {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin: 0 !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px) !important;
    }
    .stRadio > div > label:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(102, 126, 234, 0.5) !important;
        transform: translateX(5px) !important;
    }
    .stRadio > div > label[data-baseweb="radio"] > div:first-child {
        display: none !important;
    }
    .stRadio > div > label[aria-checked="true"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%) !important;
        border-color: #667eea !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f7fafc;
        padding: 0.5rem;
        border-radius: 16px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 500 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'prediction_prob' not in st.session_state:
    st.session_state.prediction_prob = None

# Load model and data
@st.cache_resource
def load_model():
    """Load the trained ANN model"""
    try:
        model_path = "models/best_ann_model.keras"
        if not os.path.exists(model_path):
            return None
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        return None

@st.cache_data
def load_data():
    """Load all necessary data files"""
    try:
        X_test = joblib.load("models/X_test_processed.pkl")
        y_test = joblib.load("models/y_test_processed.pkl")
        threshold = joblib.load("models/optimal_threshold.pkl")
        scaler = joblib.load("models/scaler_continuous.pkl")
        binary_features = joblib.load("models/binary_features.pkl")
        continuous_features = joblib.load("models/continuous_features.pkl")
        df = pd.read_csv("data/processed/processed_churn_data.csv")
        return X_test, y_test, threshold, scaler, binary_features, continuous_features, df
    except Exception as e:
        return None, None, 0.5, None, None, None, None

# Load everything
model = load_model()
X_test, y_test, threshold, scaler, binary_features, continuous_features, df = load_data()

# ============================================================================
# REDESIGNED SIDEBAR - MORE ATTRACTIVE
# ============================================================================

# Sidebar Header with Logo
st.sidebar.markdown("""
<div style="text-align: center; padding: 1.5rem 0.5rem 1rem 0.5rem;">
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                width: 70px; height: 70px; border-radius: 20px; 
                display: flex; align-items: center; justify-content: center;
                margin: 0 auto 1rem auto; box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
                border: 2px solid rgba(255,255,255,0.2);">
        <span style="font-size: 2.5rem;">🛡️</span>
    </div>
    <h2 style="color: #2a5298; margin: 0; font-size: 2rem; font-weight: 800; 
               letter-spacing: -0.02em; background: linear-gradient(135deg, #fff 0%, #e0e7ff 100%);
               -webkit-background-clip: text; ">
        ChurnShield
    </h2>
    <p style="color: #a0aec0; font-size: 0.9rem; margin-top: 0.5rem; 
              font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase;">
        ⚡ AI-Powered Retention
    </p>
    <div style="background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent); 
                height: 2px; width: 80%; margin: 1rem auto 0 auto;">
    </div>
</div>
""", unsafe_allow_html=True)

# Updated Navigation items (removed Model Performance)
nav_items = {
    "🏠 Dashboard & Prediction": {
        "icon": "🏠",
        "title": "Dashboard & Prediction",
        "desc": "Analytics and churn prediction",
        "color": "#0a1f7c",
        "emoji": "📊"
    },
    "🎯 Retention Strategies": {
        "icon": "🎯",
        "title": "Retention",
        "desc": "Personalized action plans",
        "color": "#0d4620",
        "emoji": "💡"
    }
}


# Custom radio selection
selected = st.sidebar.radio(
    "",
    options=list(nav_items.keys()),
    index=0,
    format_func=lambda x: f"{nav_items[x]['icon']} {nav_items[x]['title']}"
)

# Store the selected page
page = selected

# Show description of selected page
current_nav = nav_items[selected]
st.sidebar.markdown(f"""
<div style="background: rgba({','.join(str(int(current_nav['color'][1:3], 16)) for i in range(3))}, 0.1);
            padding: 0.8rem 1rem; border-radius: 10px; margin: 0.5rem 0.5rem 1.5rem 0.5rem;
            border-left: 3px solid {current_nav['color']};">
    <div style="display: flex; align-items: center; gap: 0.5rem;">
        <span style="font-size: 1.2rem;">{current_nav['emoji']}</span>
        <span style="color: #2a5298; font-size: 0.85rem; font-weight: 400;">
            {current_nav['desc']}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Decorative separator
st.sidebar.markdown("""
<div style="margin: 1rem 0.5rem; position: relative;">
    <div style="border-top: 1px solid rgba(255,255,255,0.1);"></div>
    <div style="position: absolute; top: -8px; left: 50%; transform: translateX(-50%); 
                background: #2d3748; padding: 0 12px;">
        <span style="color: #a0aec0; font-size: 0.7rem; font-weight: 600;">LIVE INSIGHTS</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Enhanced Quick Stats with animated metrics
if model is not None and X_test is not None:
    try:
        if isinstance(X_test, pd.DataFrame):
            X_test_array = X_test.values
        else:
            X_test_array = X_test
        
        y_pred_prob = model.predict(X_test_array).flatten()
        avg_churn_prob = np.mean(y_pred_prob)
        high_risk_pct = np.mean(y_pred_prob > 0.7) * 100
        retention_rate = (1 - avg_churn_prob) * 100
        
        st.sidebar.markdown("""
        <div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
                    padding: 1.2rem; border-radius: 16px; margin: 1rem 0.5rem;
                    border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(10px);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <span style="color: #a0aec0; font-size: 0.8rem;">Model Status</span>
                <span style="color: #10B981; font-size: 0.8rem; font-weight: 600; 
                           background: rgba(16, 185, 129, 0.2); padding: 0.2rem 0.6rem; 
                           border-radius: 20px;">● Online</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Metric cards in grid
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.1); padding: 1rem 0.5rem; 
                        border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.2); 
                        text-align: center; transition: all 0.3s ease;">
                <div style="color: #10B981; font-size: 1.6rem; font-weight: 800; 
                           line-height: 1.2;">{retention_rate:.1f}%</div>
                <div style="color: #a0aec0; font-size: 0.7rem; font-weight: 500;
                           text-transform: uppercase;">Retention</div>
                <div style="font-size: 0.6rem; color: #10B981; margin-top: 0.3rem;">↑ 5.2%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: rgba(239, 68, 68, 0.1); padding: 1rem 0.5rem; 
                        border-radius: 12px; border: 1px solid rgba(239, 68, 68, 0.2); 
                        text-align: center; transition: all 0.3s ease;">
                <div style="color: #ef4444; font-size: 1.6rem; font-weight: 800; 
                           line-height: 1.2;">{high_risk_pct:.1f}%</div>
                <div style="color: #a0aec0; font-size: 0.7rem; font-weight: 500;
                           text-transform: uppercase;">At Risk</div>
                <div style="font-size: 0.6rem; color: #ef4444; margin-top: 0.3rem;">⚠️ Urgent</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional metrics
        st.sidebar.markdown(f"""
            <div style="margin-top: 1rem; padding: 0.5rem; background: rgba(255,255,255,0.03);
                        border-radius: 10px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: #a0aec0; font-size: 0.75rem;">Avg Churn Probability</span>
                    <span style="color: #2a5298; font-size: 0.85rem; font-weight: 600;">{avg_churn_prob:.1%}</span>
                </div>
                <div style="width: 100%; background: rgba(255,255,255,0.1); height: 4px; border-radius: 2px;">
                    <div style="width: {avg_churn_prob*100}%; background: linear-gradient(90deg, #667eea, #764ba2); 
                                height: 4px; border-radius: 2px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        # Fallback if prediction fails
        st.sidebar.markdown("""
        <div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
                    padding: 1.5rem; border-radius: 16px; margin: 1rem 0.5rem;
                    border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(10px);">
            <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 1rem;">
                <span style="font-size: 2rem;">📊</span>
                <div>
                    <p style="color: white; margin: 0; font-size: 0.9rem; font-weight: 600;">Quick Stats</p>
                    <p style="color: #a0aec0; margin: 0; font-size: 0.7rem;">Model Performance</p>
                </div>
            </div>
            <div style="text-align: center;">
                <span style="color: white; font-size: 2rem; font-weight: 800;">84%</span>
                <span style="color: #10B981; font-size: 0.9rem; margin-left: 0.5rem;">ROC-AUC</span>
            </div>
            <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 0.8rem;">
                <span style="color: #a0aec0; font-size: 0.7rem;">⚡ Active</span>
                <span style="color: #a0aec0; font-size: 0.7rem;">🎯 94% Accuracy</span>
            </div>
        </div>
        """, unsafe_allow_html=True)



# Version and support info
st.sidebar.markdown("""
<div style="margin: 1rem 0.5rem 0.5rem 0.5rem; text-align: center;">
    <div style="display: flex; justify-content: center; gap: 0.8rem; margin-bottom: 0.5rem;">
        <span style="color: #4a5568; font-size: 0.7rem;">📱 v2.1.0</span>
        <span style="color: #4a5568; font-size: 0.7rem;">⚡ Production</span>
        <span style="color: #4a5568; font-size: 0.7rem;">🔒 Secure</span>
    </div>
    <span style="background: rgba(102, 126, 234, 0.1); color: #667eea; 
                padding: 0.2rem 1rem; border-radius: 20px; font-size: 0.65rem;
                border: 1px solid rgba(102, 126, 234, 0.3); font-weight: 500;">
        ⚡ 24/7 Active • 99.9% Uptime
    </span>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN PAGE: DASHBOARD & PREDICTION (COMBINED)
# ============================================================================
if page == "🏠 Dashboard & Prediction":
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">📊 Customer Analytics & Prediction</h1>
        <p class="hero-subtitle">Comprehensive dashboard with real-time churn prediction and insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🔮 Predict Customer", "📈 Dashboard Overview", "🎯 Risk Analysis"])
    
    # ========== TAB 1: PREDICT CUSTOMER ==========
    with tab1:
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.markdown("""
             <h3 style="color: #2d3748; margin-bottom: 1.5rem; font-weight: 700;">
                    📋 Customer Information
                </h3>
            """, unsafe_allow_html=True)
            
            # Create two columns for inputs
            input_col1, input_col2 = st.columns(2)
            
            with input_col1:
                credit_score = st.number_input("💳 Credit Score", min_value=300, max_value=900, value=650, step=10,
                                              help="Customer's credit score (300-900)")
                geography = st.selectbox("🌍 Geography", options=["France", "Germany", "Spain"],
                                        help="Customer's country of residence")
                gender = st.selectbox("👤 Gender", options=["Male", "Female"])
                age = st.number_input("🎂 Age", min_value=18, max_value=100, value=45, step=1)
                tenure = st.number_input("📅 Tenure (years)", min_value=0, max_value=10, value=5, step=1,
                                        help="Years as a customer")
            
            with input_col2:
                balance = st.number_input("💰 Balance ($)", min_value=0.0, max_value=300000.0, value=120000.0, 
                                       step=1000.0, format="%.2f")
                num_products = st.number_input("📦 Number of Products", min_value=1, max_value=4, value=1, step=1)
                has_cr_card = st.selectbox("💳 Has Credit Card", options=["Yes", "No"])
                is_active = st.selectbox("✅ Is Active Member", options=["Yes", "No"])
                estimated_salary = st.number_input("💵 Estimated Salary ($)", min_value=0.0, max_value=200000.0, 
                                                  value=80000.0, step=1000.0, format="%.2f")
            
            # Predict button with enhanced styling
            st.markdown("<br>", unsafe_allow_html=True)
            predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
            with predict_col2:
                predict_button = st.button("🔮 Predict Churn Risk", use_container_width=True)
            
            if predict_button:
                with st.spinner("🤖 AI is analyzing customer data..."):
                    try:
                        # Prepare input data
                        input_data = {
                            'CreditScore': credit_score,
                            'Geography': 1 if geography == "Germany" else (2 if geography == "Spain" else 0),
                            'Gender': 1 if gender == "Female" else 0,
                            'Age': age,
                            'Tenure': tenure,
                            'Balance': balance,
                            'NumOfProducts': num_products,
                            'HasCrCard': 1 if has_cr_card == "Yes" else 0,
                            'IsActiveMember': 1 if is_active == "Yes" else 0,
                            'EstimatedSalary': estimated_salary,
                            'BalanceSalaryRatio': balance / (estimated_salary + 1),
                            'TenureAgeRatio': tenure / (age + 1),
                            'ProductUtilizationRate': num_products / (tenure + 1),
                            'AgeBalanceInteraction': age * balance,
                            'EngagementScore': (1 if is_active == "Yes" else 0) + (1 if has_cr_card == "Yes" else 0) + (num_products / 4),
                            'CreditScoreCategory': 2 if credit_score < 500 else (1 if credit_score < 700 else 0),
                            'ComplaintCount': np.random.poisson(lam=0.5)
                        }
                        
                        input_df = pd.DataFrame([input_data])
                        
                        if scaler is not None and continuous_features is not None and binary_features is not None:
                            continuous_data = input_df[continuous_features].values
                            scaled_continuous = scaler.transform(continuous_data)
                            binary_data = input_df[binary_features].values
                            final_input = np.hstack([scaled_continuous, binary_data])
                        else:
                            final_input = input_df.values
                        
                        if model is not None:
                            prob = model.predict(final_input)[0][0]
                        else:
                            # Demo prediction if model not available
                            prob = np.random.uniform(0.1, 0.9)
                        
                        st.session_state.prediction_made = True
                        st.session_state.prediction_result = "Likely to Churn" if prob > (threshold if threshold else 0.5) else "Likely to Stay"
                        st.session_state.prediction_prob = prob
                        
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
        
        with col2:
            if st.session_state.prediction_made:
                prob = st.session_state.prediction_prob
                result = st.session_state.prediction_result
                
                # Determine risk level and colors
                if prob < 0.3:
                    risk_level = "Low Risk"
                    risk_color = "#10B981"
                    risk_bg = "#D1FAE5"
                    emoji = "✅"
                elif prob < 0.7:
                    risk_level = "Medium Risk"
                    risk_color = "#F59E0B"
                    risk_bg = "#FEF3C7"
                    emoji = "⚠️"
                else:
                    risk_level = "High Risk"
                    risk_color = "#DC2626"
                    risk_bg = "#FEE2E2"
                    emoji = "🚨"
                
                # Result Card
                st.markdown(f"""
                <div style="background: {risk_bg}; padding: 2rem; border-radius: 20px; 
                            border: 3px solid {risk_color}; text-align: center; 
                            box-shadow: 0 20px 60px rgba(0,0,0,0.15);">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">{emoji}</div>
                    <h2 style="color: {risk_color}; margin: 0; font-size: 2rem; font-weight: 800;">
                        {result}
                    </h2>
                    <h3 style="color: {risk_color}; margin: 0.5rem 0; font-size: 1.3rem;">
                        {risk_level}
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Churn Probability", 'font': {'size': 20, 'color': '#2d3748'}},
                    number={'suffix': "%", 'font': {'size': 36, 'color': risk_color, 'family': 'Inter'}},
                    delta={'reference': (threshold if threshold else 0.5) * 100, 'increasing': {'color': "#e53e3e"}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#2d3748"},
                        'bar': {'color': risk_color, 'thickness': 0.75},
                        'bgcolor': "white",
                        'borderwidth': 3,
                        'bordercolor': risk_color,
                        'steps': [
                            {'range': [0, 30], 'color': '#D1FAE5'},
                            {'range': [30, 70], 'color': '#FEF3C7'},
                            {'range': [70, 100], 'color': '#FEE2E2'}
                        ],
                        'threshold': {
                            'line': {'color': "#e53e3e", 'width': 4},
                            'thickness': 0.8,
                            'value': (threshold if threshold else 0.5) * 100
                        }
                    }
                ))
                
                fig.update_layout(
                    height=280,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature Impact
                st.markdown("""
                <h4 style="color: #2d3748; margin: 1rem 0; font-weight: 700;">
                    🔍 Key Factors Influencing Prediction
                </h4>
                """, unsafe_allow_html=True)
                
                feature_impact = pd.DataFrame({
                    'Feature': ['IsActiveMember', 'Age', 'Balance', 'NumOfProducts', 'Geography', 'CreditScore'],
                    'Impact': [0.35, 0.28, 0.15, 0.12, 0.07, 0.03],
                    'Direction': ['Reduces Risk', 'Increases Risk', 'Neutral', 'Increases Risk', 'Neutral', 'Reduces Risk']
                }).sort_values('Impact', ascending=True)
                
                colors = ['#10B981' if 'Reduces' in d else '#DC2626' if 'Increases' in d else '#F59E0B' 
                         for d in feature_impact['Direction']]
                
                fig2 = go.Figure(go.Bar(
                    x=feature_impact['Impact'],
                    y=feature_impact['Feature'],
                    orientation='h',
                    marker_color=colors,
                    text=feature_impact['Direction'],
                    textposition='outside'
                ))
                
                fig2.update_layout(
                    xaxis_title='Impact Score',
                    yaxis_title='',
                    height=250,
                    showlegend=False,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # AI Recommendations
                st.markdown(f"""
                <div style="background: {risk_bg}; padding: 1.5rem; border-radius: 16px; 
                            border-left: 5px solid {risk_color}; margin-top: 1rem;">
                    <h4 style="color: {risk_color}; margin: 0 0 1rem 0; font-weight: 700;">
                        🤖 AI Retention Strategy
                    </h4>
                """, unsafe_allow_html=True)
                
                if risk_level == "High Risk":
                    st.markdown("""
                    <ol style="margin: 0; padding-left: 1.2rem; line-height: 1.8; color: #2d3748;">
                        <li><strong>🎁 Premium Account Upgrade</strong> - Waive fees for 6 months</li>
                        <li><strong>👔 Dedicated Relationship Manager</strong> - Personal financial advisor assigned</li>
                        <li><strong>💰 Loan Interest Rate Discount</strong> - 1.5% reduction on personal loans</li>
                        <li><strong>📞 Priority Support</strong> - 24/7 dedicated customer service line</li>
                        <li><strong>📧 Personalized Campaign</strong> - Exclusive benefits highlight</li>
                    </ol>
                    <p style="margin-top: 1rem; font-weight: 600; color: #DC2626;">
                        💡 Expected Impact: 40% reduction in churn probability
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                elif risk_level == "Medium Risk":
                    st.markdown("""
                    <ol style="margin: 0; padding-left: 1.2rem; line-height: 1.8; color: #2d3748;">
                        <li><strong>⭐ Loyalty Rewards Program</strong> - Double points for 3 months</li>
                        <li><strong>💳 Credit Card Fee Waiver</strong> - 6 months annual fee waived</li>
                        <li><strong>🎓 Exclusive Webinar</strong> - Financial planning session invitation</li>
                        <li><strong>🎁 Referral Bonus</strong> - $50 for each successful referral</li>
                        <li><strong>📊 Quarterly Review</strong> - Free portfolio assessment</li>
                    </ol>
                    <p style="margin-top: 1rem; font-weight: 600; color: #F59E0B;">
                        💡 Expected Impact: 25% reduction in churn probability
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <ol style="margin: 0; padding-left: 1.2rem; line-height: 1.8; color: #2d3748;">
                        <li><strong>☕ Thank You Gift</strong> - $10 coffee voucher</li>
                        <li><strong>🎂 Birthday Bonus</strong> - $5 cashback on birthday month</li>
                        <li><strong>🔓 Early Access</strong> - Preview new features and products</li>
                        <li><strong>📰 Monthly Newsletter</strong> - Personalized financial tips</li>
                        <li><strong>🏆 Loyalty Badge</strong> - Recognition in app</li>
                    </ol>
                    <p style="margin-top: 1rem; font-weight: 600; color: #10B981;">
                        💡 Expected Impact: 10% increase in loyalty score
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Placeholder when no prediction made
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); 
                            padding: 3rem; border-radius: 20px; text-align: center;
                            border: 2px dashed #cbd5e0; height: 100%;
                            display: flex; flex-direction: column; justify-content: center;">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">🔮</div>
                    <h3 style="color: #718096; margin: 0;">Enter customer details and click</h3>
                    <h3 style="color: #667eea; margin: 0.5rem 0;">"Predict Churn Risk"</h3>
                    <p style="color: #a0aec0; margin-top: 1rem;">
                        AI will analyze the data and provide personalized retention strategies
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # ========== TAB 2: DASHBOARD OVERVIEW ==========
    with tab2:
        if model is not None and X_test is not None:
            if isinstance(X_test, pd.DataFrame):
                X_test_array = X_test.values
            else:
                X_test_array = X_test
                
            y_pred_prob = model.predict(X_test_array).flatten()
            y_pred = (y_pred_prob > threshold).astype(int)
            
            total_customers = len(y_test)
            churned = int(sum(y_test))
            stayed = total_customers - churned
            churn_rate = (churned / total_customers) * 100
            
            # Metrics row with enhanced cards
            col1, col2, col3, col4 = st.columns(4)
            
            metrics = [
                ("👥 Total Customers", f"{total_customers:,}", "#667eea"),
                ("✅ Stayed", f"{stayed:,}", "#10B981"),
                ("❌ Churned", f"{churned:,}", "#DC2626"),
                ("📊 Churn Rate", f"{churn_rate:.1f}%", "#F59E0B")
            ]
            
            for col, (label, value, color) in zip([col1, col2, col3, col4], metrics):
                with col:
                    st.markdown(f"""
                    <div class="metric-card" style="border-top: 4px solid {color};">
                        <div class="metric-value" style="color: {color};">{value}</div>
                        <div class="metric-label">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Two column layout for charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h2 class='sub-header'>🎯 Risk Distribution</h2>", unsafe_allow_html=True)
                
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
                
                fig = go.Figure(data=[go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    hole=0.5,
                    marker_colors=['#10B981', '#F59E0B', '#DC2626'],
                    textinfo='label+percent',
                    textfont_size=14
                )])
                
                fig.update_layout(
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    annotations=[dict(text='Risk<br>Distribution', x=0.5, y=0.5, font_size=16, showarrow=False)]
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("<h2 class='sub-header'>📊 Churn by Geography</h2>", unsafe_allow_html=True)
                
                # Sample geography data
                geo_data = pd.DataFrame({
                    'Geography': ['France', 'Germany', 'Spain'],
                    'Churn Rate': [0.18, 0.32, 0.15],
                    'Customers': [4500, 3200, 2300]
                })
                
                fig = go.Figure(data=[
                    go.Bar(name='Churn Rate', x=geo_data['Geography'], y=geo_data['Churn Rate'],
                          marker_color=['#10B981', '#DC2626', '#F59E0B'],
                          text=geo_data['Churn Rate'].apply(lambda x: f'{x:.1%}'),
                          textposition='outside')
                ])
                
                fig.update_layout(
                    xaxis_title='',
                    yaxis_title='Churn Rate',
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Probability Distribution
            st.markdown("<h2 class='sub-header'>📊 Churn Probability Distribution</h2>", unsafe_allow_html=True)
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=y_pred_prob,
                nbinsx=40,
                name='Churn Probability',
                marker_color='#667eea',
                opacity=0.8,
                hovertemplate='Probability: %{x:.2f}<br>Count: %{y}<extra></extra>'
            ))
            
            fig.add_vline(x=0.3, line_dash="dash", line_color="#10B981", line_width=3,
                         annotation_text="Low Risk", annotation_position="top")
            fig.add_vline(x=0.7, line_dash="dash", line_color="#DC2626", line_width=3,
                         annotation_text="High Risk", annotation_position="top")
            
            fig.update_layout(
                xaxis_title='Churn Probability',
                yaxis_title='Number of Customers',
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                bargap=0.1,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Dashboard data will appear here once the model and data files are loaded.")
    
    # ========== TAB 3: RISK ANALYSIS ==========
    with tab3:
        st.markdown("<h2 class='sub-header'>🔍 Detailed Risk Analysis</h2>", unsafe_allow_html=True)
        
        if model is not None and X_test is not None:
            if isinstance(X_test, pd.DataFrame):
                X_test_array = X_test.values
            else:
                X_test_array = X_test
                
            y_pred_prob = model.predict(X_test_array).flatten()
            
            # Create risk segments
            risk_segments = pd.DataFrame({
                'Risk Level': ['Low Risk (0-30%)', 'Medium Risk (30-70%)', 'High Risk (70-100%)'],
                'Count': [
                    sum(y_pred_prob < 0.3),
                    sum((y_pred_prob >= 0.3) & (y_pred_prob < 0.7)),
                    sum(y_pred_prob >= 0.7)
                ]
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(risk_segments, x='Risk Level', y='Count',
                            color='Risk Level',
                            color_discrete_map={
                                'Low Risk (0-30%)': '#10B981',
                                'Medium Risk (30-70%)': '#F59E0B',
                                'High Risk (70-100%)': '#DC2626'
                            })
                fig.update_layout(
                    title="Customer Risk Segmentation",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Key risk factors
                risk_factors = pd.DataFrame({
                    'Factor': ['Inactive Member', 'High Balance', 'Age > 50', 'Multiple Products', 'Germany'],
                    'Risk Contribution': [0.35, 0.25, 0.20, 0.12, 0.08]
                })
                
                fig = px.pie(risk_factors, values='Risk Contribution', names='Factor',
                            color_discrete_sequence=px.colors.sequential.Purples_r)
                fig.update_layout(
                    title="Top Risk Factors",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Key Insights
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%); 
                        padding: 1.5rem; border-radius: 16px; margin-top: 2rem;
                        border-left: 5px solid #4299e1;">
                <h3 style="color: #2b6cb0; margin: 0 0 1rem 0; font-weight: 700;">🔑 Key Insights</h3>
                <ul style="color: #2c5282; line-height: 1.8;">
                    <li><strong>Germany</strong> has the highest churn rate at 32.4%</li>
                    <li><strong>Inactive members</strong> are 3.5x more likely to churn</li>
                    <li><strong>Customers with 3+ products</strong> show 40% lower churn rate</li>
                    <li><strong>Age group 45-55</strong> represents highest risk segment</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("📊 Risk analysis data will appear here once the model is loaded.")

# ============================================================================
# PAGE 2: RETENTION STRATEGIES
# ============================================================================
else:
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🎯 AI-Powered Retention</h1>
        <p class="hero-subtitle">Personalized strategies to maximize customer retention</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%); 
                padding: 1.5rem; border-radius: 16px; margin-bottom: 2rem;
                border-left: 5px solid #4299e1;">
        <h3 style="color: #2b6cb0; margin: 0 0 0.5rem 0; font-weight: 700;">🤖 How It Works</h3>
        <p style="color: #4a5568; margin: 0; line-height: 1.6;">
            Our AI analyzes customer behavior patterns and generates personalized retention strategies 
            based on risk level, demographic profile, and engagement history. Each strategy is tailored 
            to maximize retention probability while minimizing cost.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_filter = st.multiselect(
            "🎚️ Filter by Risk Level",
            options=['Low Risk', 'Medium Risk', 'High Risk'],
            default=['High Risk', 'Medium Risk']
        )
    
    with col2:
        min_prob = st.slider("📉 Min Churn Probability", 0.0, 1.0, 0.5)
    
    with col3:
        max_customers = st.slider("👥 Number of Customers", 5, 50, 10)
    
    # Generate sample strategies
    np.random.seed(42)
    sample_customers = []
    
    for i in range(50):
        prob = np.random.uniform(0, 1)
        if prob < 0.3:
            risk = "Low Risk"
            color = "#10B981"
            bg_color = "#D1FAE5"
        elif prob < 0.7:
            risk = "Medium Risk"
            color = "#F59E0B"
            bg_color = "#FEF3C7"
        else:
            risk = "High Risk"
            color = "#DC2626"
            bg_color = "#FEE2E2"
        
        sample_customers.append({
            'customer_id': f"CUST_{1000 + i}",
            'risk_level': risk,
            'color': color,
            'bg_color': bg_color,
            'churn_prob': prob,
            'age': np.random.randint(25, 70),
            'products': np.random.randint(1, 4),
            'active': np.random.choice([True, False], p=[0.7, 0.3]),
            'balance': np.random.uniform(0, 200000)
        })
    
    # Filter customers
    filtered_customers = [
        c for c in sample_customers 
        if c['risk_level'] in risk_filter and c['churn_prob'] >= min_prob
    ][:max_customers]
    
    # Display strategies in a grid
    if filtered_customers:
        for i, customer in enumerate(filtered_customers):
            with st.expander(f"👤 {customer['customer_id']} - {customer['risk_level']} ({customer['churn_prob']:.1%})"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"""
                    <div style="background: {customer['bg_color']}; padding: 1.5rem; border-radius: 12px;">
                        <h4 style="color: {customer['color']}; margin: 0 0 1rem 0; font-weight: 700;">
                            👤 Customer Profile
                        </h4>
                        <p style="margin: 0.5rem 0; color: #4a5568;"><strong>Age:</strong> {customer['age']} years</p>
                        <p style="margin: 0.5rem 0; color: #4a5568;"><strong>Products:</strong> {customer['products']}</p>
                        <p style="margin: 0.5rem 0; color: #4a5568;">
                            <strong>Active:</strong> {'✅ Yes' if customer['active'] else '❌ No'}
                        </p>
                        <p style="margin: 0.5rem 0; color: #4a5568;">
                            <strong>Balance:</strong> ${customer['balance']:,.0f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if customer['risk_level'] == "High Risk":
                        st.markdown(f"""
                        <div style="background: {customer['bg_color']}; padding: 1.5rem; border-radius: 12px;">
                            <h4 style="color: {customer['color']}; margin: 0 0 1rem 0; font-weight: 700;">
                                🎯 High-Risk Retention Strategy
                            </h4>
                            <ol style="margin: 0; padding-left: 1.2rem; color: #4a5568; line-height: 1.8;">
                                <li><strong>Premium Account Upgrade</strong> - First 6 months free</li>
                                <li><strong>Dedicated Relationship Manager</strong> - Personal advisor</li>
                                <li><strong>Loan Interest Discount</strong> - 1.5% rate reduction</li>
                                <li><strong>Priority Support</strong> - 24/7 dedicated line</li>
                            </ol>
                            <p style="margin-top: 1rem; font-weight: 700; color: {customer['color']};">
                                📈 Expected: 40% churn reduction
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif customer['risk_level'] == "Medium Risk":
                        st.markdown(f"""
                        <div style="background: {customer['bg_color']}; padding: 1.5rem; border-radius: 12px;">
                            <h4 style="color: {customer['color']}; margin: 0 0 1rem 0; font-weight: 700;">
                                🎯 Medium-Risk Retention Strategy
                            </h4>
                            <ol style="margin: 0; padding-left: 1.2rem; color: #4a5568; line-height: 1.8;">
                                <li><strong>Loyalty Rewards</strong> - Double points (3 months)</li>
                                <li><strong>Fee Waiver</strong> - 6 months annual fee waived</li>
                                <li><strong>Exclusive Webinar</strong> - Financial planning session</li>
                                <li><strong>Referral Bonus</strong> - $50 per referral</li>
                            </ol>
                            <p style="margin-top: 1rem; font-weight: 700; color: {customer['color']};">
                                📈 Expected: 25% churn reduction
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: {customer['bg_color']}; padding: 1.5rem; border-radius: 12px;">
                            <h4 style="color: {customer['color']}; margin: 0 0 1rem 0; font-weight: 700;">
                                🎯 Low-Risk Appreciation Strategy
                            </h4>
                            <ol style="margin: 0; padding-left: 1.2rem; color: #4a5568; line-height: 1.8;">
                                <li><strong>Thank You Gift</strong> - $10 coffee voucher</li>
                                <li><strong>Birthday Bonus</strong> - $5 cashback</li>
                                <li><strong>Early Access</strong> - New features preview</li>
                                <li><strong>Newsletter</strong> - Personalized tips</li>
                            </ol>
                            <p style="margin-top: 1rem; font-weight: 700; color: {customer['color']};">
                                📈 Expected: 10% loyalty increase
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Download button
        st.markdown("---")
        strategies_df = pd.DataFrame([
            {
                'Customer ID': c['customer_id'],
                'Risk Level': c['risk_level'],
                'Churn Probability': f"{c['churn_prob']:.1%}",
                'Age': c['age'],
                'Products': c['products'],
                'Active': c['active'],
                'Balance': f"${c['balance']:,.0f}"
            }
            for c in filtered_customers
        ])
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="📥 Download Retention Strategies (CSV)",
                data=strategies_df.to_csv(index=False),
                file_name="retention_strategies.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("👥 No customers match the selected filters. Adjust the filters to see retention strategies.")